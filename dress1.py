import cv2
import face_recognition
import numpy as np
import os
import re
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import logging
from pathlib import Path
from sklearn.cluster import KMeans
import json
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CameraManager:
    def __init__(self, camera_id: str, camera_index: int):
        self.camera_id = camera_id
        self.camera_index = camera_index
        self.cap = None
        self.last_connection_attempt = datetime.min
        self.connection_attempts = 0
        self.max_attempts = 3  # Reduced from 5 to prevent excessive retries
        self.retry_delay = 1  # Reduced delay to 1 second
        self.last_frame = None
        self.last_frame_time = datetime.min
        self.frame_timeout = 2  # seconds before considering camera disconnected

    async def connect(self) -> bool:
        """Try to connect to the camera"""
        if self.connection_attempts >= self.max_attempts:
            logger.warning(f"Max connection attempts reached for {self.camera_id}")
            return False

        if (datetime.now() - self.last_connection_attempt).total_seconds() < self.retry_delay:
            return False

        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera {self.camera_id}")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduced resolution for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 15)  # Reduced FPS for better stability
            
            self.connection_attempts = 0
            self.last_connection_attempt = datetime.now()
            logger.info(f"Successfully connected to {self.camera_id}")
            return True
        except Exception as e:
            self.connection_attempts += 1
            self.last_connection_attempt = datetime.now()
            logger.error(f"Failed to connect to {self.camera_id}: {str(e)}")
            return False

    async def read_frame(self) -> Optional[np.ndarray]:
        """Read a frame from the camera"""
        if not self.cap or not self.cap.isOpened():
            if not await self.connect():
                return None

        try:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning(f"Failed to read frame from {self.camera_id}")
                return None
            
            # Check if frame is valid
            if frame is None or frame.size == 0:
                logger.warning(f"Invalid frame received from {self.camera_id}")
                return None
            
            self.last_frame = frame
            self.last_frame_time = datetime.now()
            return frame
        except Exception as e:
            logger.error(f"Error reading frame from {self.camera_id}: {str(e)}")
            return None

    async def release(self):
        """Release the camera"""
        if self.cap:
            try:
                self.cap.release()
                logger.info(f"Camera {self.camera_id} released")
            except Exception as e:
                logger.error(f"Error releasing {self.camera_id}: {str(e)}")

    def is_connected(self) -> bool:
        """Check if camera is still connected"""
        if not self.cap or not self.cap.isOpened():
            return False
        if (datetime.now() - self.last_frame_time).total_seconds() > self.frame_timeout:
            logger.warning(f"No frame received from {self.camera_id} for {self.frame_timeout} seconds")
            return False
        return True

class AsyncShopCCTVSystem:
    def __init__(self):
        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_names: List[str] = []
        self.known_face_info: Dict[str, str] = {
            "Staff": "Shop Staff",
            "Customer": "Shop Customer"
        }
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        self.model = None
        self._initialize_system()
        self.tracking_data = defaultdict(dict)
        self.motion_threshold = 25
        self.min_motion_area = 500
        self.last_motion = datetime.now()
        self.motion_timeout = 5  # seconds
        
        # Configuration for long-distance detection
        self.face_detection_scale = 1.0
        self.face_distance_threshold = 0.5
        self.min_face_size = 50

    async def _initialize_system(self):
        """Initialize the system"""
        # Load known faces
        celebs_dir = Path("/Users/amritpal/celebrity_face_recognition/celebs")
        for file in celebs_dir.glob('*'):
            if not file.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                continue
            try:
                image = face_recognition.load_image_file(str(file))
                encoding = face_recognition.face_encodings(image)
                if encoding:
                    self.known_face_encodings.append(encoding[0])
                    name = self._clean_name(file.stem)
                    self.known_face_names.append(name)
            except Exception as e:
                logger.warning(f"Failed to process face from {file}: {str(e)}")

        # Initialize dress classification model
        model_path = Path("dress_model.h5")
        try:
            if model_path.exists():
                logger.info("Loading pre-trained dress model from disk")
                self.model = keras.models.load_model(str(model_path))
            else:
                logger.info("Creating and training new dress model")
                self.model = self._create_model()
                fashion_mnist = keras.datasets.fashion_mnist
                (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
                
                train_images = train_images / 255.0
                test_images = test_images / 255.0
                train_images = train_images[..., np.newaxis]
                test_images = test_images[..., np.newaxis]

                self.model.fit(train_images, train_labels, epochs=5, verbose=1)
                
                # Save the model
                self.model.save('dress_model.h5')
                logger.info("Dress model trained and saved to disk")
        except Exception as e:
            logger.error(f"Error initializing dress model: {str(e)}")
            raise

    def _clean_name(self, filename: str) -> str:
        """Clean and format the name from filename"""
        name = filename.replace("_", " ").replace(",", " ")
        name = re.sub(r"\([^)]*\)", "", name)
        name = re.sub(r"\d+", "", name)
        name = re.sub(r"\s+", " ", name).strip()
        return name.title()

    def _create_model(self) -> tf.keras.Model:
        """Create and save the dress classification model"""
        model_path = Path("dress_model.h5")
        if model_path.exists():
            logger.info("Loading pre-trained dress model from disk")
            return keras.models.load_model(str(model_path))
            
        logger.info("Creating and training new dress model")
        model = keras.Sequential([
            keras.layers.InputLayer(input_shape=(28, 28, 1)),
            keras.layers.Conv2D(32, (3,3), activation='relu'),
            keras.layers.MaxPooling2D(2,2),
            keras.layers.Conv2D(64, (3,3), activation='relu'),
            keras.layers.MaxPooling2D(2,2),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        return model

    def _get_dominant_color(self, image: np.ndarray, k: int = 3) -> Tuple[int, int, int]:
        """Get dominant color from image"""
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_rgb = img_rgb.reshape((-1, 3))
        
        clt = KMeans(n_clusters=k)
        clt.fit(img_rgb)
        counts = np.bincount(clt.labels_)
        dominant_color = clt.cluster_centers_[np.argmax(counts)]
        return dominant_color.astype(int)

    def _color_name(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB to color name"""
        r, g, b = rgb
        colors = {
            "Red": (r > 150 and g < 100 and b < 100),
            "Green": (r < 100 and g > 150 and b < 100),
            "Blue": (r < 100 and g < 100 and b > 150),
            "White": (r > 150 and g > 150 and b > 150),
            "Black": (r < 50 and g < 50 and b < 50),
            "Gray": (abs(r - g) < 30 and abs(g - b) < 30 and abs(r - b) < 30),
            "Brown": (r > 100 and g > 50 and g < 150 and b < 100),
            "Yellow": (r > 150 and g > 100 and b < 50),
            "Purple": (r > 100 and g < 50 and b > 100),
            "Pink": (r > 150 and g < 100 and b > 100)
        }
        
        for name, condition in colors.items():
            if condition:
                return name
        return "Unknown"

    def _detect_motion(self, frame: np.ndarray, prev_frame: np.ndarray) -> bool:
        """Detect motion between frames"""
        if prev_frame is None:
            return False

        # Convert frames to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        # Calculate absolute difference
        frame_diff = cv2.absdiff(gray, prev_gray)
        
        # Threshold to get binary image
        _, thresh = cv2.threshold(frame_diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check if any contour is larger than minimum area
        for contour in contours:
            if cv2.contourArea(contour) > self.min_motion_area:
                return True
        
        return False

    async def process_frame(self, frame: np.ndarray, camera_id: str) -> Tuple[np.ndarray, dict]:
        """Process a single frame from a camera"""
        current_time = datetime.now()
        
        # Check for motion
        if camera_id in self.tracking_data:
            prev_frame = self.tracking_data[camera_id].get('prev_frame')
            if not self._detect_motion(frame, prev_frame):
                return frame, {}

        # Process frame at full resolution
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all face locations and encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        detected_faces = []

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            face_width = right - left
            face_height = bottom - top
            if face_width < self.min_face_size or face_height < self.min_face_size:
                continue

            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

            name = "Unknown"
            info = ""

            if True in matches:
                best_match_index = np.argmin(face_distances)
                if face_distances[best_match_index] < self.face_distance_threshold:
                    name = self.known_face_names[best_match_index]
                    info = self.known_face_info.get(name, "")

            # Draw face box & label
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            if name != "Unknown":
                # Get dress region
                face_height = bottom - top
                dress_top = int(bottom)
                dress_bottom = min(frame.shape[0], int(bottom + face_height * 2.0))
                dress_left = max(0, int(left - face_height))
                dress_right = min(frame.shape[1], int(right + face_height))

                dress_region = frame[dress_top:dress_bottom, dress_left:dress_right]

                if dress_region.size != 0:
                    dom_color = self._get_dominant_color(dress_region)
                    col_name = self._color_name(dom_color)

                    dress_gray = cv2.cvtColor(dress_region, cv2.COLOR_BGR2GRAY)
                    dress_resized = cv2.resize(dress_gray, (28, 28))
                    dress_norm = dress_resized / 255.0
                    dress_input = dress_norm[np.newaxis, ..., np.newaxis]

                    pred = self.model.predict(dress_input)
                    dress_type = self.class_names[np.argmax(pred)]

                    cv2.putText(frame, f"Dress: {dress_type}", (left, dress_bottom + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Color: {col_name}", (left, dress_bottom + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    detected_faces.append({
                        'name': name,
                        'dress_type': dress_type,
                        'color': col_name,
                        'timestamp': current_time.isoformat(),
                        'camera_id': camera_id,
                        'coordinates': [left, top, right, bottom]
                    })

        # Update tracking data
        self.tracking_data[camera_id]['prev_frame'] = frame
        self.tracking_data[camera_id]['last_motion'] = current_time

        return frame, detected_faces

async def process_camera(camera_id: str, camera: CameraManager, system: AsyncShopCCTVSystem):
    """Process frames from a single camera asynchronously"""
    try:
        # Create window for this camera
        window_name = f"Shop CCTV - {camera_id}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)  # Set initial window size

        while True:
            # Read frame with retry logic
            frame = await camera.read_frame()
            if frame is None:
                await asyncio.sleep(0.1)  # Prevent CPU hogging during retry
                continue

            # Process frame
            processed_frame, detections = await system.process_frame(frame, camera_id)

            # Show frame
            cv2.imshow(window_name, processed_frame)

            # Log detections
            if detections:
                logger.info(f"Detections in {camera_id}: {json.dumps(detections)}")

            # Small sleep to prevent CPU hogging
            await asyncio.sleep(0.01)

            # Check for exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except asyncio.CancelledError:
        logger.info(f"Stopping processing for camera {camera_id}")
        raise
    except Exception as e:
        logger.error(f"Error processing {camera_id}: {str(e)}")
    finally:
        # Close window
        cv2.destroyWindow(f"Shop CCTV - {camera_id}")
        await camera.release()

async def main():
    """Main async function to run the shop CCTV system"""
    # Initialize system
    system = AsyncShopCCTVSystem()
    await system._initialize_system()
    
    # Initialize cameras
    cameras = {
        'entrance': CameraManager('entrance', 0),  # Main entrance
        'aisle1': CameraManager('aisle1', 1),   # Aisle 1
        'aisle2': CameraManager('aisle2', 2),   # Aisle 2
        'checkout': CameraManager('checkout', 3)   # Checkout area
    }

    # Create tasks for each camera
    tasks = []
    for camera_id, camera in cameras.items():
        task = asyncio.create_task(process_camera(camera_id, camera, system))
        tasks.append(task)

    try:
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logger.info("Received interrupt, stopping all cameras...")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        # Cleanup
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())