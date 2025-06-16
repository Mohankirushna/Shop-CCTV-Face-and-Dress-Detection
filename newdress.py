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
        self.max_attempts = 3
        self.retry_delay = 1
        self.last_frame = None
        self.last_frame_time = datetime.min
        self.frame_timeout = 2

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
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 15)
            
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
        
        # Simplified clothing types - only the most distinguishable ones
        self.simple_clothing_types = [
            'shirt',      # Shirts and tops
            'dress',      # Dresses
            'jacket'      # Coats and jackets
        ]
        
        # Mapping from Fashion-MNIST to simplified types
        self.clothing_mapping = {
            0: 'shirt',    # T-shirt/top -> shirt
            1: None,       # Trouser -> ignore (focus on upper body)
            2: 'jacket',   # Pullover -> jacket
            3: 'dress',    # Dress -> dress
            4: 'jacket',   # Coat -> jacket
            5: None,       # Sandal -> ignore
            6: 'shirt',    # Shirt -> shirt
            7: None,       # Sneaker -> ignore
            8: None,       # Bag -> ignore (too confusing)
            9: None        # Ankle boot -> ignore
        }
        
        self.model = None
        
        # Initialize tracking data storage
        self.tracking_data = {}
        
        # Motion detection settings
        self.motion_threshold = 25
        self.min_motion_area = 500
        
        # Face detection settings
        self.face_distance_threshold = 0.6
        self.min_face_size = 30
        self.face_detection_model = 'hog'
        self.upsample_times = 1
        self.num_jitters = 1
        
        # Clothing detection settings
        self.clothing_confidence_threshold = 0.5  # Higher threshold for better accuracy
        
        # Initialize the system
        asyncio.create_task(self._initialize_system())
        
        # Initialize OpenCV window
        cv2.namedWindow('Shop CCTV - Press Q to quit', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Shop CCTV - Press Q to quit', 800, 600)

    async def _initialize_system(self):
        """Initialize the system"""
        # Load known faces if any
        await self._load_known_faces()
        
        # Initialize dress classification model
        try:
            self.model = await self._load_or_train_model()
            if self.model is None:
                logger.warning("Failed to load or train dress classification model")
                self.model = self._create_model()
                logger.info("Created fallback dress classification model")
        except Exception as e:
            logger.error(f"Error initializing dress model: {str(e)}")
            self.model = self._create_model()
            logger.info("Created new dress classification model after error")

    async def _load_known_faces(self):
        """Load known faces from directory"""
        celebs_dir = Path("/Users/amritpal/celebrity_face_recognition/celebs")
        if celebs_dir.exists() and celebs_dir.is_dir():
            loaded_count = 0
            for file in celebs_dir.glob('*'):
                if not file.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    continue
                try:
                    logger.info(f"Processing face image: {file.name}")
                    image = face_recognition.load_image_file(str(file))
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        self.known_face_encodings.append(encodings[0])
                        name = self._clean_name(file.stem)
                        self.known_face_names.append(name)
                        logger.info(f"Successfully loaded face: {name}")
                        loaded_count += 1
                    else:
                        logger.warning(f"No faces found in {file.name}")
                except Exception as e:
                    logger.error(f"Error processing {file.name}: {str(e)}")
            
            if loaded_count > 0:
                logger.info(f"Successfully loaded {loaded_count} known faces")
            else:
                logger.warning("No valid face images found in the celebs directory")
        else:
            logger.info("No 'celebs' directory found, running without known faces")

    async def _load_or_train_model(self):
        """Load or train dress classification model"""
        model_path = Path("simple_dress_model.h5")
        try:
            if model_path.exists():
                logger.info("Loading pre-trained simple dress model from disk")
                return keras.models.load_model(str(model_path))
            else:
                logger.info("Creating and training new simple dress model")
                model = self._create_model()
                
                # Load Fashion-MNIST dataset
                fashion_mnist = keras.datasets.fashion_mnist
                (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
                
                # Preprocess the data
                train_images = train_images.astype('float32') / 255.0
                test_images = test_images.astype('float32') / 255.0
                train_images = train_images[..., np.newaxis]
                test_images = test_images[..., np.newaxis]
                
                # Train the model with fewer epochs for faster training
                logger.info("Training simple dress model...")
                model.fit(
                    train_images, 
                    train_labels,
                    batch_size=128,
                    epochs=5,  # Reduced epochs
                    validation_data=(test_images, test_labels),
                    verbose=1
                )
                
                # Save the model
                model.save(str(model_path))
                logger.info(f"Simple dress model trained and saved to {model_path}")
                return model
        except Exception as e:
            logger.error(f"Error loading or training dress model: {str(e)}")
            return None

    def _clean_name(self, filename: str) -> str:
        """Clean and format the name from filename"""
        name = filename.replace("_", " ").replace("-", " ")
        name = re.sub(r"\([^)]*\)", "", name)
        name = re.sub(r"\d+", "", name)
        name = re.sub(r"\s+", " ", name).strip()
        return name.title()

    def _create_model(self):
        """Create a new dress classification model"""
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.5),  # Add dropout for better generalization
            keras.layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def _get_dominant_color(self, image: np.ndarray, k: int = 3) -> str:
        """Get dominant color from image with improved accuracy"""
        try:
            # Convert to HSV color space for better color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Reshape the image to be a list of pixels
            pixels = hsv.reshape((-1, 3))
            
            # Convert to float32 for k-means
            pixels = np.float32(pixels)
            
            # Define criteria and apply k-means
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
            _, labels, palette = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Get the dominant colors and their counts
            _, counts = np.unique(labels, return_counts=True)
            dominant_idx = np.argmax(counts)
            dominant_hsv = palette[dominant_idx]
            
            # Extract HSV values
            h, s, v = dominant_hsv
            
            # Scale hue to 0-360
            h_deg = h * 2  # Since OpenCV uses 0-180 for hue
            
            # Handle black, white, and gray based on value and saturation
            if v < 40:
                return "black"
            if s < 30 and v > 200:
                return "white"
            if s < 30:
                return "gray"
            
            # Define color ranges in HSV space
            if 0 <= h_deg < 15 or 345 <= h_deg <= 360:
                return "red"
            elif 15 <= h_deg < 35:
                return "orange"
            elif 35 <= h_deg < 65:
                return "yellow"
            elif 65 <= h_deg < 150:
                return "green"
            elif 150 <= h_deg < 210:
                return "blue"
            elif 210 <= h_deg < 270:
                return "purple"
            elif 270 <= h_deg < 345:
                return "pink"
            else:
                return "colorful"
                
        except Exception as e:
            logger.error(f"Error in color detection: {str(e)}")
            return "colorful"

    def _predict_clothing_type(self, clothing_region: np.ndarray) -> Optional[str]:
        """Predict clothing type from image region with simplified types"""
        if self.model is None:
            return None
            
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(clothing_region, cv2.COLOR_BGR2GRAY)
            
            # Resize to 28x28 (Fashion-MNIST size)
            resized = cv2.resize(gray, (28, 28))
            
            # Apply histogram equalization for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(resized)
            
            # Normalize
            normalized = enhanced.astype('float32') / 255.0
            
            # Reshape for model input
            input_data = normalized.reshape(1, 28, 28, 1)
            
            # Make prediction
            predictions = self.model.predict(input_data, verbose=0)[0]
            
            # Get the most confident prediction
            predicted_class = np.argmax(predictions)
            confidence = float(predictions[predicted_class])
            
            logger.info(f"Prediction - Class: {predicted_class}, Confidence: {confidence:.3f}")
            
            # Check if confidence is high enough
            if confidence < self.clothing_confidence_threshold:
                logger.info(f"Low confidence ({confidence:.3f}), skipping clothing type")
                return None
            
            # Map to simplified clothing type
            clothing_type = self.clothing_mapping.get(predicted_class)
            
            if clothing_type is None:
                logger.info(f"Class {predicted_class} mapped to None, skipping")
                return None
            
            logger.info(f"Mapped to clothing type: {clothing_type}")
            return clothing_type
            
        except Exception as e:
            logger.error(f"Error in clothing prediction: {str(e)}")
            return None

    def _generate_compliment(self, name: str, color: str, clothing_type: Optional[str]) -> str:
        """Generate a compliment message based on available information"""
        greeting = f"Hello {name}" if name != "Unknown" else "Hello"
        
        if clothing_type and color:
            return f"{greeting}, looking great in that {color} {clothing_type}!"
        elif color:
            return f"{greeting}, love that {color} outfit!"
        elif clothing_type:
            return f"{greeting}, nice {clothing_type}!"
        else:
            return f"{greeting}, looking good today!"

    async def process_frame(self, frame: np.ndarray, camera_id: str) -> Tuple[np.ndarray, dict]:
        """Process a single frame from a camera"""
        current_time = datetime.now()
        detected_faces = {}

        # Process frame at full resolution
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all face locations and encodings
        face_locations = face_recognition.face_locations(
            rgb_frame,
            number_of_times_to_upsample=self.upsample_times,
            model=self.face_detection_model
        )
        face_encodings = face_recognition.face_encodings(
            rgb_frame,
            known_face_locations=face_locations,
            num_jitters=self.num_jitters
        )

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            face_width = right - left
            face_height = bottom - top
            if face_width < self.min_face_size or face_height < self.min_face_size:
                continue

            # Calculate body region (focus on upper torso)
            height, width = frame.shape[:2]
            
            # More conservative approach for upper body clothing
            body_top = bottom
            body_bottom = min(bottom + (face_height * 2), height - 1)  # Reduced to 2x face height
            body_left = max(0, left - face_width // 3)  # Less horizontal expansion
            body_right = min(width - 1, right + face_width // 3)
            
            # Ensure we have a valid region
            if body_bottom <= body_top or body_right <= body_left:
                continue
            
            # Get the clothing region
            clothing_region = frame[body_top:body_bottom, body_left:body_right]
            
            if clothing_region.size == 0:
                continue
            
            # Draw debug rectangle for clothing detection area (optional)
            # cv2.rectangle(frame, (body_left, body_top), (body_right, body_bottom), (255, 0, 0), 1)
            
            # Get dominant color
            color_name = self._get_dominant_color(clothing_region)
            
            # Predict clothing type (with simplified types)
            clothing_type = self._predict_clothing_type(clothing_region)
            
            # Face recognition
            name = "Unknown"
            if self.known_face_encodings:
                try:
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings, 
                        face_encoding,
                        tolerance=0.6
                    )
                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings, 
                        face_encoding
                    )
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                except Exception as e:
                    logger.error(f"Error in face recognition: {str(e)}")
            
            # Generate compliment message
            message = self._generate_compliment(name, color_name, clothing_type)
            
            # Log the detection results
            logger.info(f"Detection - Name: {name}, Color: {color_name}, Type: {clothing_type}")
            logger.info(f"Message: {message}")
            
            # Draw face box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Draw text with background for better visibility
            text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_bg_top = max(0, top - text_size[1] - 10)
            text_bg_bottom = top - 5
            
            # Ensure text background stays within frame
            if text_bg_top < 0:
                text_bg_bottom -= text_bg_top
                text_bg_top = 0
            
            cv2.rectangle(frame, 
                        (left, text_bg_top), 
                        (left + text_size[0] + 10, text_bg_bottom), 
                        (0, 0, 0), 
                        -1)
            cv2.putText(frame, message, (left + 5, top - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame, detected_faces

async def process_camera(camera_id: str, camera: CameraManager, system: AsyncShopCCTVSystem):
    """Process frames from the camera asynchronously"""
    logger.info("Starting camera processing...")
    
    try:
        while True:
            frame = await camera.read_frame()
            if frame is None:
                await asyncio.sleep(1)
                continue
                
            try:
                # Process the frame
                processed_frame, _ = await system.process_frame(frame, camera_id)
                
                # Display the processed frame
                cv2.imshow('Shop CCTV - Press Q to quit', processed_frame)
                
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User requested to quit")
                    break
                    
                # Small delay to prevent high CPU usage
                await asyncio.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
                await asyncio.sleep(1)
    except asyncio.CancelledError:
        logger.info("Camera processing cancelled")
    except Exception as e:
        logger.error(f"Fatal error in camera processing: {str(e)}")
    finally:
        cv2.destroyAllWindows()

async def main():
    """Main async function to run the shop CCTV system with a single camera"""
    system = AsyncShopCCTVSystem()
    
    # Wait a bit for system initialization
    await asyncio.sleep(2)
    
    # Initialize the camera
    camera = CameraManager('main_camera', 0)
    
    try:
        # Start camera processing
        await process_camera('main_camera', camera, system)
    except asyncio.CancelledError:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        # Clean up the camera
        await camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())