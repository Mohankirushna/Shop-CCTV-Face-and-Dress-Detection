# Shop CCTV Face and Dress Detection System

This is an advanced CCTV system designed for retail shops that combines face recognition, dress type classification, and color detection. The system uses modern computer vision and machine learning techniques to provide real-time monitoring and analysis of customers and staff in a retail environment.

## Features

- **Face Recognition**: 
  - Detects and recognizes known faces using deep learning-based face recognition
  - Differentiates between staff and customers
  - Maintains a database of known faces

- **Dress Detection**: 
  - Classifies dress types using a pre-trained CNN model
  - Detects 10 different clothing categories: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
  - Uses Fashion MNIST dataset for training

- **Color Detection**: 
  - Identifies dominant colors of clothing using K-Means clustering
  - Supports multiple color categories including Red, Green, Blue, White, Black, Gray, Brown, Yellow, Purple, Pink

- **Real-time Processing**: 
  - Processes video streams in real-time using asynchronous processing
  - Handles multiple camera feeds concurrently
  - Implements motion detection to reduce processing load
  - Maintains frame rate and resolution optimization

- **Camera Management**: 
  - Supports multiple camera inputs
  - Automatic camera reconnection
  - Graceful handling of camera disconnections
  - Configurable camera settings (resolution, FPS)

## System Architecture

The system is built using an asynchronous architecture with three main components:

1. **CameraManager**
   - Handles camera connections and disconnections
   - Manages frame capture and retry logic
   - Maintains camera properties and settings
   - Implements automatic reconnection with retry delays

2. **AsyncShopCCTVSystem**
   - Core processing system for face and dress detection
   - Manages face recognition database
   - Handles dress type classification
   - Processes color detection
   - Maintains tracking data for motion detection

3. **Main Processing Loop**
   - Uses asyncio for concurrent processing
   - Implements frame processing pipeline
   - Manages display windows
   - Handles keyboard events

## Requirements

- Python 3.8+
- Required Python packages:
  - opencv-python
  - face_recognition
  - tensorflow
  - numpy
  - scikit-learn

## Installation

1. Install required packages:
```bash
pip install opencv-python face_recognition tensorflow numpy scikit-learn
```

2. Clone the repository:
```bash
git clone 
```

3. Add known faces to training directory:
- Create a directory named `celebrity_face_recognition/celebs`
- Add face images in this directory with appropriate names

## Usage

Run the main script:
```bash
python dress1.py
```

The system will:
1. Initialize the face recognition system
2. Load or train the dress classification model
3. Start processing video feeds from connected cameras
4. Display processed video feeds in separate windows
5. Log detections to console

## Configuration

The system can be configured through the following parameters in the code:

- Face Detection:
  - `face_detection_scale`: Controls image scaling for face detection
  - `face_distance_threshold`: Controls face recognition sensitivity
  - `min_face_size`: Minimum face size in pixels

- Camera Settings:
  - Resolution: 640x480
  - FPS: 15
  - Automatic reconnection with retry delays

- Motion Detection:
  - `motion_threshold`: Controls motion sensitivity
  - `min_motion_area`: Minimum area for motion detection
  - `motion_timeout`: Time before considering camera disconnected

## Output

The system displays:
- Real-time video feeds from connected cameras
- Face recognition results with names
- Dress type classifications
- Color detections
- Timestamps and camera IDs for all detections

Logs are written to console with timestamps and include:
- Camera connection status
- Face detection results
- Dress type classifications
- Color detections
- Error messages and warnings

## Error Handling

The system includes robust error handling for:
- Camera disconnections
- Frame capture failures
- Model loading errors
- Invalid frame processing
- Resource cleanup on exit

## Notes

- The system is designed to work with multiple cameras
- Only the entrance camera (index 0) is guaranteed to work
- Other camera indices (1, 2, 3) may not be available on all systems
- The system uses asynchronous processing to maintain real-time performance

## License

[Add your license information here]
