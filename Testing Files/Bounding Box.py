from pykinect2024 import PyKinect2024, PyKinectRuntime
import cv2
import numpy as np
import mediapipe as mp

# Initialize Kinect
kinect = PyKinectRuntime.PyKinectRuntime(PyKinect2024.FrameSourceTypes_Color | PyKinect2024.FrameSourceTypes_Depth)


# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

# Function to draw bounding box
def draw_bounding_box(frame, detection):
    bboxC = detection.location_data.relative_bounding_box
    ih, iw, _ = frame.shape
    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

try:
    while True:
        # Capture frame from Kinect
        if kinect.has_new_color_frame():
            frame = kinect.get_last_color_frame()
            frame = frame.reshape((1080, 1920, 4))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Convert the image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame for face detection
            results = face_detection.process(rgb_frame)

            # Draw bounding boxes if faces are detected
            if results.detections:
                for detection in results.detections:
                    draw_bounding_box(frame, detection)

            # Display the frame
            cv2.imshow('Kinect Face Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release resources
    kinect.close()
    cv2.destroyAllWindows()