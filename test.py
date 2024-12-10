from pykinect2024 import PyKinect2024, PyKinectRuntime
import cv2
import numpy as np
import mediapipe as mp

# Initialize Kinect
kinect = PyKinectRuntime.PyKinectRuntime(PyKinect2024.FrameSourceTypes_Color | PyKinect2024.FrameSourceTypes_Depth)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Function to get depth at a specific point
def get_depth_at_point(depth_frame, x, y, color_width=1280, color_height=720, depth_width=512, depth_height=424):
    x = int(x * depth_width / color_width)
    y = int(y * depth_height / color_height)
    return depth_frame[y, x]

# Main loop
while True:
    if kinect.has_new_color_frame() and kinect.has_new_depth_frame():
        color_frame = kinect.get_last_color_frame()
        color_frame = color_frame.reshape((1080, 1920, 4)).astype(np.uint8)
        color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGRA2BGR)
        color_frame = cv2.resize(color_frame, (1280, 720))

        depth_frame = kinect.get_last_depth_frame()
        depth_frame = depth_frame.reshape((424, 512)).astype(np.uint16)

        # Normalize the depth map for visualization
        depth_frame_normalized = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
        depth_frame_normalized = np.uint8(depth_frame_normalized)
        depth_frame_colormap = cv2.applyColorMap(depth_frame_normalized, cv2.COLORMAP_JET)

        # Display the color frame and depth map
        cv2.imshow('Color Frame', color_frame)
        cv2.imshow('Depth Map', depth_frame_colormap)

        result = face_mesh.process(color_frame)

        if result.multi_face_landmarks:
            for facial_landmarks in result.multi_face_landmarks:
                landmarks = facial_landmarks.landmark
                for lm in landmarks:
                    x = int(lm.x * 1280)
                    y = int(lm.y * 720)
                    cv2.circle(color_frame, (x, y), radius=2, color=(255, 0, 255), thickness=-1)

                # Display the annotated color frame
                cv2.imshow('Color Frame with Landmarks', color_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kinect.close()
cv2.destroyAllWindows()
