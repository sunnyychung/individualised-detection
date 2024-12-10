from pykinect2024 import PyKinect2024, PyKinectRuntime
import cv2
import numpy as np
from sklearn.svm import SVC
import mediapipe as mp
import time
import csv

kinect = PyKinectRuntime.PyKinectRuntime(PyKinect2024.FrameSourceTypes_Color | PyKinect2024.FrameSourceTypes_Depth)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Indices for key landmarks around eyes, mouth, and nose
key_landmarks_indices = [
    1,  # Tip of Nose Reference Point
    226, # Corner of Right Eye
    57, # Right Mouth Corner
    50, # Cheek
    164 # Philtrum
]

# Gets depth data at point x & y of the captured frame
def get_depth_at_point(depth_frame, x, y, color_width=1280, color_height=720, depth_width=512, depth_height=424):
    x = int(x * depth_width / color_width)
    y = int(y * depth_height / color_height)
    return depth_frame[y, x]

# 3D Euclidean Distance Formula
def calculate_3d_distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

def get_lmk_distance(landmarks, depth_frame):
    features = []
    num_landmarks = len(key_landmarks_indices)

    # Store key landmarks coordinates
    coordinates = []

    # Find x and y and z/depth of landmarks on captured frame
    for idx in key_landmarks_indices:
        lm = landmarks[idx]
        x = int(lm.x * 1280)
        y = int(lm.y * 720)
        coordinates.append((x, y))

    # Calculate Euclidean Distance from the tip of nose to other landmarks
    for i in range(1, num_landmarks):
        features.append(calculate_3d_distance(coordinates[0], coordinates[i]))

    return np.array(features)

def get_right_x(result):
    for facial_landmarks in result.multi_face_landmarks:
        r_side = facial_landmarks.landmark[234] #point near right cheek
        x = int(r_side.x * 1280)
        return x

def get_left_x(result):
    for facial_landmarks in result.multi_face_landmarks:
        l_side = facial_landmarks.landmark[454] #point near left cheek
        x = int(l_side.x * 1280)
        return x

def get_depth_nose(result, depth_frame):
    for facial_landmarks in result.multi_face_landmarks:
        nose = facial_landmarks.landmark[1] # Tip of Nose / Landmark 1
        x = int(nose.x * 1280)
        y = int(nose.y * 720)
        depth = get_depth_at_point(depth_frame, x, y)
        return depth
    
def get_x_nose(result):
    for facial_landmarks in result.multi_face_landmarks:
        nose = facial_landmarks.landmark[1] # Tip of Nose / Landmark 1
        x = int(nose.x * 1280)
        print("Nose" + str(x))
        return x

def get_y_chin(result):
    for facial_landmarks in result.multi_face_landmarks:
        chin = facial_landmarks.landmark[152] # Tip of Nose / Landmark 1
        y = int(chin.y * 720)
        print("Chin" + str(y))
        return y

def save_features_to_csv(features, label, filename='face_data.csv'):
    # Combine features with the label
    data = list(features) + [label]

    # Open CSV file in append mode
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the data
        writer.writerow(data)

def save_face(result):
    for facial_landmarks in result.multi_face_landmarks:
        features = get_lmk_distance(facial_landmarks.landmark, depth_frame)
        save_features_to_csv(features, "placeholder")

def draw_face(result):
    for facial_landmarks in result.multi_face_landmarks:
        for idx, lm in enumerate(facial_landmarks.landmark):
                    if idx == 234 or idx == 454 or idx == 1 or idx == 152:
                        x, y = int(lm.x * 1280), int(lm.y * 720)

                        cv2.circle(color_frame, (x, y), radius=2, color=(255, 0, 255), thickness=-1)

def draw_guide(frame, result, message):
    height, width = frame.shape[:2]
    
    frame = cv2.rectangle(frame, (779, 466), (516, 49) , (0, 255, 0), 2)
    
    draw_face(result)

    cv2.putText(color_frame, message, (516,480), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('Kinect 3D Face Landmarks', frame)

## https://towardsdatascience.com/head-pose-estimation-using-python-d165d3541600
def facial_pose(image, results):
    img_h, img_w, img_c = image.shape   
    face_3d = []
    face_2d = []

    for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])       
                
                # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                        [0, focal_length, img_w / 2],
                                        [0, 0, 1]])

                # The Distance Matrix
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360
    
    return x, y

svm = SVC(kernel='linear')

capture_limit = 5
captures = 0
start_time = time.time()
phase = 0

while True:
    if kinect.has_new_color_frame() and kinect.has_new_depth_frame():
        color_frame = kinect.get_last_color_frame()
        color_frame = color_frame.reshape((1080, 1920, 4)).astype(np.uint8)
        color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGRA2RGB)

        result = face_mesh.process(color_frame)

        color_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)

        x, y = facial_pose(color_frame, result)

        color_frame = cv2.resize(color_frame, (1280, 720))

        depth_frame = kinect.get_last_depth_frame()
        depth_frame = depth_frame.reshape((424, 512)).astype(np.uint16)
            
        result = face_mesh.process(color_frame)

        elapsed_time = time.time() - start_time

        if phase == 0:
            if (x >= 0 and x < 1 and y >= 0 and y < 1):
                message = "Great, youre facing forward"
                if (get_depth_nose(result, depth_frame) == 500):
                        message = "Success, face forward stage"
                        phase += 1
                        captures = 0
                elif(get_depth_nose(result, depth_frame) < 500):
                    message = "Go further away"
                else:
                    message = "Come forward"
            elif (x < 0.5 or x > 2):
                if (x < 0.5):
                    message = "Tilt Head Up"
                else:
                    message = "Tilt Head Down"
            elif (y > 1 or y < 0):
                if (y < 0):
                    message = "Turn Head Right"
                else:
                    message = "Turn Head Left"

            color_frame = draw_guide(color_frame, result, message)
        else:
            if captures <= capture_limit:
                print("Hold")
                save_face(result)
                captures += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kinect.close()
cv2.destroyAllWindows()