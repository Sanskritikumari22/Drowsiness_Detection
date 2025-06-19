import cv2
import numpy as np
from scipy.spatial import distance
from ultralytics import YOLO
import torch
import mediapipe as mp
import time
import streamlit as st
from PIL import Image

# Load the trained YOLOv5 model (using GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("runs/detect/train9/weights/best.pt")  # Load your custom trained model
model.to(device)  # Move the model to GPU if available

# Initialize MediaPipe face mesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    # Calculate the Euclidean distances between vertical eye landmarks
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    
    # Calculate the Euclidean distance between horizontal eye landmarks
    C = distance.euclidean(eye[0], eye[3])
    
    # Return the Eye Aspect Ratio (EAR)
    ear = (A + B) / (2.0 * C)
    return ear

# Thresholds for EAR to detect blinking and fatigue
EAR_THRESHOLD = 0.3
BLINK_CONSEC_FRAMES = 15
HEAD_DOWN_THRESHOLD = 0.5  # Relative position threshold for head-down detection

# Setup webcam capture
cap = cv2.VideoCapture(0)

frame_count = 0
drowsiness_detected = False
blink_count = 0
eye_closed_time = 0  # Total time in seconds the eyes are closed
head_down_count = 0
head_down_start_time = None  # Track when the head-down state starts
head_down_duration = 0  # Track how long head has been down
head_down_started = False  # Flag to track if the head down detection has started

# To track time when eyes are closed
last_eye_closed_time = None

# Variables to track head down event duration and count
head_down_event_started = False
head_down_event_duration = 0  # Duration the head has been down (in seconds)

# Streamlit configuration
st.title("Drowsiness and Head Down Detection")
st.write("Real-time drowsiness and head down detection using YOLO and MediaPipe")

# Video stream display
frame_placeholder = st.empty()  # Placeholder for video feed

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run the YOLO model on the current frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb)  # Get predictions using the YOLO model
    
    # The results are now contained in the `results` object (which is a list of Results objects)
    # Access the first image's result
    result = results[0]  # Get the result from the first image in the batch
    
    # Extract the predicted labels and their classes
    boxes = result.boxes  # This contains the bounding box information
    class_ids = boxes.cls.cpu().numpy().astype(int)  # Predicted class IDs (use `.cls` for class IDs)
    labels = result.names  # Map class IDs to class labels (e.g., 'Awake', 'Drowsy')
    
    # Check if the person is Awake or Drowsy based on YOLO output
    for i in range(len(boxes)):
        label = labels[class_ids[i]]
        if label == 'Drowsy':
            drowsiness_detected = True
            cv2.putText(frame, "Drowsy", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            drowsiness_detected = False
            cv2.putText(frame, "Awake", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Convert the frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Get face landmarks using MediaPipe
    results_mesh = face_mesh.process(rgb_frame)
    
    if results_mesh.multi_face_landmarks:
        head_down_started = False  # Reset flag when face is detected
        
        for face_landmarks in results_mesh.multi_face_landmarks:
            # Get the left and right eye landmarks (indices are based on the 468 landmarks of MediaPipe)
            left_eye = [face_landmarks.landmark[i] for i in range(33, 133)]
            right_eye = [face_landmarks.landmark[i] for i in range(133, 233)]
            
            # Convert MediaPipe landmarks to pixel coordinates
            h, w, _ = frame.shape
            left_eye = [(int(landmark.x * w), int(landmark.y * h)) for landmark in left_eye]
            right_eye = [(int(landmark.x * w), int(landmark.y * h)) for landmark in right_eye]
            
            # Calculate EAR for both eyes
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            
            # Average EAR for both eyes
            ear = (left_ear + right_ear) / 2.0
            
            # Check for fatigue based on EAR
            if ear < EAR_THRESHOLD:
                frame_count += 1
                if frame_count >= BLINK_CONSEC_FRAMES:
                    blink_count += 1
                    cv2.putText(frame, "Fatigue Detected", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
                # Track how long eyes are closed
                if last_eye_closed_time is None:
                    last_eye_closed_time = time.time()
                else:
                    eye_closed_time = int(time.time() - last_eye_closed_time)
            else:
                frame_count = 0
                last_eye_closed_time = None
        
            # Draw the landmarks on the face
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

            # Detect head down (assuming the face orientation in the frame is used)
            head_pos_y = face_landmarks.landmark[1].y  # Use a landmark near the center of the face
            if head_pos_y > HEAD_DOWN_THRESHOLD:  # If the head position is below a threshold
                cv2.putText(frame, "Head Down Detected", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    else:
        # No face detected, assume head down and start counting
        if not head_down_event_started:
            head_down_event_started = True
            head_down_start_time = time.time()  # Record the time when no face is detected
    
    # If no face is detected, calculate how long the head has been down
    if head_down_event_started:
        head_down_event_duration = int(time.time() - head_down_start_time)
        cv2.putText(frame, f"Head Down for: {head_down_event_duration}s", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Once the head has been down for a threshold time, count it as a head down event
        if head_down_event_duration >= 5:  # Adjust threshold to suit your needs
            head_down_count += 1
            head_down_event_started = False  # Reset head down event tracking after counting
    
    # Display real-time statistics (seconds closed, blink count, head down count)
    cv2.putText(frame, f"Eyes Closed: {eye_closed_time}s", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Blink Count: {blink_count}", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Head Down Count: {head_down_count}", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # Convert frame to Image for Streamlit
    image = Image.fromarray(frame)

    # Update the video feed in Streamlit
    frame_placeholder.image(image, channels="RGB", use_column_width=True)

    # Display statistics on Streamlit
    st.write(f"Eyes Closed: {eye_closed_time}s")
    st.write(f"Blink Count: {blink_count}")
    st.write(f"Head Down Count: {head_down_count}")

    # Break the loop if 'q' is pressed (not needed in Streamlit as it runs in the web interface)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

