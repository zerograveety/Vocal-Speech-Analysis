import cv2
import mediapipe as mp

# Initialize mediapipe pose and drawing utils
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Load your video
cap = cv2.VideoCapture('Respect every choice you make at every moment. _ Hanke Yang _ TEDxHWSSH Youth.mp4')

with mp_pose.Pose(static_image_mode=False,
                  model_complexity=1,
                  enable_segmentation=False,
                  min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB for mediapipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Convert back to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw the landmarks on the frame
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Show the frame
        cv2.imshow('Mediapipe Pose Visualization', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()