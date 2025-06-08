import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("asl_model.keras")

# Update this list to exactly match your model training classes order
class_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ['del', 'nothing', 'space']

# Initialize MediaPipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start camera
cap = cv2.VideoCapture(0)

print("Starting ASL Recognition... Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally for natural (mirror) view
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get center of the palm (landmark 9)
            center_x = int(hand_landmarks.landmark[9].x * w)
            center_y = int(hand_landmarks.landmark[9].y * h)

            box_size = 128  # size of the crop box

            xmin = max(center_x - box_size // 2, 0)
            xmax = min(center_x + box_size // 2, w)
            ymin = max(center_y - box_size // 2, 0)
            ymax = min(center_y + box_size // 2, h)

            # Crop the hand image
            hand_img = frame[ymin:ymax, xmin:xmax]

            if hand_img.size == 0 or hand_img.shape[0] == 0 or hand_img.shape[1] == 0:
                continue  # skip if crop is invalid

            # Show cropped hand image for debugging
            cv2.imshow("Hand Crop", hand_img)

            # Preprocess the cropped hand image
            hand_img = cv2.resize(hand_img, (64, 64))
            hand_img = hand_img.astype("float32") / 255.0
            hand_img = np.expand_dims(hand_img, axis=0)

            # Predict using the model
            prediction = model.predict(hand_img)
            predicted_class = class_names[np.argmax(prediction)]

            # Display prediction on the frame
            cv2.putText(frame, f'Prediction: {predicted_class}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ASL Live", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

