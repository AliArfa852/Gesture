import cv2
import mediapipe as mp
import numpy as np
import socket

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def recognize_gesture(hand_landmarks, image_width, image_height):
    gestures = {
        "stop": False,
        "left": False,
        "right": False,
        "up": False,
        "down": False,
        "servo_1": None,  # None means no detection, -1 to 1 for specific control
        "servo_2": None,
    }

    # Convert normalized coordinates to pixel values
    def get_coords(landmark):
        return int(landmark.x * image_width), int(landmark.y * image_height)

    # Extract landmark positions
    thumb_tip = get_coords(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP])
    index_tip = get_coords(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP])
    middle_tip = get_coords(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP])
    ring_tip = get_coords(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP])
    pinky_tip = get_coords(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP])
    wrist = get_coords(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST])

    # Calculate distances
    def distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    # Check for 'stop' gesture (all fingers extended)
    if (distance(thumb_tip, wrist) > distance(index_tip, wrist) and
        distance(middle_tip, wrist) > distance(index_tip, wrist) and
        distance(ring_tip, wrist) > distance(index_tip, wrist) and
        distance(pinky_tip, wrist) > distance(index_tip, wrist)):
        gestures["stop"] = True
        return gestures

    # Check for thumb gestures
    if thumb_tip[0] < wrist[0] and abs(thumb_tip[1] - wrist[1]) < 50:
        gestures["left"] = True
        return gestures

    if thumb_tip[0] > wrist[0] and abs(thumb_tip[1] - wrist[1]) < 50:
        gestures["right"] = True
        return gestures

    if thumb_tip[1] < wrist[1] and abs(thumb_tip[0] - wrist[0]) < 50:
        gestures["forward"] = True
        return gestures

    if thumb_tip[1] > wrist[1] and abs(thumb_tip[0] - wrist[0]) < 50:
        gestures["backward"] = True
        return gestures

    # Check for servo control gestures
    # Servo 1: Only index finger extended
    if (distance(index_tip, wrist) < distance(middle_tip, wrist) and
        distance(middle_tip, wrist) > distance(ring_tip, wrist) and
        distance(ring_tip, wrist) > distance(pinky_tip, wrist)):
        relative_position = (index_tip[0] - wrist[0]) / image_width
        gestures["servo_1"] = np.clip(relative_position * 2 - 1, -1, 1)  # Mapping to [-1, 1]
        return gestures

    # Servo 2: Index and middle fingers extended
    if (distance(index_tip, wrist) < distance(middle_tip, wrist) and
        distance(middle_tip, wrist) < distance(ring_tip, wrist) and
        distance(ring_tip, wrist) > distance(pinky_tip, wrist)):
        relative_position = (index_tip[0] - wrist[0]) / image_width
        gestures["servo_2"] = np.clip(relative_position * 2 - 1, -1, 1)  # Mapping to [-1, 1]
        return gestures

    return gestures

def send_command_to_bot(ip, command):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((ip, 80))
        request = f'POST /command HTTP/1.1\r\nContent-Type: text/plain\r\nContent-Length: {len(command)}\r\n\r\n{command}'
        s.sendall(request.encode())

# Get the bot's IP address from the user
bot_ip = input("Enter the bot's IP address: ")

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Recognize gestures
                image_height, image_width, _ = image.shape
                gestures = recognize_gesture(hand_landmarks, image_width, image_height)
                
                # Display recognized gesture and send command to bot
                for gesture, recognized in gestures.items():
                    if recognized:
                        if isinstance(recognized, bool) and recognized:
                            cv2.putText(image, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                            send_command_to_bot(bot_ip, gesture)
                            break  # Only one command at a time
                        elif isinstance(recognized, float):
                            cv2.putText(image, f"{gesture}: {recognized:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                            if gesture == "servo_1":
                                angle = int((recognized + 1) * 90)  # Convert [-1, 1] to [0, 180]
                                send_command_to_bot(bot_ip, f"servo1:{angle}")
                            elif gesture == "servo_2":
                                angle = int((recognized + 1) * 90)  # Convert [-1, 1] to [0, 180]
                                send_command_to_bot(bot_ip, f"servo2:{angle}")
                            break  # Only one command at a time

        # Flip the image horizontally for a selfie-view display.
        flipped_image = cv2.flip(image, 1)
        cv2.imshow('MediaPipe Hands', flipped_image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
