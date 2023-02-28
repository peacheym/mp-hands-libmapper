#!/usr/bin/python3

import cv2
import mediapipe as mp
import argparse

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Handle ArgParse
parser = argparse.ArgumentParser()

parser.add_argument("--max-hands", default=2, help="Maximum number of hands tracked by MediaPipe")
parser.add_argument("--model-complexity", default=0, help="Model complexity (0 or 1). 0 is better inference performance whereas 1 is better model accuracy")
parser.add_argument("--min-detection-confidence", default=0.5, help="Minimum confidence required by the ML model to detect a landmark")
parser.add_argument("--min-tracking-confidence", default=0.5, help="Minimum confidence required by the ML model to track a landmark")


args = parser.parse_args()

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=int(args.model_complexity),
    min_detection_confidence=float(args.min_detection_confidence),
    min_tracking_confidence=float(args.min_tracking_confidence),
    max_num_hands=int(args.max_hands)) as hands:
    
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
0
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
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
        
        # TODO: Update libmapper signals here
        # print(hand_landmarks)
        
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('libmapper + MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
  
cap.release()
