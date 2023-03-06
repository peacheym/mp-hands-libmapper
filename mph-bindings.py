#!/usr/bin/python3

import cv2
import mediapipe as mp
import libmapper as mpr
import argparse

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def get_landmark_index():
  # Todo: Ensure this function always returns the correct index for the joint's being estimated
  return 8


class MPRunner:
  def __init__(self, dev_name, joint_type, model_complexity, min_detection_confidence, min_tracking_confidence, max_hands):
    # Set class members
    self.dev_name = dev_name
    self.joint_type = joint_type
    self.model_complexity = model_complexity
    self.min_detection_confidence = min_detection_confidence
    self.min_tracking_confidence = min_tracking_confidence
    self.max_hands = max_hands
    
  def _setup_libmapper(self):
    # Handle libmapper setup  
    self.dev = mpr.Device(self.dev_name)
    self.thumb = self.dev.add_signal(mpr.Direction.OUTGOING, self.format_sig_name("Thumb"), 3, mpr.Type.FLOAT, None, 0, 1)
    self.index = self.dev.add_signal(mpr.Direction.OUTGOING, self.format_sig_name("Index"), 3, mpr.Type.FLOAT, None, 0, 1)
    self.middle = self.dev.add_signal(mpr.Direction.OUTGOING, self.format_sig_name("Middle"), 3, mpr.Type.FLOAT, None, 0, 1)
    self.ring = self.dev.add_signal(mpr.Direction.OUTGOING, self.format_sig_name("Ring"), 3, mpr.Type.FLOAT, None, 0, 1)
    self.pinky = self.dev.add_signal(mpr.Direction.OUTGOING, self.format_sig_name("Pinky"), 3, mpr.Type.FLOAT, None, 0, 1)
    self.wrist = self.dev.add_signal(mpr.Direction.OUTGOING, "Wrist", 3, mpr.Type.FLOAT, None, 0, 1)
    
    
  def format_sig_name(self, name):
    joint = str(args.joint_type)
    if name == "Thumb":
      joint = self.convert_joint_names()
    return name + "_{}".format(joint.upper())
    
  def convert_joint_names(self):
    if self.joint_type == "tip":
      return self.joint_type
    if self.joint_type == "dip":
      return "IP"
    if self.joint_type == "pip":
      return "MCP"
    if self.joint_type == "mcp":
      return "CMC"
  
  def poll(self):
    self.dev.poll()
    
  def run_mp(self):
    self._setup_libmapper()
    
    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        model_complexity=int(self.model_complexity),
        min_detection_confidence=float(self.min_detection_confidence),
        min_tracking_confidence=float(self.min_tracking_confidence),
        max_num_hands=int(self.max_hands)) as hands:
        
      while cap.isOpened():
          
        self.poll()
          
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue
      
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
            
            # TODO: Update libmapper signals here, based on landmark index
            lm = hand_landmarks.landmark[get_landmark_index()]
            self.index.set_value([lm.x, lm.y, lm.z])
            
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('libmapper + MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
          break
  
    cap.release()
    
    
# Handle ArgParse
parser = argparse.ArgumentParser()

parser.add_argument("--max-hands", default=2, help="Maximum number of hands tracked by MediaPipe")
parser.add_argument("--model-complexity", default=0, help="Model complexity (0 or 1). 0 is better inference performance whereas 1 is better model accuracy")
parser.add_argument("--min-detection-confidence", default=0.5, help="Minimum confidence required by the ML model to detect a landmark")
parser.add_argument("--min-tracking-confidence", default=0.5, help="Minimum confidence required by the ML model to track a landmark")

parser.add_argument("--joint-type", default="tip", choices=["mcp", "pip", "dip", "tip", "all"], help="Determine which joints to report, per finger (other than thumb). ")

args = parser.parse_args()


runner = MPRunner("MPHands", args.joint_type, args.model_complexity, args.min_detection_confidence, args.min_tracking_confidence, args.max_hands)
runner.run_mp()