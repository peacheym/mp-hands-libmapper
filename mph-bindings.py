#!/usr/bin/python3

import cv2
import numpy as np
import mediapipe as mp
import libmapper as mpr
import argparse
import signal

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
done = False

def handler_done(signum, frame):
    global done
    done = True

signal.signal(signal.SIGINT, handler_done)
signal.signal(signal.SIGTERM, handler_done)

class MPRunner:
  """
  This class is responsible for the running the web-cam, mediapipe pose estimation, libmapper signal initialization & updates, etc.
  
  """
  def __init__(self, dev_name, joint_type, model_complexity, min_detection_confidence, min_tracking_confidence, max_hands, palm_viz):
    # Set class members
    self.dev_name = dev_name
    self.joint_type = joint_type
    self.model_complexity = model_complexity
    self.min_detection_confidence = min_detection_confidence
    self.min_tracking_confidence = min_tracking_confidence
    self.max_hands = max_hands
    self.joints = ["mcp", "pip", "dip", "tip"]
    self.palm_viz = palm_viz
    
  def _setup_libmapper(self):
    # Handle libmapper setup  
    graph = mpr.Graph()
    # graph.set_interface("wlp0s20f3") 
    self.dev = mpr.Device(self.dev_name, graph)
    self.signals = {}
    self.signals["wrist"] = self.dev.add_signal(mpr.Direction.OUTGOING, "Wrist", 3, mpr.Type.FLOAT, None, 0, 1)
    self.signals["thumb"] = self.dev.add_signal(mpr.Direction.OUTGOING, self.format_sig_name("Thumb"), 3, mpr.Type.FLOAT, None, 0, 1)
    self.signals["index"] = self.dev.add_signal(mpr.Direction.OUTGOING, self.format_sig_name("Index"), 3, mpr.Type.FLOAT, None, 0, 1)
    self.signals["middle"] = self.dev.add_signal(mpr.Direction.OUTGOING, self.format_sig_name("Middle"), 3, mpr.Type.FLOAT, None, 0, 1)
    self.signals["ring"] = self.dev.add_signal(mpr.Direction.OUTGOING, self.format_sig_name("Ring"), 3, mpr.Type.FLOAT, None, 0, 1)
    self.signals["pinky"] = self.dev.add_signal(mpr.Direction.OUTGOING, self.format_sig_name("Pinky"), 3, mpr.Type.FLOAT, None, 0, 1)
    self.signals["PalmRotation"] = self.dev.add_signal(mpr.Direction.OUTGOING, "PalmRotation", 3, mpr.Type.FLOAT, None, -1, 1) # Todo: double check that -1 to 1 is the actual min/max.
    
  def format_sig_name(self, name):
    joint = self.joint_type
    if name == "Thumb":
      joint = self.convert_joint_names()
    return name + "{}".format(joint.upper())
    
  def convert_joint_names(self):
    if self.joint_type == "tip":
      return self.joint_type
    if self.joint_type == "dip":
      return "IP"
    if self.joint_type == "pip":
      return "MCP"
    if self.joint_type == "mcp":
      return "CMC"
    
  def get_landmark_index(self, finger_index):
    if finger_index == 0:
      return 0 # Special case for wrist
    else:
      return 4 * (finger_index-1) + self.joints.index(self.joint_type) + 1
  
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
        
      while not done and cap.isOpened():
          
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


            if self.palm_viz:
              self.draw_palm_box(image, hand_landmarks)
            
            self.signals["PalmRotation"].set_value(self.compute_plane_rotation(hand_landmarks.landmark))
            # self.compute_palm_plane(hand_landmarks.landmark)
                                    
            for i, (k, v) in enumerate(self.signals.items()): # For every signal
              if k == "PalmRotation":
                continue
              lm = hand_landmarks.landmark[self.get_landmark_index(i)] # Compute which landmark to fetch estimations from
              # print(v.get_property("name"), self.get_landmark_index(i))
              v.set_value([lm.x, lm.y, lm.z]) # Update signals x,y,z components.

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('libmapper + MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
          break
  
    cap.release()
    self.dev.free()
    
  def compute_plane_rotation(self, lm):
    """
    Formula found here: https://math.stackexchange.com/questions/2249307/orientation-of-a-3d-plane-using-three-points
    """
    A = np.asarray([lm[0].x, lm[0].y, lm[0].z])
    B = np.asarray([lm[5].x, lm[5].y, lm[5].z])
    C = np.asarray([lm[17].x, lm[17].y, lm[17].z])
    
    cross = np.cross(B-A, C-A)
    
    U = cross/np.linalg.norm(cross) # This is the unit vector of the plane
    
    return np.arcsin(U) # Return all angles

  def find_plane_corners(self, hand_landmarks, image_cols, image_rows):
    lm = hand_landmarks.landmark
    min_x, max_x = min([lm[0].x, lm[5].x, lm[17].x]), max([lm[0].x, lm[5].x, lm[17].x])
    min_y, max_y = min([lm[0].y, lm[5].y, lm[17].y]), max([lm[0].y, lm[5].y, lm[17].y])

    tl = mp_drawing._normalized_to_pixel_coordinates(min_x, min_y,
                                                   image_cols, image_rows)
  
    br = mp_drawing._normalized_to_pixel_coordinates(max_x, max_y,
                                                   image_cols, image_rows)
    return tl, br
  
  def draw_palm_box(self, image, landmarks,):
    image_rows, image_cols, _ = image.shape
    tl, br = self.find_plane_corners(landmarks, image_cols, image_rows)
    cv2.rectangle(image, tl, br, (54, 200, 219), 4)
    
# Handle ArgParse
parser = argparse.ArgumentParser()

parser.add_argument("--max-hands", default=2, help="Maximum number of hands tracked by MediaPipe")
parser.add_argument("--model-complexity", default=0, help="Model complexity (0 or 1). 0 is better inference performance whereas 1 is better model accuracy")
parser.add_argument("--min-detection-confidence", default=0.5, help="Minimum confidence required by the ML model to detect a landmark")
parser.add_argument("--min-tracking-confidence", default=0.5, help="Minimum confidence required by the ML model to track a landmark")
parser.add_argument("--joint-type", default="tip", choices=["mcp", "pip", "dip", "tip"] , help="Determine which joints to report, per finger (other than thumb). ")
parser.add_argument("--palm-viz", default=False, help="Determine whether or not to display a border-box containing the palm")


args = parser.parse_args()

runner = MPRunner("MPHands", args.joint_type, args.model_complexity, args.min_detection_confidence, args.min_tracking_confidence, args.max_hands, args.palm_viz)
runner.run_mp()
