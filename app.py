import mediapipe as mp # package for pose detection
import cv2 # package for computer vision
import time # python time module
import threading # run multiple processes at the same time
from gtts import gTTS # google text-to-speech module
import os # importing os module which provides functions for interacting with the operating system

# importing self defined utility functions
from util import *

# configuring mediapipe functions
mp_drawing = mp.solutions.drawing_utils #used to detect our pose landmarks
mp_drawing_styles = mp.solutions.drawing_styles 
mp_holistic = mp.solutions.holistic #integrates separate models for pose, face and hand components, each of them are optimized   for their particular domain

# Processes Real time feed to analyse frames
def cam_feed(cap):

    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():

            # handling errors
            # if any errors is occured during the running of the program
            # then it handles
            try:

                success, img = cap.read()

                if not success:
                    print("Unable to load video.")

                img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC) # used to estimate unknown data points between two known data points
                # mostly used to impute missing values in the dataframe or series while preprocessing data

                # Recolor Feed
                img.flags.writeable = False # False locks the data,and hence making it read-only
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Make Detections
                results = holistic.process(img) #analyse image details and return them from holistic model

            except not results:
                continue

            # Recolor image back to BGR for rendering
            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Right Hand
            mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Left Hand
            mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


            # Pose Detections
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())


            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('Real-time Webcam', cv2.flip(img, 1))

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

# Function to check Posture
def check(cap):

    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():

            # handling errors
            # if any erros is occured during the running of the program
            # then it handles
            try:
                success, img = cap.read()

                if not success:
                    print("Unable to load video.")
                    # If loading a video, use 'break' instead of 'continue'.

                img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

                # Recolor Feed
                img.flags.writeable = False
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Make Detections
                results = holistic.process(img)

                # Recolor image back to BGR for rendering
                img.flags.writeable = True
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # instance is created
                coord = detect_landmark(img, results, True)

            except AttributeError:
                continue

            posture = correct_posture(coord)
            if not posture:
                text = build_message(coord)
                # speech = gTTS(text = text, lang = 'en', slow = False)
                # speech.save("text.mp3")
                # os.system("start text.mp3")
                print(text)

                # Flip the image horizontally for a selfie-view display.
            cv2.imshow('Posture Corrector', cv2.flip(img, 1))
                # time.sleep(20)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

def main():

    # Getting Real-time Webcam feed
    cap = cv2.VideoCapture(0)

    # Thread to run multiple processes at the same time
    t1 = threading.Thread(target=cam_feed, args=[cap])
    t2 = threading.Thread(target=check, args=[cap])

    # Thread starts
    t1.start()
    t2.start()

    # Waits for both threads to end before releasing cap
    t1.join()
    t2.join()

    # Destroys Video Capture object and closes the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()