import mediapipe as mp #package for pose detection
import cv2 #package to access webcam

# Boundary indices
key_points = {
    'Left Eye' : 2,
    'Right Eye' : 5,
    'Nose' : 0,
    'Left Ear' : 7,
    'Right Ear' : 8,
    'Left Shoulder' : 11,
    'Right Shoulder' : 12,
}

def check_lean_forward(pose_coord, scale=1):
    """
        Checks whether an individual is leaning forward (head towards computer screen)
        :return: Boolean
            True if not leaning forward; False otherwise
    """
    if pose_coord[key_points['Right Shoulder']][0] != -1 and pose_coord[key_points['Right Ear']][0] != -1 \
            and pose_coord[key_points['Right Ear']][0] - pose_coord[key_points['Right Shoulder']][0]  < (100 * scale):
        return False
    if pose_coord[key_points['Left Shoulder']][0] != -1 and pose_coord[key_points['Left Ear']][0] != -1 \
            and  pose_coord[key_points['Left Shoulder']][0] - pose_coord[key_points['Left Ear']][0] < (100 * scale):
        return False
    return True


def check_slump(pose_coord, scale=1):
    """
        Checks whether a uses is slumped down in their chair (shoulders at nose level)
        :return: Boolean
            True if not slumped; False if slumped
    """
    if pose_coord[key_points['Right Shoulder']][1] != -1 and pose_coord[key_points['Nose']][1] != -1 \
            and (pose_coord[key_points['Right Shoulder']][1] - pose_coord[key_points['Nose']][1] <= (150 * scale)):
         return False
    return True

def check_head_drop(pose_coord, scale=1):
    """
        Checks whether a use has tilted their head downwards, chin towards chest (eyes are at ear level)
        :return: Boolean
            True if not head not tilted downwards; False if tilted downward
    """
    if pose_coord[key_points['Left Eye']][1] != -1 and pose_coord[key_points['Left Ear']][1] != -1 \
            and pose_coord[key_points['Left Ear']][1] - pose_coord[key_points['Left Eye']][1] <= (5 * scale):
        return False
    if pose_coord[key_points['Right Eye']][1] != -1 and pose_coord[key_points['Right Ear']][1] != -1 \
            and pose_coord[key_points['Right Ear']][1] - pose_coord[key_points['Right Eye']][1] <= (5 * scale):
        return False
    return True

def correct_posture(pose_coord):
    """
        Checks all current posture functions
        :return: Boolean
            True if all posture functions return True; False otherwise
    """
    return all([check_slump(pose_coord), check_head_drop(pose_coord), check_lean_forward(pose_coord)])


# Building warning message
def build_message(pose_coord):
    """
        Builds a string with advice to the user on how to correct their posture
        :return: string
            The string containing specific advice
    """
    current_message = ""
    if not check_head_drop(pose_coord):
        current_message += "Lift up your head!\n"
    if not check_lean_forward(pose_coord):
        current_message += "Lean back!\n"
    if not check_slump(pose_coord):
        current_message += "Sit up in your chair, you're slumping!\n"
    return current_message

def detect_landmark(img, results, draw=False):
    """
            Detects Landmarks required for post estimation
            :return: List
                The list containing tuples of coordinates
    """
    img_height, img_width= img.shape[:2]
    pose_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.pose_landmarks.landmark]

    if draw :
        if not check_lean_forward(pose_coord):
            [cv2.line(img, (pose_coord[key_points['Right Shoulder']]), (pose_coord[key_points['Right Ear']]), (0,0,255), 3, 8)]
            [cv2.line(img, (pose_coord[key_points['Left Shoulder']]), (pose_coord[key_points['Left Ear']]), (0,0,255), 3, 8)]
        
        if not check_head_drop(pose_coord):
            [cv2.line(img, (pose_coord[key_points['Left Eye']]), (pose_coord[key_points['Left Ear']]), (0,0,255), 3, 8)]
            [cv2.line(img, (pose_coord[key_points['Right Eye']]), (pose_coord[key_points['Right Ear']]), (0,0,255), 3, 8)]
        if not check_slump(pose_coord):
            [cv2.line(img, (pose_coord[key_points['Right Shoulder']]), (pose_coord[key_points['Nose']]), (0,0,255), 3, 8)]
    return pose_coord