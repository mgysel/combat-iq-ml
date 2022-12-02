#YOLOv7 Find Angle Function
#By Augmented Startups
#Visit www.augmentedstartups.com
import math
import cv2

def getCoordinates(kpts):
    '''
    Helper function that converts keypoints output from run_pose 
    to more useful format
    '''
    coord = []
    no_kpt = len(kpts) // 3
    for i in range(no_kpt):
        cx, cy = kpts[3*i], kpts[3*i + 1]
        conf = kpts[3 * i + 2]
        coord.append([i, cx, cy, conf])
    
    return coord

def getXY(coordinates, pt):
    '''
    Gets the x, y coordinates for a specific point
    '''
    x, y = coordinates[pt][1:3]
    return (x, y)

def getDistance(coordinates, p1, p2):
    '''
    Gets the distance between two points p1 and p2
    # NOTE: KEYPOINT MAP HERE: https://github.com/JRKagumba/2D-video-pose-estimation-yolov7
    '''
    x1, y1 = coordinates[p1][1:3]
    x2, y2 = coordinates[p2][1:3]
    distance = math.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
    return distance

# =3.0=Takes three points and returns the angle between them========
def getAngle(coordinates, points):
    '''
    Determines angle between two points
    Coordinates: all of the coordinates of every point
    Points: (p1, p2, p3) tuple of three body part points
    # NOTE: KEYPOINT MAP HERE: https://github.com/JRKagumba/2D-video-pose-estimation-yolov7
    # Example: (6, 8, 10) would determine the angle of the right elbow
    '''
    p1, p2, p3 = points[0], points[1], points[2]

    # =3.1=Get landmarks========
    x1, y1 = coordinates[p1][1:3]
    x2, y2 = coordinates[p2][1:3]
    x3, y3 = coordinates[p3][1:3]

    # =3.2=Calculate the Angle========
    angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))

    if angle < 0:
        angle += 360
    if angle > 180:
        angle = 360 - angle

    return int(angle)

# NOTE: OTHER ANGLES
# shoulder to hand

def getImportantAngles(coordinates):
    '''
    Gets the most significant angles which are used to determine punch type
    # NOTE: KEYPOINT MAP HERE: https://github.com/JRKagumba/2D-video-pose-estimation-yolov7
    # Right arm=(6,8,10)
    # left arm=(5,7,9)
    # Right-arm to shoulder (5, 6, 8)
    # Left arm to shoulder (6, 5, 7)
    '''
    right_elbow = getAngle(coordinates, (6, 8, 10))
    left_elbow = getAngle(coordinates, (5, 7, 9))
    right_arm_shoulder = getAngle(coordinates, (5, 6, 8))
    left_arm_shoulder = getAngle(coordinates, (6, 5, 7))

    return [right_elbow, left_elbow, right_arm_shoulder, left_arm_shoulder]

def getImportantDistances(coordinates):
    '''
    Gets the most significant distances which are used to determine punch type
    # NOTE: KEYPOINT MAP HERE: https://github.com/JRKagumba/2D-video-pose-estimation-yolov7
    # right arm to right shoulder = (6,10)
    # left arm to left shoulder =(5,9)
    # Right arm to right hip (10, 12)
    # Left arm to left hip (9, 11)
    # FOR A STATIC DISTANCE, left/right shoulder to left/right hip
    # Right shoulder to right hip = (6, 12)
    # Left shoulder to left hip = (5, 11)
    '''
    right_arm_shoulder = getDistance(coordinates, 6, 10)
    left_arm_shoulder = getDistance(coordinates, 5, 9)
    right_arm_hip = getDistance(coordinates, 10, 12)
    left_arm_hip = getDistance(coordinates, 9, 11)
    right_shoulder_hip = getDistance(coordinates, 6, 12)
    left_shoulder_hip = getDistance(coordinates, 5, 11)

    return [right_arm_shoulder, left_arm_shoulder, right_arm_hip, left_arm_hip, right_shoulder_hip, left_shoulder_hip]

def getImportantCoordinates(coordinates):
    '''
    Gets the most signifiant coordinates (x, y) for different points
    Right hand = 10
    Left hand = 9
    Right shoulder = 6
    Left shoulder = 5
    Right hip = 12
    Left hip = 11
    '''
    x1, y1 = coordinates[10][1:3]
    x2, y2 = coordinates[9][1:3]
    x3, y3 = coordinates[6][1:3]
    x4, y4 = coordinates[5][1:3]
    x5, y5 = coordinates[12][1:3]
    x6, y6 = coordinates[11][1:3]
    
    return [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6)]

# ===========================================

def drawImportantAngles(image, coordinates):
    '''
    Draws angles between shoulders and arms
    # NOTE: KEYPOINT MAP HERE: https://github.com/JRKagumba/2D-video-pose-estimation-yolov7
    # Right arm=(6,8,10)
    # left arm=(5,7,9)
    # Right-arm to shoulder (5, 6, 8)
    # Left arm to shoulder (6, 5, 7)
    '''
    # points
    drawCircle(image, coordinates, 5)
    drawCircle(image, coordinates, 6)
    drawCircle(image, coordinates, 7)
    drawCircle(image, coordinates, 8)
    drawCircle(image, coordinates, 9)
    drawCircle(image, coordinates, 10)

    # Shoulder
    drawLine(image, coordinates, 5, 6)
    # Left arm
    drawLine(image, coordinates, 5, 7)
    drawLine(image, coordinates, 7, 9)
    # Right arm
    drawLine(image, coordinates, 6, 8)
    drawLine(image, coordinates, 8, 10)

def drawImportantAngleText(draw, coordinates):
    impAngles = getImportantAngles(coordinates)

    # d represents the distance the text is drawn from the keypoint
    d = 20
    # Right Elbow
    x, y = getXY(coordinates, 8)
    draw.text((x+d, y+d), f"{impAngles[0]}, ({x}, {y})", fill=(255, 255, 255))
    # Left Elbow
    x, y = getXY(coordinates, 7)
    draw.text((x+d, y+d), f"{impAngles[1]}, ({x}, {y})", fill=(255, 255, 255))
    # Right Shoulder
    x, y = getXY(coordinates, 6)
    draw.text((x+d, y+d), f"{impAngles[2]}, ({x}, {y})", fill=(255, 255, 255))
    # Left Shoulder
    x, y = getXY(coordinates, 5)
    draw.text((x+d, y+d), f"{impAngles[3]}, ({x}, {y})", fill=(255, 255, 255))

def drawAngle(image, coordinates, p1, p2, p3):
    '''
    Draws angle between three points
    p1/p2/p3: corresponds to specific body point number
    (# NOTE: KEYPOINT MAP HERE: https://github.com/JRKagumba/2D-video-pose-estimation-yolov7)
    '''
    drawCircle(image, coordinates, p1)
    drawCircle(image, coordinates, p2)
    drawCircle(image, coordinates, p3)
    drawLine(image, coordinates, p1, p2)
    drawLine(image, coordinates, p2, p3)

def drawCircle(image, coordinates, p1):
    '''
    Used to draw a circle for a specific point
    p1: corresponds to specific body point number
    (# NOTE: KEYPOINT MAP HERE: https://github.com/JRKagumba/2D-video-pose-estimation-yolov7)
    ex: p1=5 corresponds to left shoulder
    '''
    x1, y1 = coordinates[p1][1:3]
    cv2.circle(image, (int(x1), int(y1)), 10, (255, 255, 255), cv2.FILLED)
    cv2.circle(image, (int(x1), int(y1)), 20, (235, 235, 235), 5)

def drawLine(image, coordinates, p1, p2):
    '''
    Used to draw a line from one point ot another
    p1/p2: correspond to specific body point number
    (# NOTE: KEYPOINT MAP HERE: https://github.com/JRKagumba/2D-video-pose-estimation-yolov7)
    ex: p1=5 corresponds to left shoulder and p2=6 corresponds to right shoulder
    this would draw a line from left shoulder to right shoulder if p1=5 and p2=6
    '''   
    x1, y1 = coordinates[p1][1:3]
    x2, y2 = coordinates[p2][1:3]
    cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 3)