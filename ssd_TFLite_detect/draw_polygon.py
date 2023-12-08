
# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import importlib.util
import json

# Define and parse input arguments
parser = argparse.ArgumentParser()

parser.add_argument('--Width_video',help='Width_frame video' , default= 1080) 
parser.add_argument('--Height_video',help='Width_frame video' , default= 720) 
parser.add_argument('--video', help='Name of the video file',default='test.mp4')
parser.add_argument('--path_save_json', help='Path save polygon jon',default="polygon.json")

args = parser.parse_args()

VIDEO_NAME = args.video
WIDTH_VIDEO = int(args.Width_video)
HEIGHT_VIDEO = int(args.Height_video)

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to video file
VIDEO_PATH = os.path.join(CWD_PATH,VIDEO_NAME)


# Open video file
video = cv2.VideoCapture(VIDEO_PATH)

# click left mouse to add point for polygon of dict points left and right 
def handle_point_click(event, x, y, flags, points):
    if event == cv2.EVENT_LBUTTONDOWN:
        points['left'].append([x, y])
    if event == cv2.EVENT_RBUTTONDOWN:
        points['POINT_LEFT'] = [x, y]

# draw polygon left and right
def draw_polygon (frame, points):
    for point in points['left']:
        frame = cv2.circle( frame, (point[0], point[1]), 3, (255,0,0), -1)
    
    for point in points['right']:
        frame = cv2.circle( frame, (point[0], point[1]), 3, (0,255,0), -1)

    frame = cv2.polylines(frame, [np.int32(points['left'])], False, (255,0, 0), thickness=1)
    frame = cv2.polylines(frame, [np.int32(points['right'])], False, (0,255, 0), thickness=1)
    return frame


POINTS = {}
POINTS['left'] =  []
POINTS['right'] =  []
POINTS['POINT_RIGHT']= [0,0]
POINTS['POINT_LEFT']= [0,0]
point_medial = []

print("draw right road \n")
while(video.isOpened()):

    # Acquire frame and resize to expected shape [1xHxWx3]
    ret, frame = video.read()
    if not ret:
      print('Reached the end of the video!')
      break
    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame, (WIDTH_VIDEO, HEIGHT_VIDEO))
    # input_data = np.expand_dims(frame_resized, axis=0)

    # Ve ploygon
    frame = draw_polygon(frame_resized, POINTS)
    frame = cv2.circle( frame, (POINTS['POINT_RIGHT'][0], POINTS['POINT_RIGHT'][1]), 5, (0,255,255), -1)
    frame = cv2.circle( frame, (POINTS['POINT_LEFT'][0], POINTS['POINT_LEFT'][1]), 5, (255,0,255), -1)
    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame_resized)

    cv2.setMouseCallback('Object detector', handle_point_click, POINTS)
    # cv2.setMouseCallback('Object detector', handle_right_click, POINTS['POINT'] )

    if cv2.waitKey(1) == ord('p'):
        POINTS['POINT_RIGHT'] = POINTS['POINT_LEFT']

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break
    elif cv2.waitKey(1) == ord('t'):
        print("draw left road \n")
        POINTS['right'] = POINTS['left'] 
        POINTS['left'] = []

# Clean up
video.release()
cv2.destroyAllWindows()

print('POINT LEFT : ',POINTS['left'])
print('\n')
print('POINT RIGHT : ',POINTS['right'])

POINTS['size_width'] = WIDTH_VIDEO
POINTS['size_height'] = HEIGHT_VIDEO
print('\n')
print("SIZE SCREEN : ",POINTS['size_width'] ,POINTS['size_height'] )

# # Specify the file path where you want to save the JSON data

JSON_PATH = args.path_save_json

# Write the dictionary to a JSON file
with open(JSON_PATH, 'w') as json_file:
    json.dump(POINTS, json_file)