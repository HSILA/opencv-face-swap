import cv2
import numpy as np
import argparse
import json

parser = argparse.ArgumentParser(description='Point Selector')
parser.add_argument('--input', type=str, help='input image path.')
parser.add_argument('--output', type=str, default='points.json', help='output json path.')
parser.add_argument('--max', type=int, default=10, help='max points.')
args = parser.parse_args()

MAX_POINTS = args.max
IMAGE = cv2.imread(args.input)
points = np.zeros((MAX_POINTS, 2),np.int)
counter = 0

def mousePoints(event, x, y, flags, params):
    global counter
    if event == cv2.EVENT_LBUTTONDOWN and counter < MAX_POINTS:
        points[counter] = int(x), int(y)
        counter = counter + 1
        print(points)

def select():
    while True:
        if counter == MAX_POINTS:
            with open(args.output, 'w') as file:
                json.dump(points.tolist(), file)
            break

        for x in range(0, MAX_POINTS):
            cv2.circle(IMAGE,(points[x][0], points[x][1]),3,(0,255,0),cv2.FILLED)

        cv2.imshow("Original Image", IMAGE)
        cv2.setMouseCallback("Original Image", mousePoints)
        cv2.waitKey(1)

if __name__ == '__main__':
    select()