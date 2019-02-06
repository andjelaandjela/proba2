import cv2
import numpy as np

def get_blue_and_green_line(video):
    blue_line = None
    green_line = None
    while True:
        ret,  frame = video.read()     
        if ret is not True:
            break
        blue, green = cv2.split(frame)[0:2]
        if blue_line is None:  
            blue_line = get_line(blue)
        if green_line is None:
            green_line = get_line(green)
        if  blue_line is not None and green_line is not None:
            video.set(cv2.CAP_PROP_POS_FRAMES , 0)
            return blue_line, green_line

def get_line( channel): 
    blurred = cv2.GaussianBlur(channel, (7, 7), 0) # Bluring channel
    erosion_first_hand = cv2.erode(blurred, np.ones((3, 3))) # Erode two times
    erosion_fine = cv2.erode(erosion_first_hand, np.ones((2, 2)))
    binary = cv2.threshold(erosion_fine,70,255,cv2.THRESH_BINARY)[1] # Make binary image (black or white pixel)
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180, 100,250, 50)
    line_selected = None
    if lines is not None:
        line_selected = lines[0][0]
        for line in lines:
            if(line[0][0] < line_selected[0] or line[0][1] > line_selected[1] ):
                line_selected[0] = line[0][0]
                line_selected[1] = line[0][1]
            if(line[0][2] > line_selected[2] or line[0][3] < line_selected[3] ):
                line_selected[2] = line[0][2]
                line_selected[3] = line[0][3]
    return line_selected