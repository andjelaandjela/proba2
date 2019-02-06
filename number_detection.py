import cv2
import math
import numpy as np
from neural_network import NeuralNetwork

INFO = False

class Number:
    def __init__(self, x, y, width, height, predictions ):
        self.x = x
        self.y = y
        self.w = width
        self.h = height
        self.predictions = predictions
        self.is_counted_plus = False
        self.is_counted_minus = False
        self.not_move = 0
        self.update = False
        self.count =0
    
    def updatePosition(self,x,y):
        if  math.sqrt(math.pow(x-self.x,2) + math.pow(y-self.y,2)) < 18 :
            self.x = x
            self.y = y
            self.count += 1
            self.update = True
            self.not_move = 0
            return True
        return False 
    
    def update_predictions(self, predictions):
        if self.count % 5 == 0:
            if max(predictions) > max(self.predictions): 
                predicitions = predictions

    def count_plus(self):
        self.is_counted_plus = True
        return np.argmax(self.predictions)
    
    def count_minus(self):
        self.is_counted_minus = True
        return np.argmax(self.predictions)
    def __str__(self):
            return f'{self.x},{self.y} -> {np.argmax(self.predictions)}'

def find_numbers(frame):
    number_regions = []
    red = cv2.split(frame)[2]
    blurred = cv2.GaussianBlur(red, (7, 7), 0)
    erosion = cv2.erode(blurred, np.ones((2,2)))
    dilation = cv2.dilate(erosion, np.ones((4,4)))
    binary =cv2.threshold(dilation,70, 255, cv2.THRESH_BINARY)[1]
    
    contours =  cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour) 
        if h > 11 and w > 2:
            number_regions.append((x,y,w,h))
    return number_regions

def check_number_position(number, line):
    return (
        ( 
            (number.x + number.w >= line[0] and number.x + number.w <= line[2] and number.y <= line[1] and number.y >= line[3]       ) 
            or
            (number.x  >= line[0] and number.x <= line[2] and number.y + number.h <= line[1] and number.y + number.h >= line[3]) 
        ) and ( 
            (number.y - line[1] >= ((line[3] - line[1]) / (line[2] - line[0])) * (number.x + number.w - line[0] )  )
            or
            (number.y + number.h - line[1] >= (( line[3] - line[1]) / (line[2] - line[0])) * (number.x - line[0]) )
        )   
    )

def calculate_numbers(video, blue_line, green_line):
    numbers = []
    result = 0
    neural_network = NeuralNetwork()

    while True:
        ret, frame = video.read()
        if ret is not True:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray,70,255,cv2.THRESH_BINARY)[1] 
        number_regions =  find_numbers(frame)
        if INFO:
            cv2.line(frame, (blue_line[0] , blue_line[1]), (blue_line[2] , blue_line[3]) , (0,0,255),2)
            cv2.line(frame, (green_line[0] , green_line[1]), (green_line[2] , green_line[3]) , (0,0,255),2)
        for number_region in number_regions:
            x, y, w, h = number_region
            
            number_image = binary[y-2:y+h+2, x-2:x+w+2]  
            
            resized_number_image = cv2.resize(number_image.copy(), (28,28))
            reshaped_number_image = resized_number_image.reshape(1,28,28,1)
            predictions = neural_network.predict_number(reshaped_number_image)
            
            existing_number = None
            for number in numbers:
                if number.updatePosition(x,y):
                    existing_number = number
                    break
                    
            if existing_number is None:  
                if x < 80 or y <80 or len(numbers) == 0:                        
                    number = Number(x,y,w,h, predictions)
                    numbers.append(number)
            else:
                existing_number.update_predictions(predictions)
            if INFO:
                cv2.rectangle(frame, (x + w , y), (x,y+h) , (0,0,255),1)
        delete_numbers = []
        for number in numbers:
            if not number.update:
                number.not_move +=1
            if number.not_move > 20:
                delete_numbers.append(number)
        for number in delete_numbers:
            numbers.remove(number)            
        for number in numbers:
            if not number.is_counted_plus and check_number_position(number, blue_line):
                result += number.count_plus() 
            if not number.is_counted_minus and check_number_position(number, green_line):
                result -= number.count_minus() 
            number.update = False
        
        if INFO:
            cv2.imshow('frame', frame)
            for number in numbers:
                if True:
                    print('[' + str(number) , end='] ')
            print(f'Count: {len(numbers)}')
            print(f'Result: {result}')

            if cv2.waitKey(0) & 0xFF == ord('q'):
                pass
    return result

        
