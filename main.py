import cv2
import os
import line_detection
import number_detection

DIRECTORY_PATH = 'assets'
OUT_PATH = 'test/out.txt'

STUDENT = 'RA 126/2015 Jelena Tadic\n'
HEADER = 'file\tsum\n'

def main():
    out_file = open(OUT_PATH,'w')
    out_file.write(STUDENT)
    out_file.write(HEADER) 
    for filename in os.listdir(DIRECTORY_PATH):

        video = cv2.VideoCapture(os.path.join(DIRECTORY_PATH , filename))  
        blue_line, green_line = line_detection.get_blue_and_green_line(video)
        print(f'Started counting {filename}...', end='')
        result = number_detection.calculate_numbers(video, blue_line, green_line)
        out_file.write(f'{filename}\t{result}\n')
        video.release()
        print(f'Finished counting {filename}')
    out_file.close()

main()