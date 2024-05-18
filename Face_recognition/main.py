# import the necessary packages
from __future__ import print_function
import argparse
import imutils


# -------------------------- References --------------------------
#   ***************************************************************************************/
#   *    Title: CNN and HOG
#   *    Author: Adrian Rosebrock
#   *    Date: 2021
#   *    Code version: n/a
#   *    Availability: https://pyimagesearch.com/2021/04/19/face-detection-with-dlib-hog-and-cnn/
#   *
#   ***************************************************************************************/
#   ***************************************************************************************/
#   *    Title: Face_Recognition tutorial
#   *    Author: Adam Geitgey
#   *    Date: 2018
#   *    Code version: n/a
#   *    Availability: https://github.com/ageitgey/face_recognition
#   *
#   ***************************************************************************************/
#   ***************************************************************************************/
#   *    Title: OpenCV Haar Cascades tutorial
#   *    Author: Vidhya Analytics
#   *    Date: 2020
#   *    Code version: n/a
#   *    Availability: https://medium.com/analytics-vidhya/facial-landmarks-and-face-detection-in-python-with-opencv-73979391f30e
#   *
#   ***************************************************************************************/
#   ***************************************************************************************/
#   *    Title: HOG_MultiScale tutorial
#   *    Author: Adrian Rosebrock
#   *    Date: 2015
#   *    Code version: n/a
#   *    Availability: https://pyimagesearch.com/2015/11/16/hog-detectmultiscale-parameters-explained/
#   *
#   ***************************************************************************************/
#   ***************************************************************************************/
#   *    Title: MTCNN tutorial 1/2
#   *    Author: Iván de Paz Centeno
#   *    Date: 2021
#   *    Code version: n/a
#   *    Availability: https://github.com/ipazc/mtcnn
#   *
#   ***************************************************************************************/
#   ***************************************************************************************/
#   *    Title: MTCNN tutorial 2/2
#   *    Author: Saransh Rajput
#   *    Date: 2020
#   *    Code version: n/a
#   *    Availability: https://medium.com/@saranshrajput/face-detection-using-mtcnn-f3948e5d1acb
#   *
#   ***************************************************************************************/



import cv2
from mtcnn import MTCNN
import os
import matplotlib.pyplot as plt
import time
import face_recognition
import dlib
import csv


images = ""
pics_path = ""
title = ""
faces = ""
cycle = 0
cycle_faces = 0
total_faces = 0
brightness_choice = -225
i = 0
data = [0] * 29


# ---------------------- Algorithm functions -----------------------
def HaarCascade(image):
    haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
    haarcascade = "haarcascade_frontalface_alt2.xml"

    # if (haarcascade in os.listdir(os.curdir)):
    #     print("File exists")
    # else:
    #     urlreq.urlretrieve(haarcascade_url, haarcascade)
    #     print("File downloaded")

    # create an instance of the Face Detection Cascade Classifier
    detector = cv2.CascadeClassifier(haarcascade)

    # Detect faces using the haarcascade classifier on the "grayscale image"
    faces = detector.detectMultiScale(image)

    for face in faces:
        (x,y,w,d) = face
        cv2.rectangle(image,(x,y),(x+w, y+d),(255, 0, 0), 30)

    return faces

def face_recog(image):
    faces = face_recognition.face_locations(image)

    for face in faces:
        cv2.rectangle(image, (face[3], face[0]),(face[1], face[2]), (255,0,0), 3)

    return faces

def mtcnn(image):
    detector = MTCNN()
    faces = detector.detect_faces(image)

    if faces:
        for face in faces:
            x, y, w, h = face['box']

        cv2.rectangle(image,(x,y),(x+w, y+h),(255, 0, 0), 3)

    return faces

def hog_multiscale(image):
    
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required=True,
    # 	help="path to the input image")
    ap.add_argument("-w", "--win-stride", type=str, default="(8, 8)",
        help="window stride")
    ap.add_argument("-p", "--padding", type=str, default="(16, 16)",
        help="object padding")
    ap.add_argument("-s", "--scale", type=float, default=1.05,
        help="image pyramid scale")
    ap.add_argument("-m", "--mean-shift", type=int, default=-1,
        help="whether or not mean shift grouping should be used")
    args = vars(ap.parse_args())

    # evaluate the command line arguments (using the eval function like
    # this is not good form, but let's tolerate it for the example)
    winStride = eval(args["win_stride"])
    padding = eval(args["padding"])
    meanShift = True if args["mean_shift"] > 0 else False
    # initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    # load the image and resize it
    new_image = imutils.resize(image, width=min(400, image.shape[1]))

    # detect people in the image
    (rects, weights) = hog.detectMultiScale(new_image, winStride=winStride,
        padding=padding, scale=args["scale"], useMeanshiftGrouping=meanShift)
    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)

    return rects

def dlib_HOG(image):

    # https://pyimagesearch.com/2021/04/19/face-detection-with-dlib-hog-and-cnn/

    detector = dlib.get_frontal_face_detector()
    faces = detector(image)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)

    return faces

def dlib_CNN(image):

    # https://pyimagesearch.com/2021/04/19/face-detection-with-dlib-hog-and-cnn/

    # https://towardsdatascience.com/cnn-based-face-detector-from-dlib-c3696195e01c

    detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

    faces = detector(image)

    for face in faces:
        x = face.rect.left()
        y = face.rect.top()
        w = face.rect.right() - x
        h = face.rect.bottom() - y
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)

    return faces


# ------------------------- Main functions -------------------------
def menu(speed, algorithm):
    global cycle
    cycle += 1


    # Resetting cycle faces to 0
    global cycle_faces
    cycle_faces = 0


    # Instant mode automatically increasing num
    global brightness_choice
    if speed != "i":
        print("State brightness % (increments of 25 from -200 to 500)")
        brightness_choice = int(input("Choice: "))
    else:
        if brightness_choice < 500:
            brightness_choice += 25
        else:
            print("Finished")
            exit()


    # Sort image path out with brightness level
    global images
    global pics_path
    pics_path = ("Face_recognition\Pics/" + str(brightness_choice) + "%")
    images=os.listdir(pics_path)
    images.sort()
    

    # Send user inputs to get images prepared
    for image in range(len(images)):
        prepare_image(speed, algorithm, image)


    # output cycle results once done
    results(cycle, brightness_choice)


    # repeat steps once cycle has been recorded
    menu(speed, algorithm)

def prepare_image(speed, algorithm, image_num):

    if speed != "i":
        print("Displaying image " + str(image_num))


    # Preparing the image
    if image_num >= len(images):
        global i
        i = 0
        image_num = 0
    active_image = images[image_num]
    file = pics_path + "/" + active_image
    image = cv2.imread(file)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    # Select correct algorithm with prepared image
    global title
    if algorithm == 1:
        faces = HaarCascade(image_rgb)
        title = "Haar Cascades"
    elif algorithm == 2:
        faces = face_recog(image_rgb)
        title = "face_recognition"
    elif algorithm == 3:
        faces = mtcnn(image_rgb)
        title = "mtcnn"
    elif algorithm == 4:
        faces = hog_multiscale(image_rgb)
        title = "hog_multiscale"
    elif algorithm == 5:
        faces = dlib_HOG(image_rgb)
        title = "dlib_HOG"
    elif algorithm == 6:
        faces = dlib_CNN(image_rgb)
        title = "dlib_CNN"


    # Instant = no messages, quick = messages, slow = visuals
    if speed == "i":
        global cycle_faces
        if len(faces) > 0:
            cycle_faces += 1
    else:
        if len(faces) > 0:
            cycle_faces += 1
            print("True")
            print("Faces:\n", faces)
        else:
            print("False")

        if speed == "s":
            plt.axis("off")
            plt.imshow(image_rgb)
            plt.title(title)
            plt.show()

def results(cycle, brightness):

    global total_faces
    global cycle_faces
    total_faces += cycle_faces
    print("Cycle " + str(cycle) + "(" + str(brightness) + "%): " + str(cycle_faces) + "/" + str(len(images)))
    print("Total: " + str(total_faces) + "/" + str(len(images) * cycle))


    # Storing data in array and exporting to Results.csv
    if speed == "i":

        global data
        data[cycle - 1] = (str(cycle_faces) + "/" + str(len(images)))

        # if finished 
        if brightness == 500:


            # Get the time taken to 2 dp
            total_time = time.time() - start_time
            time_taken = [float(f'{total_time:.2f}')]

            
            # Check if previous algorithm cycle exists 
            rows = []
            row_deleted = False
            with open("Results.csv", 'r', newline='') as file:
                reader = csv.reader(file)
                for row in reader:

                    # Delete the row
                    global title
                    if row[0] == title:
                        row_deleted = True
                        continue
                    rows.append(row)
                    
            # Write all rows back in except for the deleted row
            if row_deleted:
                with open("Results.csv", 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(rows)

            # Add new data
            with open("Results.csv", 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([title] + data + time_taken)
                
            


speed = input("Select speed (i = instant, q = quick, s = slow): ")
print("1) Haar Cascades\n2) face_recognition\n3) mtcnn\n4) hog_multiscale\n5) dlib_HOG\n6) dlib_CNN")
algorithm = int(input("Select which algorithm (number): "))
start_time = time.time()
menu(speed, algorithm)
