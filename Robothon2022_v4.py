# Robothon Grand Challenge 2022 - Team RoboTechX
# Detecting blue button and keyhole using Hough algorithm
# by Judhi j.pprasetyo@mdx.ac.ae
# rev 4 01-06-2022 speed up camera setup using DSHOW on Windows system
# rev 3 29-05-2022 adjusted HSV values to keep up with various lights
# rev 2 28-05-2022 added HSV conversion to filter out based on colours
# rev 1 19-05-2022 detecting circles based on radius

print("Starting, loading modules...")
from pickle import FALSE, FRAME
import cv2
import numpy as np
import socket
from time import sleep
print("Modules loaded")

# camera intrinsic matrix taken from OpenCV camera calibration
# ref: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
cam_mtx = np.array( [[6.06240775e+03, 0.00000000e+00, 9.99979660e+02],
 [0.00000000e+00, 5.70798372e+03, 1.06812022e+03],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

new_mtx = np.array([[5.56844434e+03, 0.00000000e+00, 1.00948419e+03],
 [0.00000000e+00, 5.56374512e+03, 1.01977920e+03],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

dist = np.array([[-2.35194297e+00, -1.34158664e+01, -2.35690227e-01,  1.62212622e-03,
   2.22389791e+02]])


# function for mapping pixel to world coordinate
# the formula is derived from polynomial trend line from robot world coordinate calibration
def calculateXY(xc, yc):
    X = -0.00001*xc**2 - 0.254*xc + 354.87 
    Y =0.00001*yc**2 - 0.2786*yc - 429.06 
    return round(Y,3), round(X,3)

# define a video capture object
print("Starting camera")
vid = cv2.VideoCapture(2,cv2.CAP_DSHOW) # activate Windows Direct Show for faster camera setup
# vid = cv2.VideoCapture(0) # for other systems

print("Setting video resolution")
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) # max 3840 for 4K, 1920 for FHD
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) # max 2160 for 4K, 1080 for FHD

n_frame = 1 # frame counter

while(vid.isOpened()):
    print("=======Ready to capture=======")   
    # Capture the video frame by frame
    print("Capturing frame")
    ret, img_dist = vid.read()

    # undistorting image
    print("Undistorting")
    img = cv2.undistort(img_dist, cam_mtx, dist, None, new_mtx)
    
    # Display the resulting frame
    #cv2.putText(img, "Frame "+str(n_frame), (50,50), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2 )
    #cv2.imshow('Frame '+str(n_frame), img)
    #cv2.waitKey(0)

    # convert to HSV for color detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # ----------------- daylight --------------
    #lower_blue = np.array([75, 56, 56])
    #higher_blue = np.array([115, 255, 205])
    #lower_blue = np.array([75, 72, 129])  
    #higher_blue = np.array([115, 253, 245])

    # --------------- most reliable params so far (29-05-2022) --------------
    lower_blue = np.array([70, 108, 88])
    higher_blue = np.array([136, 255, 255])
    lower_bright = np.array([0, 0, 0])
    higher_bright = np.array([180, 255, 114])

    # getting the range of blue color in frame
    blue_range = cv2.inRange(hsv, lower_blue, higher_blue)
    bright_range = cv2.inRange(hsv, lower_bright, higher_bright)
    res_blue = cv2.bitwise_and(img,img, mask=blue_range)
    res_bright = cv2.bitwise_and(img,img, mask=bright_range)

    # Convert to grayscale and 
    # Blur using 3 * 3 kernel.
    print("Converting to grayscale")
    gray_blue = cv2.cvtColor(res_blue, cv2.COLOR_BGR2GRAY)
    gray_blue_blurred = cv2.blur(gray_blue, (3, 3))
    gray_bright = cv2.cvtColor(res_bright, cv2.COLOR_BGR2GRAY)
    gray_bright_blurred = cv2.blur(gray_bright, (3,3))

    # Apply Hough transform on the blurred image. One for the blue button, one for the keyhole
    print("Detecting circles")
    detected_blue_circles = cv2.HoughCircles(gray_blue_blurred, 
                    cv2.HOUGH_GRADIENT, 0.5, 1000, param1 = 75, #55
                param2 = 20, minRadius = 19, maxRadius = 73)

    detected_bright_circles = cv2.HoughCircles(gray_bright_blurred, 
                    cv2.HOUGH_GRADIENT, 0.5, 1000, param1 = 25, #55
                param2 = 20, minRadius = 25, maxRadius = 31)

    # Draw circles if detected.
    print("Drawing circles")
    if detected_blue_circles is not None and detected_bright_circles is not None:
    
        # Convert the circle parameters a, b and r to integers.
        detected_blue_circles = np.uint16(np.around(detected_blue_circles))
        detected_bright_circles = np.uint16(np.around(detected_bright_circles))

        for pt in detected_blue_circles[0]:
            a1, b1, r1 = pt[0], pt[1], pt[2]

            # Draw the circle
            cv2.circle(img, (a1, b1), r1, (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(img, (a1, b1), 1, (0, 0, 255), 3)
            # add text label
            cv2.putText(img, "(" + str(a1) + ","+ str(b1) + ") r=" + str(r1), (a1+r1+2,b1+10), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2 )

        for pt2 in detected_bright_circles[0]:
            a2, b2, r2 = pt2[0], pt2[1], pt2[2]
            # Draw the circle
            cv2.circle(img, (a2, b2), r2, (0, 255, 0), 2)
            # Draw the center of the circle 
            cv2.circle(img, (a2, b2), 1, (0, 0, 255), 3)
            # add text label
            cv2.putText(img, "(" + str(a2) + ","+ str(b2) + ") r=" + str(r2), (a2+r2+2,b2+10), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2 )
        
        # add lable for human input to start the robot OR quit OR recapture image
        cv2.putText(img, "Press 'g' = start robot, 'q' = quit, other key = recapture", (50,100), cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,255),2)
        
        # check the ditance between the Blue button and the Keyhole
        pixel_distance = np.sqrt((int(a2)-int(a1))**2 + (int(b2)-int(b1))**2)
        print("Pixel distance : " + str(round(pixel_distance,0)) ) 
        # resize the image to show
        img_r = cv2.resize(img,(860,540))
        cv2.imshow("Detected Circle", img_r)

        # points only valid if the distance above is more than 550 pixels (FHD) 
        if (pixel_distance > 550) :
            print("Blue Button:")
            print("Pixel is at x=" + str(a1) + "  y="+ str(b1) + " r=" + str(r1))
            x, y = calculateXY(a1, b1)
            print("World coordinate is at x mm= " + str(x) + "  y mm= "+ str(y))
            print("-----")
            print("Keyhole:")
            print("Pixel is at x=" + str(a2) + "  y="+ str(b2) + " r=" + str(r2))
            x1, y1 = calculateXY(a2, b2)
            print("World coordinate is at x mm= " + str(x1) + "  y mm= "+ str(y1))
            print("MAKE SURE ROBOT IS READY!")
            print("Press 'g' to start robot")
            print("Press 'q' to quit")
            print("Any other key to re-capture image")
            # waiting for human's input
            k = cv2.waitKey(0)

            # if human selected to run the robot
            if k == ord('g'):
                # Create a client socket
                clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM);
                # Connect to the VT6 robot
                clientSocket.connect(("192.168.156.2",2001));
                # Send data to VT6 robot
                data = "DATA " + str(x) + " " + str(y) + " " + str(x1) + " " + str(y1) + "\r\n"
                clientSocket.send(data.encode())
                sleep(1)
                # Close the connection to start the robot
                clientSocket.close

            # if human selected to quit
            if k == ord('q'):
                break

            # any other key will bring back to the beginning of the loop / recapture image
        
        # otherwise the points are invalid
        else:
            print("Blue button or Keyhole not found")

    # otherwise no circle object were found, go back to recapture image
    else:
        print("Sorry, no item found")

    # increase frame counter
    n_frame = n_frame + 1

# end