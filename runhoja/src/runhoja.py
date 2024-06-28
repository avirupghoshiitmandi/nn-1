#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import sys
class ImageSubscriber:
    def __init__(self):
        rospy.init_node('image_subscriber', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/image_raw', Image, self.image_callback)
        rospy.spin()

    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            print(e)
            return
        ARUCO_DICT = {
      "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
      "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
      "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
      "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
      "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
      "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
      "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
      "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
      "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
      "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
      "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
      "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
      "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
      "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
      "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
      "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
      "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
       } 

# Select the ArUco dictionary
        args = "DICT_6X6_100"
        if ARUCO_DICT.get(args) is None:
         
         sys.exit(0)

    # Load the ArUCo dictionary, grab the ArUCo parameters, and detect the markers
        arucoDict =  cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args])
        arucoParams = cv2.aruco.DetectorParameters_create()
        (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

    # Verify at least one ArUCo marker was detected0
    
        if len(corners) > 0:
        # Flatten the ArUCo IDs list
         ids = ids.flatten()
         for (markerCorner, markerID) in zip(corners, ids):
            # Extract the marker corners (which are always returned in top-left, top-right, bottom-right, and bottom-left order)
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            # Convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # Draw the bounding box of the ArUCo detection
            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

            # Compute and draw the center (x, y)-coordinates of the ArUCo marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

            # Draw the ArUCo marker ID on the image
            cv2.putText(image, str(markerID),
                        (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
           
        # Process your image for object detection here
        # Example: perform object detection using OpenCV or other libraries

        # Example: Display the image (you may remove this in actual deployment)
        cv2.imshow("Image", image)
        cv2.waitKey(1)  # Adjust as needed for displaying the image

def main():
    try:
        image_subscriber = ImageSubscriber()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
