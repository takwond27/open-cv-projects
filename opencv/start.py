# # # # import cv2
# # # # import numpy as np

# # # # # Load images
# # # # image1 = cv2.imread('SHHAPE1.jpg', cv2.IMREAD_GRAYSCALE)
# # # # image2 = cv2.imread('SHAPE2.jpg', cv2.IMREAD_GRAYSCALE)

# # # # # Threshold images
# # # # _, binary1 = cv2.threshold(image1, 127, 255, cv2.THRESH_BINARY)
# # # # _, binary2 = cv2.threshold(image2, 127, 255, cv2.THRESH_BINARY)

# # # # # Find contours
# # # # contours1, _ = cv2.findContours(binary1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# # # # contours2, _ = cv2.findContours(binary2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# # # # # Compare the first contour in each image
# # # # match_score = cv2.matchShapes(contours1[0], contours2[0], cv2.CONTOURS_MATCH_I3, 0)
# # # # print(f"Shape Similarity Score: {match_score}")

# # # # # Draw contours
# # # # output1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
# # # # out = output1
# # # # output2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

# # # # for cnt in contours1:
# # # #     # Approximate contour to get the number of sides
# # # #     epsilon = 0.02 * cv2.arcLength(cnt, True)
# # # #     approx = cv2.approxPolyDP(cnt, epsilon, True)
    
# # # #     # Get shape name based on number of sides
# # # #     shape_name = "Unknown"
# # # #     if len(approx) == 3:
# # # #         shape_name = "Triangle"
# # # #     elif len(approx) == 4:
# # # #         shape_name = "Rectangle"  # Could be square too
# # # #     elif len(approx) > 4:
# # # #         shape_name = "Circle"

# # # #     # Draw shape name
# # # #     x, y, w, h = cv2.boundingRect(cnt)
# # # #     cv2.putText(output1, shape_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
   
# # # #     cv2.drawContours(output1, [cnt], -1, (0, 255, 255), 2)

# # # # for cnt in contours2:
# # # #     # Approximate contour to get the number of sides
# # # #     epsilon = 0.02 * cv2.arcLength(cnt, True)
# # # #     approx = cv2.approxPolyDP(cnt, epsilon, True)
    
# # # #     # Get shape name based on number of sides
# # # #     shape_name = "Unknown"
# # # #     if len(approx) == 3:
# # # #         shape_name = "Triangle"
# # # #     elif len(approx) == 4:
# # # #         shape_name = "Rectangle"  # Could be square too
# # # #     elif len(approx) > 4:
# # # #         shape_name = "Circle"

# # # #     # Draw shape name
# # # #     x, y, w, h = cv2.boundingRect(cnt)
# # # #     cv2.putText(output2, shape_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
# # # #     cv2.drawContours(output2, [cnt], -1, (0, 255, 255), 2)

# # # # cv2.imshow("bina",binary1)
# # # # cv2.drawContours(out, contours1, -1, (0, 0, 255), 2)
# # # # # Show images
# # # # cv2.imshow('Shape 1', output1)
# # # # cv2.imshow('Shape 2', output2)
# # # # cv2.waitKey(0)
# # # # cv2.destroyAllWindows()

# # # # # Open webcam (0 is default webcam)
# # # # """cap = cv2.VideoCapture(0)

# # # # hile True:
# # # #     ret, frame = cap.read()  # Read frame
# # # #     if not ret:
# # # #         break

# # # #     cv2.imshow("Live Video", frame)  # Show video feed

# # # #     if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
# # # #         break

# # # # cap.release()  # Release camera
# # # # cv2.destroyAllWindows()  # Close all windows
# # # # import cv2"""

# # # # # cap = cv2.VideoCapture(0)
# # # # # #fgbg = cv2.createBackgroundSubtractorMOG2()  # Background subtractor
# # # # # a = None
# # # # # i = 0
# # # # # while True:
# # # # #     print(a)
# # # # #     ret, frame = cap.read()
# # # # #     if not ret:
# # # # #         break
# # # # #     i =+ 1
# # # # #     #fgmask = fgbg.apply(frame)  # Apply background subtraction
# # # # #     blac = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
# # # # #     _, thres = cv2.threshold(blac,127,255,cv2.THRESH_BINARY)
# # # # #     counter,_ = cv2.findContours(thres,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# # # # #     thres = cv2.cvtColor(thres,cv2.COLOR_GRAY2BGR)
# # # # #     cv2.drawContours(thres,counter,-1,(0,0,255),5)
# # # # #     motion = False
# # # # #     if i == 1:
# # # # #         a = counter
    
# # # # #     else:
# # # # #         b = counter
# # # # #         if a == b:
# # # # #             print("no motion")
# # # # #             motion = False
# # # # #         else:
# # # # #             if motion == 0:
# # # # #                 motion = True
# # # # #                 print("motin detected")
# # # # #                 a = b
# # # # #             else:
# # # # #                 print("motion continues")
# # # # #                 a = b

# # # # #     cv2.imshow("Original Video", frame)
# # # # #     #cv2.imshow("Motion Detection", fgmask)
# # # # #     cv2.imshow("thres",thres)  # Show movement mask

# # # # #     if cv2.waitKey(1) & 0xFF == ord('q'):
# # # # #         break

# # # # # cap.release()
# # # # # cv2.destroyAllWindows()

# # # # # cap = cv2.VideoCapture(0)
# # # # # ret, frame1 = cap.read()
# # # # # ret, frame2 = cap.read()

# # # # # while True:
# # # # #     diff = cv2.absdiff(frame1, frame2)  # Compute absolute difference between frames
# # # # #     gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)  # Convert difference to grayscale
# # # # #     #blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian blur to remove noise
# # # # #     _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  # Apply thresholding
# # # # #     #dilated = cv2.dilate(thresh, None, iterations=3)  # Dilate to fill gaps
# # # # #     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours

# # # # #     motion = False
# # # # #     for contour in contours:
# # # # #         if cv2.contourArea(contour) > 500:  # Ignore small changes
# # # # #             motion = True
# # # # #             (x, y, w, h) = cv2.boundingRect(contour)
# # # # #             cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw box around motion

# # # # #     if motion:
# # # # #         print("Motion Detected!")
# # # # #     else:
# # # # #         print("No Motion")

# # # # #     cv2.imshow("Original Video", frame1)
# # # # #     cv2.imshow("Threshold", thresh)

# # # # #     frame1 = frame2  # Update frame for next iteration
# # # # #     ret, frame2 = cap.read()
    
# # # # #     if not ret or cv2.waitKey(1) & 0xFF == ord('q'):
# # # # #         break

# # # # # cap.release()
# # # # # cv2.destroyAllWindows()
# # # # import cv2

# # # # # Load pre-trained face detection model
# # # # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # # # cap = cv2.VideoCapture(0)

# # # # while True:
# # # #     ret, frame = cap.read()
# # # #     if not ret:
# # # #         break
    
# # # #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
# # # #     faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect faces

# # # #     for (x, y, w, h) in faces:
# # # #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

# # # #     cv2.imshow("Face Detection", frame)

# # # #     if cv2.waitKey(1) & 0xFF == ord('q'):
# # # #         print(frame[0])
# # # #         break

# # # # cap.release()
# # # # cv2.destroyAllWindows()
# # import cv2
# # import numpy as np

# # # # Load image
# # # image = cv2.imread('secen1.jpg')
# # # 
# # # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # # # Apply Harris Corner Detection
# # # gray = np.float32(gray)
# # # dst = cv2.cornerHarris(gray, blockSize=5, ksize=3, k=0.0005)

# # # # Mark the corners
# # # image[dst > 0.01 * dst.max()] = [0, 255, 0]

# # # cv2.imshow('Harris Corners', image)
# # # cv2.waitKey(0)
# # # cv2.destroyAllWindows()
# # # Load image
# # image = cv2.imread('secen1.jpg')
# # image = cv2.resize(image,(1200,1200))
# # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # # Shi-Tomasi Corner Detector
# # corners = cv2.goodFeaturesToTrack(gray, maxCorners=50, qualityLevel=0.01, minDistance=10)

# # for corner in corners:
# #     x, y = np.int8(corner[0])
# #     cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

# # cv2.imshow('Shi-Tomasi Corners', image)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# import cv2

# # Load images
# img1 = cv2.imread('SHHAPE1.jpg', 0)  # Query image
# img2 = cv2.imread('SHAPE2.jpg', 0)  # Train image

# # ORB Detector
# orb = cv2.ORB_create()

# # Find keypoints and descriptors
# kp1, des1 = orb.detectAndCompute(img1, None)
# kp2, des2 = orb.detectAndCompute(img2, None)

# # Brute-Force Matcher
# bf = cv2.BFMatcher(cv2.NORM_TYPE_MASK, crossCheck=True)

# # Match descriptors
# matches = bf.match(des1, des2)

# # Sort matches by distance (lower distance = better match)
# matches = sorted(matches, key=lambda x: x.distance)

# # Draw matches
# result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)

# cv2.imshow('ORB Feature Matching', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# Read video
import cv2
import numpy as np
cap = cv2.VideoCapture('vedio.mp4')

# Read first frame
ret, frame1 = cap.read()

prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Create mask for visualization
hsv_mask = np.zeros_like(frame1)
hsv_mask[..., 1] = 255

while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Compute Dense Optical Flow
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Convert flow to HSV format
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv_mask[..., 0] = ang * 180 / np.pi / 2
    hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # Convert HSV to BGR
    bgr = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)

    cv2.imshow('Dense Optical Flow', bgr)

    prvs = next.copy()

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
