import cv2
import numpy as np
# Reading an image
img_path = "deepspace2.jpg"
img = cv2.imread(img_path)
# Converting to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Returns numpy array
cv2.imshow("gray_img",gray)
# Applying Gaussian Blur
img2 = cv2.GaussianBlur(img, (5, 5), 0)  # Kernel size (5,5), standard deviation 0
gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
cv2.imshow("gaussian_blur",img2)
cv2.imshow("gaussian_blur_gray",gray_img2)
#removing noise
kernel = np.ones((3,3), np.uint8)
#kernel1 = np.ones((5,5), np.uint8)
eroded = cv2.erode(gray, kernel, iterations=1)
#eroded1 = cv2.erode(gray, kernel1, iterations=1)
cv2.imshow("rem_noise",eroded)
#cv2.imshow("rem_noise1",eroded1)
#tresholding
_,thresh = cv2.threshold(eroded, 150, 255, cv2.THRESH_BINARY)#to find boundary
cv2.imshow("thresholded_img",thresh)
#edge detection
# Step 3: Edge Detection using Canny with Otsu’s thresholding
high_thresh, _ = cv2.threshold(eroded, 0, 255, cv2.THRESH_OTSU)  # Otsu’s method
low_thresh = 0.5 * high_thresh
edges = cv2.Canny(thresh, 100, 200)  # Threshold values 100 and 200
cv2.imshow("edge_detection",edges)
# Step 4: edge Detection using Hough Circle Transform
contours, hierarchy = cv2.findContours(edges,  
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

cv2.drawContours(img, contours, -1, (0, 255, 0), 3) 
cv2.imshow("final",img)

cv2.waitKey(0)