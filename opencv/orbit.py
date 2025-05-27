import cv2
import numpy as np
import matplotlib.pyplot as plt
#from google.colab.patches import cv2_imshow

# Load the image in grayscale
image = cv2.imread('12jan2022sunspot.png', cv2.IMREAD_GRAYSCALE)
#cv2.imshow('image',image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
# Step 1: Contrast Enhancement using CLAHE
clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(5,5))
enhanced = clahe.apply(image)
#print("Contrast Enhanced Image:")
#cv2.imshow(enhanced)  # Display in Colab
# Step 2: Noise Reduction using Median Blur
blurred = cv2.medianBlur(enhanced, 17)
blurred = cv2.resize(blurred,(500,500))
print("Noise Reduced Image (Median Blur):")
#cv2.imshow("blurred_Image",blurred)  # Display in Colab
#cv2.waitKey(0)
#finding the sunspots
# Step 3: Edge Detection using Canny with Otsu’s thresholding
high_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU)  # Otsu’s method
low_thresh = 0.5 * high_thresh
edges = cv2.Canny(blurred, low_thresh, high_thresh)

print("Edge Detection Result:")


# Step 4: Crater Detection using Hough Circle Transform
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=25,
                           param1=high_thresh, param2=25, minRadius=15, maxRadius=100)

# Convert grayscale image to BGR for visualization
image = cv2.resize(image,(500,500))
output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Draw detected sunspots
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)  # Draw crater
        cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 2)  # Draw center

print("Detected Craters:")
cv2.imshow("detectedspots",output)  # Display in Colab
cv2.imshow("detectedspot",edges)  # Display in Colab
cv2.waitKey(0)