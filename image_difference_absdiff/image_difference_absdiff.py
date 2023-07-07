import cv2

# Load images as grayscale
image1 = cv2.imread("./images/1.jpg", 0)
image2 = cv2.imread("./images/2.jpg", 0)

# Calculate the per-element absolute difference between 
# two arrays or between an array and a scalar
diff = 255 - cv2.absdiff(image1, image2)

cv2.imshow('diff', diff)
cv2.waitKey()