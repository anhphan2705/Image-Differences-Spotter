from skimage.metrics import structural_similarity
import cv2
import numpy as np

def get_image(directory):
    print("[Console] Getting image")
    return cv2.imread(directory)

def show_image(header, image):
    print("[Console] Showing image")
    cv2.imshow(header, image)
    cv2.waitKey()
    
def write_image(directory, image):
    print("[Console] Saving image")
    cv2.imwrite(directory, image)

def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def convert_to_cv2_format(image):
    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type in the range [0,1] 
    # so we must convert the array to 8-bit unsigned integers in the range
    # [0,255] before we can use it with OpenCV
    image = (image * 255).astype("uint8")
    return image

def get_threshold(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

def get_contours(image):
    print("[Console] Finding contours")
    threshold_img = get_threshold(image)    
    contours = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    return contours
    
def get_structural_simlarity(first_image, second_image):
    print("[Console] Calculating differences")
    # Convert to gray for comparison
    first_gray = convert_to_gray(first_image)
    second_gray = convert_to_gray(second_image)
    # Compare
    (score, diff_img) = structural_similarity(first_gray, second_gray, full=True)
    # Convert return format to cv2 readable
    diff_img = convert_to_cv2_format(diff_img)
    print("[Console] Similarity score of {:.4f}%".format(score * 100))
    return diff_img

def get_diff_mask(image, diff_image):
    mask = np.zeros(image.shape, dtype='uint8')
    contours = get_contours(diff_image)
    # Multiple objects in a contours
    for c in contours:
        area = cv2.contourArea(c)
        # For any object that has a contour area > 40 pixels
        if area > 40:
            # Mark it
            x,y,w,h = cv2.boundingRect(c)
            cv2.drawContours(mask, [c], 0, (255,255,255), -1)
    return mask

def get_diff_rect(image, diff_image):
    print("[Console] Drawing rectangle around the differences")
    img = image.copy()
    contours = get_contours(diff_image)
    # Multiple objects in a contours
    for c in contours:
        area = cv2.contourArea(c)
        # For any object that has a contour area > 40 pixels
        if area > 40:
            # Mark it
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 2)
    return img
            
def get_diff_filled(image, diff_image):
    contours = get_contours(diff_image)
    # Multiple objects in a contours
    for c in contours:
        area = cv2.contourArea(c)
        # For any object that has a contour area > 40 pixels
        if area > 40:
            # Mark it
            x,y,w,h = cv2.boundingRect(c)
            cv2.drawContours(image, [c], 0, (0,255,0), -1)
    return image
        
# Main
first_img = get_image("./images/1.jpg")
second_img = get_image("./images/2.jpg")
diff_img = get_structural_simlarity(first_img, second_img)
      
first_rect = get_diff_rect(first_img, diff_img)
second_rect = get_diff_rect(second_img, diff_img)
mask = get_diff_mask(first_img, diff_img)
filled_img = get_diff_filled(second_img ,diff_img)

# Output
show_image('before', first_rect)
show_image('after', second_rect)
show_image('diff', diff_img)
show_image('mask', mask)
show_image('filled after', filled_img)


