from skimage.metrics import structural_similarity
from skimage.exposure import equalize_adapthist
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

def resize_image(image, height, width):
    print("[Console] Resizing image to 720p")
    dim = (height, width)
    return dim, cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def get_canny_edge(image):
    return cv2.Canny(image, 100, 200)
    
def preprocess_image(image):
    gray = convert_to_gray(image)
    # equa = get_equalize_adapt(gray)
    blur = get_blur(gray)
    show_image("bn", blur)
    canny = get_canny_edge(blur)
    # show_image("bn", canny)
    return canny

def convert_to_cv2_format(image):
    # Convert any dtype back to cv2 readable
    image = (image * 255).astype("uint8")
    return image

def get_blur(image):
    # d: Diameter of each pixel neighborhood.
    # sigmaColor: Value of sigma in the color space. The greater the value, the colors farther to each other will start to get mixed.
    # sigmaSpace: Value of sigma in the coordinate space. The greater its value, the more further pixels will mix together, given that their colors lie within the sigmaColor range.
    # return cv2.bilateralFilter(image, 25, 85, 75)
    return image - cv2.GaussianBlur(image, (21, 21), 3)+127

def get_equalize_adapt(image):
    equalized = equalize_adapthist(image, kernel_size=None, clip_limit=0.01, nbins=256)
    return convert_to_cv2_format(equalized)

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
    # Compare
    (score, diff_img) = structural_similarity(first_image, second_image, full=True)
    # Convert return format to cv2 readable
    diff_img = convert_to_cv2_format(diff_img)
    print("[Console] Similarity score of {:.4f}%".format(score * 100))
    return diff_img

def get_diff_mask(image, diff_image, minDiffArea):
    mask = np.zeros(image.shape, dtype='uint8')
    contours = get_contours(diff_image)
    # Multiple objects in a contours
    for c in contours:
        area = cv2.contourArea(c)
        # For any object that has a contour area > minDiffArea (40 pixel for smaller details)
        if area > minDiffArea:
            # Mark it
            cv2.drawContours(mask, [c], 0, (255,255,255), -1)
    return mask

def get_diff_rect(image, diff_image, minDiffArea):
    print("[Console] Drawing rectangle around the differences")
    img = image.copy()
    contours = get_contours(diff_image)
    # Multiple objects in a contours
    for c in contours:
        area = cv2.contourArea(c)
        # For any object that has a contour area > minDiffArea (40 pixel for smaller details)
        if area > minDiffArea:
            # Mark it
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 2)
    return img
            
def get_diff_filled(image, diff_image, minDiffArea):
    contours = get_contours(diff_image)
    # Multiple objects in a contours
    for c in contours:
        area = cv2.contourArea(c)
        # For any object that has a contour area > minDiffArea (40 pixel for smaller details)
        if area > minDiffArea:
            # Mark it
            cv2.drawContours(image, [c], 0, (0,255,0), -1)
    return image
        
# Main
first_img = get_image("./images/real/1.jpg")
second_img = get_image("./images/real/2.jpg")
first_img = resize_image(first_img, 1280, 720)[1]
second_img = resize_image(second_img, 1280, 720)[1]

first_pre = preprocess_image(first_img)
second_pre = preprocess_image(second_img)

diff_img = get_structural_simlarity(first_pre, second_pre)
      
first_rect = get_diff_rect(first_img, diff_img, 750)
second_rect = get_diff_rect(second_img, diff_img, 750)
# mask = get_diff_mask(first_img, diff_img, 500)
filled_img = get_diff_filled(second_img ,diff_img, 750)

# Output
show_image('before', first_rect)
# write_image("./output/real/1_2/first_rect.jpg", first_rect)
show_image('after', second_rect)
# write_image("./output/real/1_2/second_rect.jpg", second_rect)
# show_image('diff', diff_img)
# write_image("./output/test/diff_img.jpg", diff_img)
# show_image('mask', mask)
# write_image("./output/real/2_3/mask.jpg", mask)
show_image('filled', filled_img)
# write_image("./output/real/1_2/filled.jpg", filled_img)