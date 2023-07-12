from fastapi import FastAPI, UploadFile, Response, Form
from fastapi.responses import HTMLResponse
from PIL import Image
from io import BytesIO
from skimage.metrics import structural_similarity
from skimage.exposure import equalize_adapthist
import cv2
import numpy as np

app = FastAPI()


def convert_byte_to_arr(byte_image):
    """
    Converts an image from byte array format to a NumPy array.

    Args:
        byte_image (bytes): Byte array representing the image.

    Returns:
        numpy.ndarray: Image in NumPy array format.
    """
    # Convert byte array to PIL Image
    image = Image.open(BytesIO(byte_image))
    # Convert PIL Image to NumPy array
    arr_image = np.array(image)
    return arr_image


def convert_arr_to_byte(arr_image):
    """
    Converts an image from NumPy array format to a byte array.

    Args:
        arr_image (numpy.ndarray): Image in NumPy array format.

    Returns:
        bytes: Byte array representing the image.

    Raises:
        Exception: If the conversion fails.
    """
    # Convert RGB image to BGR format
    arr_image_cvt = cv2.cvtColor(arr_image, cv2.COLOR_RGB2BGR)
    # Encode the image as JPEG format
    success, byte_image = cv2.imencode(".jpg", arr_image_cvt)
    if success:
        return byte_image.tobytes()
    else:
        raise Exception("Cannot convert array image to byte image")


def get_image(directory):
    """
    Reads an image file from the specified directory.

    Args:
        directory (str): Directory path of the image file.

    Returns:
        numpy.ndarray: Image in NumPy array format.
    """
    return cv2.imread(directory)


def show_image(header, image):
    """
    Displays an image using OpenCV's imshow function.

    Args:
        header (str): Window title/header for the image.
        image (numpy.ndarray): Image to be displayed.
    """
    cv2.imshow(header, image)
    cv2.waitKey()


def write_image(directory, image):
    """
    Saves an image to the specified directory.

    Args:
        directory (str): Directory path to save the image.
        image (numpy.ndarray): Image to be saved.
    """
    cv2.imwrite(directory, image)


def resize_image(image, height, width):
    """
    Resizes an image to the specified height and width.

    Args:
        image (numpy.ndarray): Image to be resized.
        height (int): Desired height of the image.
        width (int): Desired width of the image.

    Returns:
        tuple: A tuple containing the new dimensions and the resized image as a NumPy array.
    """
    dim = (height, width)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return dim, resized_image


def convert_to_gray(image):
    """
    Converts an image from BGR to grayscale.

    Args:
        image (numpy.ndarray): Image to be converted.

    Returns:
        numpy.ndarray: Grayscale image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def convert_to_cv2_format(image):
    """
    Converts any image dtype back to a CV2-readable image array.

    Args:
        image (numpy.ndarray): Image to be converted.

    Returns:
        numpy.ndarray: Converted image as a CV2-readable image array.
    """
    image = (image * 255).astype("uint8")
    return image


def get_blur(image, d=30, sigColor=80, sigSpace=80):
    """
    Applies a bilateral filter blur to an image.

    Args:
        image (numpy.ndarray): Image to be blurred.
        d (int): Diameter of each pixel neighborhood.
        sigColor (float): Value of sigma in the color space.
        sigSpace (float): Value of sigma in the coordinate space.

    Returns:
        numpy.ndarray: Blurred image.
    """
    return cv2.bilateralFilter(image, d, sigColor, sigSpace)


def get_equalize_adapt(image, c_limit=0.1):
    """
    Applies contrast limited adaptive histogram equalization (CLAHE) to an image.

    Args:
        image (numpy.ndarray): Input image.
        c_limit (float): Clipping limit, normalized between 0 and 1.

    Returns:
        numpy.ndarray: Image with adjusted contrast.
    """
    equalized = equalize_adapthist(
        image, kernel_size=None, clip_limit=c_limit, nbins=256
    )
    return convert_to_cv2_format(equalized)


def get_threshold(image):
    """
    Applies thresholding to an image.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Thresholded image.
    """
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


def get_edge(gray_img):
    """
    Detects edges in a grayscale image using Sobel edge detection.

    Args:
        gray_img (numpy.ndarray): Grayscale image.

    Returns:
        numpy.ndarray: Image with detected edges.
    """
    img_sobelx = cv2.Sobel(gray_img, -1, 1, 0, ksize=1)
    img_sobely = cv2.Sobel(gray_img, -1, 0, 1, ksize=1)
    img_sobel = cv2.addWeighted(img_sobelx, 0.5, img_sobely, 0.5, 0)
    return img_sobel


def get_contours(image):
    """
    Finds contours in a binary image.

    Args:
        image (numpy.ndarray): Binary image.

    Returns:
        list: List of contours found in the image.
    """
    threshold_img = get_threshold(image)
    contours = cv2.findContours(
        threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = contours[0] if len(contours) == 2 else contours[1]
    return contours


def get_diff_mask(image, diff_image, minDiffArea):
    """
    Generates a mask indicating the differences between two images.

    Args:
        image (numpy.ndarray): Original image.
        diff_image (numpy.ndarray): Image showing the differences.
        minDiffArea (int): Minimum area of a difference to be considered.

    Returns:
        numpy.ndarray: Mask indicating the differences between the images.
    """
    mask = np.zeros(image.shape, dtype="uint8")
    contours = get_contours(diff_image)
    for c in contours:
        area = cv2.contourArea(c)
        if area > minDiffArea:
            cv2.drawContours(mask, [c], 0, (255, 255, 255), -1)
    return mask


def get_diff_rect(image, diff_image, minDiffArea):
    """
    Draws rectangles around the differences between two images.

    Args:
        image (numpy.ndarray): Original image.
        diff_image (numpy.ndarray): Image showing the differences.
        minDiffArea (int): Minimum area of a difference to be considered.

    Returns:
        numpy.ndarray: Image with rectangles drawn around the differences.
    """
    img = image.copy()
    contours = get_contours(diff_image)
    for c in contours:
        area = cv2.contourArea(c)
        if area > minDiffArea:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (36, 255, 12), 2)
    return img


def get_diff_filled(image, diff_image, minDiffArea):
    """
    Fills the differences between two images with green color.

    Args:
        image (numpy.ndarray): Original image.
        diff_image (numpy.ndarray): Image showing the differences.
        minDiffArea (int): Minimum area of a difference to be considered.

    Returns:
        numpy.ndarray: Image with differences filled with green color.
    """
    contours = get_contours(diff_image)
    for c in contours:
        area = cv2.contourArea(c)
        if area > minDiffArea:
            cv2.drawContours(image, [c], 0, (0, 255, 0), -1)
    return image


def get_structural_simlarity(first_image, second_image):
    """
    Calculates the structural similarity between two images.

    Args:
        first_image (numpy.ndarray): First image.
        second_image (numpy.ndarray): Second image.

    Returns:
        tuple: A tuple containing the similarity score and the difference image.
    """
    (score, diff_img) = structural_similarity(first_image, second_image, full=True)
    diff_img = convert_to_cv2_format(diff_img)
    return score, diff_img


def preprocess_image(image, gray=True, contrast=False, blur=False, edge=False):
    """
    Preprocesses an image by applying various image processing techniques.

    Args:
        image (numpy.ndarray): Image to be preprocessed.
        gray (bool): Convert the image to grayscale.
        contrast (bool): Adjust the image's contrast.
        blur (bool): Blur the image.
        edge (bool): Show edges instead of full details.

    Returns:
        numpy.ndarray: Preprocessed image.
    """
    if gray:
        image = convert_to_gray(image)
    if contrast:
        image = get_equalize_adapt(image)
    if blur:
        image = get_blur(image)
    if edge:
        image = get_edge(image)
    return image


@app.get("/")
def welcome_page():
    """
    Serves the root route ("/") and displays a welcome message with a link to the API documentation.
    """
    return HTMLResponse(
        """
        <h1>Welcome to Banana</h1>
        <p>Click the button below to go to /docs/:</p>
        <form action="/docs" method="get">
            <button type="submit">Visit Website</button>
        </form>
    """
    )


@app.post("/find_differences")
async def find_differences(in_images: list[UploadFile]):
    """
    Compares two uploaded images and finds the differences between them.

    Args:
        in_images (list[UploadFile]): List of uploaded images.

    Returns:
        Response: HTTP response containing the difference image and a response header with the similarity score.
    """
    images = []
    for in_image in in_images:
        byte_image = await in_image.read()
        arr_image = convert_byte_to_arr(byte_image)
        images.append(arr_image)
        if len(images) == 2:
            break

    first_img = images[0]
    second_img = images[1]

    if first_img.shape != second_img.shape:
        first_img = resize_image(first_img, 1280, 720)[1]
        second_img = resize_image(second_img, 1280, 720)[1]

    first_pre = preprocess_image(
        first_img, gray=True, contrast=True, blur=True, edge=True
    )
    second_pre = preprocess_image(
        second_img, gray=True, contrast=True, blur=True, edge=True
    )

    score, diff_img = get_structural_simlarity(first_pre, second_pre)
    filled_img = get_diff_filled(second_img, diff_img, 750)

    byte_image = convert_arr_to_byte(filled_img)

    response_text = "Similarity score of {:.4f}%".format(score * 100)

    response = Response(content=byte_image, media_type="image/jpg")
    response.headers["Result"] = response_text

    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
