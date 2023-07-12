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
    print("[Console] Converting image from byte to np.arr")
    arr_image = np.array(Image.open(BytesIO(byte_image)))
    return arr_image


def convert_arr_to_byte(arr_image):
    print("[Console] Converting image from np.arr to byte")
    arr_image_cvt = cv2.cvtColor(arr_image, cv2.COLOR_RGB2BGR)
    success, byte_image = cv2.imencode(".jpg", arr_image_cvt)
    if success:
        return byte_image.tobytes()
    else:
        raise Exception("[Console] Cannot convert array image to byte image")


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
    return dim, cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def convert_to_gray(image):
    """Convert from BGR to GRAY"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def convert_to_cv2_format(image):
    """Convert any dtype back to cv2 readable image array"""
    image = (image * 255).astype("uint8")
    return image


def get_blur(image, d=30, sigColor=80, sigSpace=80):
    """Bilateral Filter Blur
    -----
    Notes:
    Blur unimportant details and leave the edges
    -----
    Param: Adjust if needed
        image:      Input image
        d:          Diameter of each pixel neighborhood.
        sigColor:   Value of sigma in the color space. The greater the value, the colors farther to each other will start to get mixed.
        sigSpace:   Value of sigma in the coordinate space. The greater its value, the more further pixels will mix together, given that their colors lie within the sigmaColor range.
    """
    return cv2.bilateralFilter(image, d, sigColor, sigSpace)


def get_equalize_adapt(image, c_limit=0.1):
    """Contrast Limited Adaptive Histogram Equalization (CLAHE)
    -----
    Notes:
    An algorithm for local contrast enhancement, that uses histograms computed over different tile regions of the image. Local details can therefore be enhanced even in regions that are darker or lighter than most of the image.
    -----
    Param:
        image:      input image
        c_limit:    Clipping limit, normalized between 0 and 1 (higher values give more contrast).
    -----
    Return
        equalized:  an image that has it contrast adjusted
    """
    equalized = equalize_adapthist(
        image, kernel_size=None, clip_limit=c_limit, nbins=256
    )
    return convert_to_cv2_format(equalized)


def get_threshold(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


def get_edge(gray_img):
    """Sobel Edge Detection
    -----
    Notes:
    Edge detection involves mathematical methods to find points in an image where the brightness of pixel intensities changes distinctly
    -----
    Param:
        gray_image: input an image in gray scale
    -----
    Return:
        img_sobel:  an image in np.array that has been edge detected through sobel
    """
    img_sobelx = cv2.Sobel(gray_img, -1, 1, 0, ksize=1)
    img_sobely = cv2.Sobel(gray_img, -1, 0, 1, ksize=1)
    img_sobel = cv2.addWeighted(img_sobelx, 0.5, img_sobely, 0.5, 0)
    return img_sobel


def get_contours(image):
    print("[Console] Finding contours")
    threshold_img = get_threshold(image)
    contours = cv2.findContours(
        threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = contours[0] if len(contours) == 2 else contours[1]
    return contours


def get_diff_mask(image, diff_image, minDiffArea):
    mask = np.zeros(image.shape, dtype="uint8")
    contours = get_contours(diff_image)
    # Multiple objects in a contours
    for c in contours:
        area = cv2.contourArea(c)
        # For any object that has a contour area > minDiffArea (40 pixel for smaller details)
        if area > minDiffArea:
            # Mark it
            cv2.drawContours(mask, [c], 0, (255, 255, 255), -1)
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
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (36, 255, 12), 2)
    return img


def get_diff_filled(image, diff_image, minDiffArea):
    contours = get_contours(diff_image)
    # Multiple objects in a contours
    for c in contours:
        area = cv2.contourArea(c)
        # For any object that has a contour area > minDiffArea (40 pixel for smaller details)
        if area > minDiffArea:
            # Mark it
            cv2.drawContours(image, [c], 0, (0, 255, 0), -1)
    return image


def get_structural_simlarity(first_image, second_image):
    print("[Console] Calculating differences")
    # Compare
    (score, diff_img) = structural_similarity(first_image, second_image, full=True)
    # Convert return format to cv2 readable
    diff_img = convert_to_cv2_format(diff_img)
    print("[Console] Similarity score of {:.4f}%".format(score * 100))
    return score, diff_img


def preprocess_image(image, gray=True, contrast=False, blur=False, edge=False):
    """Pre-process Image
    -----
    Notes:
    Provide methods to prepare image for further examination
    -----
    Param:
        gray:       Get the image in gray scale
        contrast:   Adjust image's contrast
        blur:       Blur image
        edge:       Showing edges instead of full details
    -----
    Return:
        image:      Processed image
    """
    if gray:
        image = convert_to_gray(image)
    # show_image("Gray", gray)
    if contrast:
        image = get_equalize_adapt(
            image
        )  # Optional. Adjust contrast level through skimage.exposure.equalize_adapthist
    # show_image("Adjust Contrast", gray)
    if blur:
        image = get_blur(
            image
        )  # Optional. Bilateral Filter Blur for edge detect purpose
    # show_image("Blur", gray)
    if edge:
        image = get_edge(
            image
        )  # Optional. Detect different object through shape mainly, less dependent on color and noise
    # show_image("Edge", gray)
    return image


# API
@app.get("/")
def welcome_page():
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
    # Add images
    images = []
    for in_image in in_images:
        byte_image = await in_image.read()
        arr_image = convert_byte_to_arr(byte_image)
        images.append(arr_image)
        if len(images) == 2:
            break
    # Main
    # Get Image
    first_img = images[0]
    second_img = images[1]

    # Resize image if there is a difference in size
    # Modify this if needed
    if first_img.shape != second_img.shape:
        first_img = resize_image(first_img, 1280, 720)[1]
        second_img = resize_image(second_img, 1280, 720)[1]

    # Preprocess the image before comparing
    # Main step for the accuracy of the program
    # Only use the methods that are needed for the processing images, otherwise comment out
    first_pre = preprocess_image(
        first_img, gray=True, contrast=True, blur=True, edge=True
    )
    second_pre = preprocess_image(
        second_img, gray=True, contrast=True, blur=True, edge=True
    )
    # Compare and get the result
    score, diff_img = get_structural_simlarity(first_pre, second_pre)
    # Marking the differences
    filled_img = get_diff_filled(second_img, diff_img, 750)

    # Show response
    byte_image = convert_arr_to_byte(filled_img)

    # Create the text response
    response_text = "Similarity score of {:.4f}%".format(score * 100)

    # Create a response object with the image and text
    response = Response(content=byte_image, media_type="image/jpg")
    response.headers["Result"] = response_text

    return response
    # return Response(byte_image, media_type="image/jpg")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
