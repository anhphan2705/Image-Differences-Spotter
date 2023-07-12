# Image Differences Spotter

This code demonstrates image processing and comparison using the scikit-image and OpenCV libraries. It allows you to preprocess images, calculate structural similarity, and visualize the differences between two images.

## Prerequisites

- Python 3.x
- Required libraries: scikit-image, OpenCV

## Usage

1. Install the required libraries by running the following command:

   ```
   pip install scikit-image opencv-python
   ```

2. Place the images you want to compare in the `images` directory. The example assumes the images are named `1.jpg` and `2.jpg`.

3. Adjust the code as needed. You can modify the image directory, resize dimensions, and preprocessing options to fit your requirements.

4. Run the script and observe the results. It will display the original images, the differences between them, and the marked regions where differences are detected.

   ```
   python ./image_spot_differences_source.py
   ```

5. Optionally, you can uncomment the `write_image` lines to save the output images to the `output` directory.

## Code Structure

The code is structured as follows:

- `get_image(directory)`: Loads an image from the specified directory.
- `show_image(header, image)`: Displays an image with a header.
- `write_image(directory, image)`: Saves an image to the specified directory.
- `resize_image(image, height, width)`: Resizes an image to the specified dimensions.
- `convert_to_gray(image)`: Converts an image from BGR to grayscale.
- `convert_to_cv2_format(image)`: Converts an image to the CV2 readable format.
- `get_blur(image, d, sigColor, sigSpace)`: Applies a bilateral filter blur to an image.
- `get_equalize_adapt(image, c_limit)`: Applies contrast-limited adaptive histogram equalization to an image.
- `get_threshold(image)`: Applies a binary threshold to an image.
- `get_edge(gray_img)`: Performs Sobel edge detection on a grayscale image.
- `get_contours(image)`: Finds contours in an image.
- `get_diff_mask(image, diff_image, minDiffArea)`: Generates a mask highlighting the differences between two images.
- `get_diff_rect(image, diff_image, minDiffArea)`: Draws rectangles around the differences in an image.
- `get_diff_filled(image, diff_image, minDiffArea)`: Fills the differences in an image with green color.
- `get_structural_similarity(first_image, second_image)`: Calculates the structural similarity between two images.
- `preprocess_image(image, gray, contrast, blur, edge)`: Preprocesses an image for further examination.

## Contributing

Contributions are welcome! If you have any suggestions or improvements for this code, please feel free to submit a pull request.

## License

This code is licensed under the [MIT License](LICENSE).

## Acknowledgments

- The scikit-image library: https://scikit-image.org/
- OpenCV: https://opencv.org/

Feel free to customize this README file to fit your specific use case and provide any additional information that may be helpful for users of your code.