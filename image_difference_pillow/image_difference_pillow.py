from PIL import Image, ImageChops

img1 = Image.open("./images/1.jpg")
img2 = Image.open("./images/2.jpg")

""" How ImageChops.difference works
    - ImageChops.difference computes the 'absolute value of the pixel-by-pixel difference between the two images'
    - Results in a difference image that is returned.
"""

# Main
diff = ImageChops.difference(img1, img2)
# Output
img1.show()
img2.show()
diff.show()