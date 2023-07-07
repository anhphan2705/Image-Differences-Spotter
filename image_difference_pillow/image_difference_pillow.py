from PIL import Image, ImageChops

img1 = Image.open("./images/1.jpg")
img2 = Image.open("./images/2.jpg")

# Main
diff = ImageChops.difference(img1, img2)
# Output
img1.show()
img2.show()
diff.show()