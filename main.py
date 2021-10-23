'''Program to convert image file to cartoonish form.'''

# Importing the libraries
import numpy as np
import cv2


# Defining the function to read the image file
def read_file(filename):
    img = cv2.imread(filename)
    cv2.imshow("Image", img)
    return img


# Enter your image file name or path
filename = input("Enter the file name (or path): ")
img = read_file(filename)


# Edge mask function to transform the image into grayscale and reduce the noise of the blurred grayscale image
def edge_mask(img, line_size, blur_value):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 9)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
    return edges


# A larger line size means the thicker edges that will be emphasized in the image.
line_size = int(input("Enter the value you want for line size: "))

# The larger blur value means fewer black noises appear in the image
blur_value = int(input("Enter the value you want for blur size: "))

edges = edge_mask(img, line_size, blur_value)
cv2.imshow("edges", edges)


# Defining the function to reduce the number of colors in the photo
def color_quantization(img, k):
    # Transform the image
    data = np.float32(img).reshape((-1, 3))

    # Determine criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

    # Implementing K-Means
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result


# k value is to determine the number of colors we want to apply to the image
total_color = int(input("Enter the total color value (k value): "))
img = color_quantization(img, total_color)


# To reduce the noise in the image we use a bilateral filter. It would give a bit blurred and sharpness-reducing effect to the image.

# d — Diameter of each pixel neighborhood
d = int(input("Enter the value of d: "))

# sigmaColor — A larger value of the parameter means larger areas of semi-equal color.
sigmaColor = int(input("Enter the value of sigmaColor: "))

# sigmaSpace –A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough.
sigmaSpace = int(input("Enter the value of sigmaSpace: "))

blurred = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)

cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)
cv2.imshow("Cartoon", cartoon)
cv2.waitKey(0)
cv2.destroyAllWindows()