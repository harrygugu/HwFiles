import cv2
import numpy as np

# Load the RGB image
image = cv2.imread('data/for_watson.png')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Split the image into its three color regions
height = gray_image.shape[0]
section_height = height // 3

# Extract each section (top, middle, bottom)
top_section = gray_image[:section_height, :]
middle_section = gray_image[section_height:2*section_height, :]
bottom_section = gray_image[2*section_height:, :]

# Apply histogram equalization to each section to enhance contrast
top_enhanced = cv2.equalizeHist(top_section)
middle_enhanced = cv2.equalizeHist(middle_section)
bottom_enhanced = cv2.equalizeHist(bottom_section)

# Combine the enhanced sections back into a single image
enhanced_image = cv2.vconcat([top_enhanced, middle_enhanced, bottom_enhanced])

# Save and display the grayscale image
cv2.imwrite('result/grayscale_message.png', enhanced_image)

