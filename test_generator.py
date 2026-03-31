import cv2
import numpy as np

# Create a dark background
img = np.zeros((800, 800, 3), dtype=np.uint8)

# Define a white polygon (a skewed rectangle)
pts = np.array([[200, 150], [650, 200], [600, 700], [150, 600]], np.int32)
pts = pts.reshape((-1, 1, 2))

# Fill the polygon with white
cv2.fillPoly(img, [pts], (255, 255, 255))

# Add some 'text' (black lines) inside the polygon
cv2.line(img, (250, 300), (550, 320), (0, 0, 0), 5)
cv2.line(img, (240, 400), (540, 420), (0, 0, 0), 5)
cv2.line(img, (230, 500), (530, 520), (0, 0, 0), 5)

# Save the image
cv2.imwrite('sample_doc.jpg', img)
print("sample_doc.jpg created successfully.")
