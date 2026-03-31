# Smart Document Scanner 📄🔍

This is a Computer Vision project developed as part of a Bring Your Own Project (BYOP) capstone. It solves the real-world problem of digitizing physical documents (like receipts, handwritten notes, or whiteboard sketches) using just a camera, eliminating the need for bulky, expensive scanning hardware.

## 🌟 Real-World Problem Solved
When taking pictures of documents using a smartphone, the resulting images are often skewed, poorly lit, and difficult to read. Physical scanners solve this, but they aren't always accessible. 

This project uses **Computer Vision algorithms** to automatically detect the edges of a document in an image, apply a mathematical perspective transformation to "flatten" it, and enhance the contrast so it looks like a clean, scanned PDF document.

## 🛠️ How it Works
The pipeline uses `OpenCV` and `Python` and consists of 3 main steps:
1. **Edge Detection:** The image is resized, converted to grayscale, blurred (to remove high-frequency noise), and passed through a Canny Edge Detector.
2. **Contour Extraction:** We find the contours in the edged image. We assume that the largest contour with exactly four corners is our document.
3. **Perspective Transform & Thresholding:** Using the coordinates of the four corners, we apply a top-down "birds-eye view" perspective warp. Finally, adaptive thresholding is applied to give it the classic black-and-white scanned effect.

## 🚀 Setup & Installation

### Prerequisites
You need Python 3 installed on your machine. All required packages are listed in `requirements.txt`.

### Installation Steps
1. Clone this repository (or download the files).
2. Open a terminal or command prompt in the project folder.
3. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

To scan a document, run the script from the command line and pass the path to your image using the `-i` or `--image` argument.

```bash
python scanner.py --image path/to/your/image.jpg
```

### What to expect:
The script is interactive and will show you the step-by-step process:
1. Press `ANY KEY` to advance from the Edge Detection view.
2. Press `ANY KEY` to advance from the Contour Outline view.
3. Press `ANY KEY` to close the final Scanned document view.

The final enhanced and flattened output will be automatically saved in the same directory as `scanned_output.jpg`.

## 📂 Project Structure
- `scanner.py` - The main Python script containing the computer vision pipeline.
- `requirements.txt` - Python dependencies needed to run the project.
- `README.md` - Documentation and setup instructions.

## 🎓 Learning Outcomes Applied
- Image filtering and edge detection (`Canny`, `GaussianBlur`)
- Contour detection and shape approximation (`findContours`, `approxPolyDP`)
- Geometric image transformations (`getPerspectiveTransform`, `warpPerspective`)
- Image thresholding (`adaptiveThreshold`)
