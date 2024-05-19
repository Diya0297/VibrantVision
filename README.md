# Color Detection and Tracking

This project uses OpenCV to detect and track colors in real-time through a webcam. It identifies the largest object of predefined colors and displays the name of the color on the screen.

## Requirements

- Python 3.x
- OpenCV
- NumPy

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/your-repo.git
    cd your-repo
    ```

2. Install the required packages:
    ```sh
    pip install opencv-python numpy
    ```

## Usage

Run the following command to start the color detection:
```sh
python colorDetection.py

##Code Explanation
Color Boundaries
The boundaries list contains the HSV color ranges for various colors that the program can detect. Each color range is associated with a color name.

boundaries = [
    ([0, 0, 33], [255, 255, 47], "Black"),
    ([0, 0, 47], [255, 255, 60], "Brown"),
    ([0, 0, 0], [255, 50, 50], "Undefined"),
    ([0, 50, 60], [10, 255, 255], "Red"),
    ([10, 50, 60], [25, 255, 255], "Orange"),
    ([25, 50, 60], [35, 255, 255], "Yellow"),
    ([35, 50, 60], [85, 255, 255], "Green"),
    ([85, 50, 60], [125, 255, 255], "Cyan"),
    ([125, 50, 60], [170, 255, 255], "Blue"),
    ([170, 50, 60], [180, 255, 255], "Violet")
]

##Color Name Function
The get_color_name function returns the name of the color based on the HSV values

def get_color_name(h, s, v):
    if 33 < v < 47:
        return "Black"
    if 47 < v <= 60:
        return "Brown"
    if s < 50 or v < 50:
        return "Undefined"
    if (0 <= h <= 10) or (170 <= h <= 180):
        return "Red"
    elif 10 < h <= 25:
        return "Orange"
    elif 25 < h <= 35:
        return "Yellow"
    elif 35 < h <= 85:
        return "Green"
    elif 85 < h <= 125:
        return "Cyan"
    elif 125 < h <= 170:
        return "Blue"
    elif 170 < h <= 180:
        return "Violet"
    else:
        return "Undefined"

## Main Program
The main program captures video from the webcam, processes each frame to detect colors, and highlights the largest detected object of a specific color with a bounding box and label.

	1. Capture video from the webcam:
	cap = cv2.VideoCapture(0)
	
	2. Read and process each frame:
	while True:
	    ret, frame = cap.read()
	    if not ret:
	        print("Error: Could not read frame.")
	        break
	
	    blur = cv2.GaussianBlur(frame, (15, 15), 0)
	    hsv_frame = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
	
	    largest_contour = None
	    largest_area = 0
	    largest_color_name = ""
	
	    for (lower, upper, name) in boundaries:
	        lower = np.array(lower, dtype="uint8")
	        upper = np.array(upper, dtype="uint8")
	        mask = cv2.inRange(hsv_frame, lower, upper)
	        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	        for c in cnts:
	            area = cv2.contourArea(c)
	            if area > largest_area:
	                largest_area = area
	                largest_contour = c
	                largest_color_name = name
	
	    if largest_contour is not None:
	        x, y, w, h = cv2.boundingRect(largest_contour)
	        cv2.drawContours(frame, [largest_contour], 0, (0, 0, 0), 2)
	        cv2.putText(frame, largest_color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
	        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
	
	    cv2.imshow("frame", frame)
	
	    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break
	
	cap.release()
	cv2.destroyAllWindows()

