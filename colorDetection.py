
# import cv2

# # Define HSV ranges for rainbow colors
# def get_color_name(h, s, v):
#     if 33 < v < 47:
#         return "Black"
#     if 47 < v <= 60:
#         return "Brown"
#     if s < 50 or v < 50:
#         return "Undefined"
#     if (0 <= h <= 10) or (170 <= h <= 180):
#         return "Red"
#     elif 10 < h <= 25:
#         return "Orange"
#     elif 25 < h <= 35:
#         return "Yellow"
#     elif 35 < h <= 85:
#         return "Green"
#     elif 85 < h <= 125:
#         return "Cyan"
#     elif 125 < h <= 170:
#         return "Blue"
#     elif 170 < h <= 180:
#         return "Violet"
#     else:
#         return "Undefined"

# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Error: Could not open video capture.")
#     exit()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Could not read frame.")
#         break

#     # Convert the frame to HSV color space
#     hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     height, width, _ = frame.shape
#     cx = int(width / 2)
#     cy = int(height / 2)
#     center = hsv_frame[cy, cx]
#     h_value = center[0]
#     s_value = center[1]
#     v_value = center[2]

#     color = get_color_name(h_value, s_value, v_value)

#     print(f"HSV value at center: {center}, Color: {color}")
#     cv2.putText(frame, color, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
#     cv2.circle(frame, (cx, cy), 5, (255, 0, 0), 3)

#     cv2.imshow("Frame", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



# import cv2
# import numpy as np
# import math
# import matplotlib.pyplot as plt 

# boundaries = [
# 	([0, 0, 33], [255, 255, 47], "Black"),
# 	([0, 0, 47], [255, 255, 60],"Brown"),
# 	([0, 0, 0], [255, 50, 50],"Undefined"),
# 	([0, 50, 60], [10, 255, 255],"Red"),
# 	([10, 50, 60], [25, 255, 255],"Orange"),
#     ([25, 50, 60], [35, 255, 255],"Yellow"),
#     ([35, 50, 60], [85, 255, 255],"Green"),
#     ([85, 50, 60], [125, 255, 255],"Cyan"),
#     ([125, 50, 60], [170, 255, 255],"Blue"),
#     ([170, 50, 60], [180, 255, 255],"Violet")
# ]
# def get_color_name(h, s, v):
#     if 33 < v < 47:
#         return "Black"
#     if 47 < v <= 60:
#         return "Brown"
#     if s < 50 or v < 50:
#         return "Undefined"
#     if (0 <= h <= 10) or (170 <= h <= 180):
#         return "Red"
#     elif 10 < h <= 25:
#         return "Orange"
#     elif 25 < h <= 35:
#         return "Yellow"
#     elif 35 < h <= 85:
#         return "Green"
#     elif 85 < h <= 125:
#         return "Cyan"
#     elif 125 < h <= 170:
#         return "Blue"
#     elif 170 < h <= 180:
#         return "Violet"
#     else:
#         return "Undefined"
     
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Error: Could not open video capture.")
#     exit()   
# #frame=camel
# while(True):
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Could not read frame.")
#         break
#     blur = cv2.GaussianBlur(frame,(1,1),0)
#         # Convert the frame to HSV color space
#     hsv_frame = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
#     height, width,color  = frame.shape
#     cx = int(width / 2)
#     cy = int(height / 2)
#     center = hsv_frame[cy, cx]
#     h_value = center[0]
#     s_value = center[1]
#     v_value = center[2]
#     contours=[]
#     blank=frame*0+255
#     for (lower, upper, name) in boundaries:
#         lower = np.array(lower, dtype = "uint8")
#         upper = np.array(upper, dtype = "uint8")
#         mask = cv2.inRange(hsv_frame, lower, upper)
#         output = cv2.bitwise_and(hsv_frame, hsv_frame, mask = mask)
#         cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#         for c in cnts:
#             j=cv2.contourArea(c)
#             x,y,w,h = cv2.boundingRect(c)
#             cv2.drawContours(frame,[c], 0, (0,0,0), 3)
#             cv2.putText(frame, name, (math.floor(x+w/2),math.floor(y+h/2)), cv2.FONT_HERSHEY_SIMPLEX, 40*j/width/height, (255, 0, 0), 2)
#             cv2.drawContours(blank,[c], 0, (0,0,0), 3)
#             cv2.putText(blank, name, (math.floor(x+w/2),math.floor(y+h/2)), cv2.FONT_HERSHEY_SIMPLEX, 40*j/width/height, (255, 0, 0), 2)
#         contours.append(cnts)
#         print(name)
#     cv2.imshow("frame",frame)
#     cv2.imshow("Frame",blank)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np
import math

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

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Apply Gaussian blur with a smaller kernel size
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
        # Set the font scale and thickness for better readability
        font_scale = 1.0
        thickness = 2
        cv2.putText(frame, largest_color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)
        # Draw a bounding box around the largest contour
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
