import easyocr
import cv2
import matplotlib.pyplot as plt
import os

reader = easyocr.Reader(['en'], gpu=True)
directory_path = r'toolScripts\boat_numbers'

for root, dirs, files in os.walk(directory_path):
    for file in files:

        file_path = os.path.join(root, file)

        image = cv2.imread(file_path)

        detections = reader.readtext(image)

        # Process detections
        for detection in detections:
            bbox, text, score = detection  
            
            # Draw bounding box
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)  # Green box with thickness 2
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Put the detected text above the box
            cv2.putText(image, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Process the detected text
            text = text.upper().replace(' ', '')
            if text:
                print(text)

        # Display the image using matplotlib
        plt.figure(figsize=(10, 10))
        plt.imshow(image_rgb)
        plt.axis('off')  # Hide the axes
        plt.show()