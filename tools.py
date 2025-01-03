import os
import cv2
from ultralytics import YOLO


def load_model():
    model_path = os.path.join('.', 'runs', 'detect', 'train3', 'weights', 'last.pt')
    model = YOLO(model_path) 
    return model


def process_image(model, image, threshold):
    results = model(image)[0]
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(image, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    image = cv2.resize(image, (600, 600))
    cv2.imshow("Output", image)

    if cv2.waitKey(0) & 0xFF == ord('q'):  # Press 'q' to quit early
        return

###Image processing
def process_images(input_dir):
    model = load_model()
    image_files = fetch_img_files(input_dir)
    threshold = 0.5

    for idx, file_name in enumerate(image_files):
        input_path = os.path.join(input_dir, file_name)

        # Read the image
        image = cv2.imread(input_path)
        process_image(model, image, threshold)

    cv2.destroyAllWindows()


def fetch_img_files(input_dir):
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))]
    return image_files

