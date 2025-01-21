import os
import cv2
from ultralytics import YOLO

from sort.sort import *
import easyocr
import matplotlib.pyplot as plt

##Image detection pipeline
def process_folder(input_dir, model_name, threshold):
    # Create OCR reader
    num_reader = easyocr.Reader(['en'], gpu=True)
    

    model = load_model(model_name)
    imageFiles, image_extensions = fetch_img_files(input_dir)

    for idx, file_name in enumerate(imageFiles):
        inputPath = os.path.join(input_dir, file_name)
        if inputPath.lower().endswith(image_extensions):
            handle_image(inputPath, model, threshold, None, num_reader)
        else:
            handle_video(inputPath, model, threshold, num_reader)


def load_model(model_name):
    model_path = os.path.join('.', 'runs', 'detect', model_name, 'weights', 'last.pt')
    model = YOLO(model_path) 
    return model


def fetch_img_files(input_dir):
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv')

    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    img_files = [f for f in files if f.lower().endswith(image_extensions+video_extensions)]
    return img_files, image_extensions


def handle_image(inputPath, model, threshold, boat_tracker, num_reader):
    image = cv2.imread(inputPath)
    modified_image = process_image(model, image, threshold, boat_tracker, num_reader)
    show_image(modified_image)


def process_image(model, image, threshold, boat_tracker, reader):
    results = model(image)[0]
    detections = []
    ## All boats
    for result in results.boxes.data.tolist():
        
        x1, y1, x2, y2, score, class_id = result
        if score > threshold and class_id == 0:

            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(image, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            detections.append([x1, y1, x2, y2, score])

    if boat_tracker:
        boat_track_ids = boat_tracker.update(np.asarray(detections))

    ##All boat numbers
    for result in results.boxes.data.tolist():

        if score < threshold or class_id != 1:
            continue


        if boat_id == -1:
            continue
        
        xcar1, ycar1, scar2, ycar2, boat_id = get_boat(result, boat_track_ids)

        boatNumberCropThreshold = crop_image(image, result)


        resultImage = resize_image(boatNumberCropThreshold, scale_factor=5)
        #show_image(resultImage)

        boatNumberText, boatNumberTextScore = read_boat_number(boatNumberCropThreshold, reader)
        print(boatNumberText)

        image = draw_on_image(image, results, result)


    image = cv2.resize(image, (600, 600))

    return image


def get_boat(boatNumber, boat_track_ids):
    x1, y1, x2, y2, score, class_id= boatNumber


    foundIt = False
    for j in range(len(boat_track_ids)):
        xcar1, ycar1, xcar2, ycar2, carID = boat_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_index = j
            foundIt = True
            break

    if foundIt:
        return boat_track_ids[car_index]
    
    return -1, -1, -1, -1, -1


def show_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis('off')  # Hide the axes
    plt.show(block=True)


def resize_image(image, scale_factor):
    image = cv2.resize(image, (image.shape[:2][1]*scale_factor, image.shape[:2][0]*scale_factor), interpolation=cv2.INTER_LINEAR)
    return image


def crop_image(image, result):
    x1, y1, x2, y2, score, class_id = result
    boatNumberCrop = image[int(y1):int(y2), int(x1):int(x2), :]
    boatNumberCropGray = cv2.cvtColor(boatNumberCrop, cv2.COLOR_BGR2GRAY)
    _, boatNumberCropThreshold = cv2.threshold(boatNumberCropGray, 150, 255, cv2.THRESH_BINARY_INV)

    return boatNumberCropThreshold


def draw_on_image(image, results, result):
    x1, y1, x2, y2, score, class_id = result

    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
    cv2.putText(image, results.names[int(class_id)].upper(), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1, cv2.LINE_AA)
    
    return image


def read_boat_number(boatNumber, reader):
    #show_image(boatNumber)
    detections = reader.readtext(boatNumber)
    best_guess = ''

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')
        if text != None:
            best_guess = text
    return best_guess, 0 


def clean_text(text):
    dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}
    
    resultText = ''
    for letter in text:
        if letter in dict_char_to_int:
            resultText += dict_char_to_int[letter]
        else:
            resultText += letter

    return resultText


def handle_video(inputPath, model, threshold, num_reader):
    boat_tracker = Sort()
    video_path_out = './data/results/resultVideo.mp4'
    cap = cv2.VideoCapture(inputPath)
    ret, frame = cap.read()

    out = cv2.VideoWriter(video_path_out, 
                      cv2.VideoWriter_fourcc(*'mp4v'), 
                      int(cap.get(cv2.CAP_PROP_FPS)), 
                      (600, 600))
    
    while ret:
        modified_image = process_image(model, frame, threshold, boat_tracker, num_reader)
        out.write(modified_image)
        
        ret, frame = cap.read()

    cap.release()
    out.release()