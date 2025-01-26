import os
import cv2
from ultralytics import YOLO

from sort.sort import *
import easyocr
import matplotlib.pyplot as plt

import csv

##Image detection pipeline
def process_folder(input_dir, model_name, threshold, preferences):
    # Create OCR reader
    num_reader = easyocr.Reader(['en'], gpu=True)
    model = load_model(model_name)
    boatDict = load_csv_to_dict('database\sailing_results.csv')
    imageFiles, image_extensions = fetch_img_files(input_dir)

    for idx, file_name in enumerate(imageFiles):
        inputPath = os.path.join(input_dir, file_name)
        if inputPath.lower().endswith(image_extensions):
            handle_image(inputPath, model, threshold, None, num_reader, preferences, boatDict)
        else:
            handle_video(inputPath, model, threshold, num_reader, preferences, boatDict)


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


def handle_image(inputPath, model, threshold, boat_tracker, num_reader, preferences, boatDict):
    image = cv2.imread(inputPath)
    modified_image = process_image(model, image, threshold, boat_tracker, num_reader, preferences, boatDict)


def process_image(model, image, threshold, boat_tracker, reader, preferences, boatDict):
    showImg, showCrop = preferences
    results = model(image, verbose=False)[0]
    boat_track_ids = None

    detections = handle_boat_coords(image, results, threshold)

    if boat_tracker:
        boat_track_ids = boat_tracker.update(np.asarray(detections))

    text = handle_boat_num_coords(image, results, threshold, boat_track_ids, reader, showCrop, boatDict)

    if showImg:
        show_image(image)

    return image


def handle_boat_coords(image, results, threshold):
    detections = []
    colourDict = {0:(0, 255, 0),
                  2:(255, 255, 255),
                  3:(0, 0, 204),
                  4:(255, 0, 255),
                  5:(255, 255, 51),
                  6:(255, 128, 0),
                  7:(96, 96, 96),
                  8:(0, 153, 76),
                  9:(178, 253, 51),
                  10:(0, 0, 0),
                  11:(255, 102, 102),
                  12:(0, 204, 204),}
                  
    for result in results.boxes.data.tolist():
        
        x1, y1, x2, y2, score, class_id = result
        if score < threshold or class_id == 1:
            continue
        
        image = draw_on_image(image, result, results, colour=colourDict[class_id])
        detections.append([x1, y1, x2, y2, score])

    return detections


def handle_boat_num_coords(image, results, threshold, boat_track_ids, reader, showCrop, boatDict):
    boatNums = []
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score < threshold or class_id != 1:
            continue

        if boat_track_ids is not None:
            xcar1, ycar1, scar2, ycar2, boat_id = get_boat(result, boat_track_ids)

        ##Cropping, reading and drawing on image
        boatNumberCropThreshold = crop_image(image, result)
        boatNumberText, boatNumberTextScore = read_boat_number(boatNumberCropThreshold, reader)

        font_scale, thickness = customize_size(image)
        coords = centre_text(boatNumberText, font_scale, thickness, result)

        cv2.putText(image, boatNumberText, coords, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

        image = draw_on_image(image, result, results, colour=(255, 0, 0))

        resultImage = resize_image(boatNumberCropThreshold, scale_factor=5)
        if showCrop:
            show_image(boatNumberCropThreshold)

        boatNumberText = check_boat_number(boatNumberText, boatDict, results.names[int(class_id)].upper())
        if boatNumberText:
            cv2.putText(image, boatDict[boatNumberText]['Helm Name'], coords, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (207, 159, 255), thickness, cv2.LINE_AA)

        boatNums.append(boatNumberText)

    return boatNums


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
    _, boatNumberCropThreshold = cv2.threshold(boatNumberCropGray, 100, 255, cv2.THRESH_BINARY_INV)

    return boatNumberCropThreshold


def draw_on_image(image, result, results, colour):
    x1, y1, x2, y2, score, class_id = result
    text = ''

    font_scale, thickness = customize_size(image)

    if class_id !=0 and class_id !=1:
        text = results.names[int(class_id)].upper()
        x1,y1, x2,y2 = x1-10,y1-10, x2+10,y2+10

    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), colour, 3)
    cv2.putText(image, text, (int(x1), int(y1-5)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, colour, thickness, cv2.LINE_AA)
    
    return image


def read_boat_number(boatNumber, reader):
    detections = reader.readtext(boatNumber)
    best_guess = ''
    best_score = 0

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')
        if text != None and score>best_score:
            best_score = score
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


def handle_video(inputPath, model, threshold, reader, preferences, boatDict):

    boat_tracker = Sort()
    video_path_out = inputPath.replace('/testing\\', '/results/')
    #video_path_out = './data/results/resultVideo.mp4'
    cap = cv2.VideoCapture(inputPath)

    ret, frame = cap.read()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(video_path_out, 
                      cv2.VideoWriter_fourcc(*'mp4v'), 
                      int(cap.get(cv2.CAP_PROP_FPS)), 
                      (width, height))
    
    while ret:
        modified_image = process_image(model, frame, threshold, boat_tracker, reader, preferences, boatDict)
        out.write(modified_image)
        
        ret, frame = cap.read()

    cap.release()
    out.release()


def customize_size(image):
    height, width, _ = image.shape
    # Calculate font scale based on image size
    base_font_scale = 0.001 
    font_scale = base_font_scale * height

    # Set text thickness proportional to the image size
    base_thickness = 2
    thickness = max(1, int(base_thickness * (height / 500)))
    return font_scale, thickness


def centre_text(boatNumberText, font_scale, thickness, result):
    x1, y1, x2, y2, score, class_id = result
    text_size = cv2.getTextSize(boatNumberText, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_width, text_height = text_size
    box_center_x = int((x1 + x2) / 2)
    
    text_x = int(box_center_x - text_width / 2)
    text_y = int(max(y1 - 5, text_height))

    return (text_x, text_y)


def load_csv_to_dict(csv_path):
    boat_data = {}

    with open(csv_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            boat_number = row.get('Boat Number')
            if boat_number:
                boat_data[boat_number] = {'Helm Name': row.get('Helm Name', 'N/A'),
                                            'Crew Name': row.get('Crew Name', 'N/A'),
                                            'Boat Type': row.get('Class', 'N/A')}
                
    return boat_data

def check_boat_number(boatNumberText, boatDict, boat_type):
    if len(boatNumberText)>2:
        return boatNumberText
    else:
        if boatNumberText in boatDict and boatNumberText.isdigit() and boatDict[boatNumberText]['Boat Type'] == boat_type:
            return boatNumberText
        return None
        
    


  