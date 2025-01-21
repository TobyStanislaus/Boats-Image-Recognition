from tools import *

test_images_path = "./data/testing"
model_name = 'train18'
threshold = 0.3


showImg = False
showCrop = True

preferences = [showImg, showCrop]

process_folder(test_images_path, model_name, threshold, preferences)

