from tools import *
import time
test_images_path = "./data/testing"
model_name = 'train23'
threshold = 0.1


showImg = False
showCrop = False
useWebCam = False

preferences = [showImg, showCrop]

start_time = time.time()
process_folder(test_images_path, model_name, threshold, preferences, useWebCam)
end_time = time.time()
print('Time Taken:',end_time-start_time)
