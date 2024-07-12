    # Data Collection based on the target values of the FER dataset.
import os
from bing_image_downloader import downloader
import logging
import cv2


    # Setting up the log file
logging.basicConfig(
        filename= 'logs\data_ingestion_logs.log',
        level= logging.INFO,
        format= '%(asctime)s - %(levelname)s - %(message)s'
    )

    # Harcascade classifier to ensure images captured are of humans ( not emojis)
face_classifier = cv2.CascadeClassifier('Harcascades\haarcascade_frontalface_default.xml')


def face_check(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray , scaleFactor = 1.1)
    return len(faces) > 0

    # Target Values
emotions = ["Angry person face", "Disgusted person face", "Fearful person face", "Happy person face", "Sad person face", "Surprised person face", "Neutral person face"]

limit = 50
train_test_split_ratio = 0.8

base_output_dir = 'target_dir_images'
train_dir = 'train'
test_dir = 'test'

    # base directory
if not os.path.exists(base_output_dir):
    os.makedirs(base_output_dir)

for emotion in emotions:
        
    emotion_dir = os.path.join(base_output_dir , emotion)
        # Target directory
    if not os.path.exists(emotion_dir):
        os.makedirs(emotion_dir)
        
        # Splitting Train and Testing images. 
    num_train_images = int(limit * train_test_split_ratio)

    num_test_images = limit - num_train_images

    try:
        logging.info(f'Download for : {emotion} type has begun !')
        print(f'Download for : {emotion} type has begun !')
        downloader.download(
                emotion, limit = limit, output_dir = emotion_dir, adult_filter_off = True , force_replace = False, timeout = 60 
        )

        logging.info(f'Download successful for : {emotion} !')
        print(f'Download successful for : {emotion} !')

        all_filenames = os.listdir(emotion_dir)

        valid_filenames = []

        # filtering out images wihtout images 
        for filename in all_filenames:
            image_path = os.path.join(emotion_dir, filename)
            if face_check(image_path):
                valid_filenames.append(filename)
            else:
                os.remove(image_path)
                logging.info('File with no image is removed!!')
         
        # Ensuring that I enough valid images
        if len(valid_filenames) < limit:
            logging.info("Not enough valid images !!")
            print('Not enough valid images!!')
            continue    

        train_dir = os.path.join(emotion_dir, 'train')
        test_dir = os.path.join(emotion_dir, 'test')
        os.makedirs(train_dir , exist_ok= True)
        os.makedirs(test_dir, exist_ok= True)


        for i, filename in enumerate(valid_filenames[:limit]):
            image_path = os.path.join(emotion_dir, filename)
            
            if i < num_train_images:
                os.rename(image_path , os.path.join(train_dir, filename))
            else:
                os.rename(image_path, os.path.join(test_dir, filename))
            logging.info('Train and Test Set Filled !')    

    except Exception as e:
        logging.info(f'Download Failure for:{emotion} due to {e}')
        print(f'Download Failure for:{emotion} due to {e}')
        



