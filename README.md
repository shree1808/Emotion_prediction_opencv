# Real-Time Facial Emotion Analysis Application 

![Bing Image Downloader API](image-1.png) +  ![OpenCV](open_cv.png) + ![Tensorflow + Keras](image.png)

## Introduction

Classifying emotion is crucial in the modern digital era, especially in social media platforms where understanding user sentiments can enhance user experience, improve content recommendations, and provide better customer support. Accurately identifying emotions in real-time can enable more personalized interactions and foster stronger community engagement.

## Data Collection, Ingestion, and Processing

For this project, I utilized the Bing Image Downloader API to collect open-source images from the internet, categorized into seven distinct emotions:
- Sad
- Fearful
- Neutral
- Disgusted
- Happy
- Surprised
- Angry

To ensure the images were accurately captured and free from noise, OpenCV's frontal face Haar Cascade was used alongside the Bing Image Downloader API. This combination ensured that only relevant facial images were downloaded and processed.

## Image Preprocessing

Due to class imbalances in some categories, proper data augmentation techniques and weight assignments were employed. This helped balance the dataset and improve model performance. Other preprocessing steps included resizing images, normalizing pixel values, and converting images to grayscale for consistency.


## Model Development

For predicting emotions, First I have performed a comparative analysis between a custom Convolutional Neural Network (CNN) and MobileNetV2. The evaluation determined that MobileNetV2 outperformed the custom CNN in terms of accuracy and efficiency. Consequently, MobileNetV2 was selected for integration with OpenCV's Video Capture for live emotion detection.

The application was built using OpenCV's frontal face Haar Cascade to detect faces in real-time, and MobileNetV2 to classify the detected emotions.

## Conclusion

This Real-Time Facial Emotion Analysis Application effectively classifies emotions, leveraging advanced image processing and deep learning techniques. The application demonstrates the potential of integrating real-time emotion detection in various domains, especially in enhancing user interactions on social media platforms.

&copy;
Shree Sudame 
2024