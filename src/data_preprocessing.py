# src/data_preprocessing.py
import os
import numpy as np
import cv2

def load_and_preprocess_data(data_dir):
    images = []
    labels = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        for img_file in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (224, 224))  # redimensionner l'image
            images.append(img)
            labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

if __name__ == "__main__":
    data_dir = "path/to/data"
    images, labels = load_and_preprocess_data(data_dir)
    np.save('data/images.npy', images)
    np.save('data/labels.npy', labels)