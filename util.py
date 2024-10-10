import numpy as np
import cv2
from ultralytics import YOLO
import tensorflow as tf
import json

model_path = 'best.pt'

model = YOLO(model_path)
VGG_model = tf.keras.applications.VGG16(weights = 'imagenet')
base_model = tf.keras.models.Model(inputs = VGG_model.input, outputs = VGG_model.get_layer('fc2').output)


def load_embeddings(embedding_db_path):
    try:
        with open(embedding_db_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    
def preprocess(image):
    image = cv2.resize(image, (224,224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.vgg16.preprocess_input(image)
    return image

def get_embeddings(image):
    result = model(image)[0]

    for box in result.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = box
        if score > 0.5: 
            cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
            preprocessed_face = preprocess(cropped_image)
            preprocessed_face = np.expand_dims(preprocessed_face, axis=0) 
            embedding = base_model.predict(preprocessed_face) 
            return embedding
    
    return None