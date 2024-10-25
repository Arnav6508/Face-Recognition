import numpy as np
import cv2
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO
import tensorflow as tf
import torch
import sqlite3

model_path = 'best.pt'
model = YOLO(model_path)
base_model = InceptionResnetV1(pretrained='vggface2').eval()

def preprocess(image):
    img = cv2.resize(image, (160,160)) 
    img = np.array(img).astype(np.float32)
    img = (img - 127.5) / 128.0   # scales to [-1,1]

    img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    img = np.expand_dims(img, axis=0) 
    img = torch.tensor(img)
    return img

def get_embeddings(image):
    result = model(image)[0]

    for box in result.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = box
        if score > 0.5: 
            cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
            preprocessed_face = preprocess(cropped_image)
            embedding = base_model(preprocessed_face)
            return embedding.detach().numpy()
    
    return None

def initialize_db(embedding_db_path):
    conn = sqlite3.connect(embedding_db_path)
    cursor = conn.cursor()

    cursor.execute('''
            CREATE TABLE IF NOT EXISTS Embeddings(
                name TEXT,
                embedding BLOB
            )
        '''
        )
    conn.commit()
    conn.close()

def save_embeddings(person_name, embedding, embedding_db_path):

    initialize_db(embedding_db_path)

    conn = sqlite3.connect(embedding_db_path)
    cursor = conn.cursor()

    # check if name already exists
    cursor.execute('SELECT name FROM Embeddings WHERE name = ?',(person_name,))
    is_name_present = cursor.fetchone()
    if is_name_present: 
        conn.close()
        return False

    # insert new entry
    embedding_blob = np.array(embedding).astype(np.float32).tobytes()
    cursor.execute('INSERT INTO Embeddings (name, embedding) VALUES(?,?)',(person_name,embedding_blob))
    conn.commit()
    conn.close()