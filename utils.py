import numpy as np
import cv2
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO
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
                person_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
        '''
        )
    conn.commit()
    conn.close()

def save_embeddings(person_id, person_name, embedding, embedding_db_path):

    initialize_db(embedding_db_path)

    conn = sqlite3.connect(embedding_db_path)
    cursor = conn.cursor()

    # check if name already exists
    cursor.execute('SELECT name FROM Embeddings WHERE person_id = ?',(person_id,))
    is_name_present = cursor.fetchone()
    if is_name_present: 
        cursor.execute('UPDATE Embeddings SET name = ?, embedding = ? WHERE person_id = ?', (person_name, embedding, person_id))
        conn.close()
        return 

    # insert new entry
    embedding_blob = np.array(embedding).astype(np.float32).tobytes()
    cursor.execute('INSERT INTO Embeddings (person_id, name, embedding) VALUES(?,?,?)',(person_id, person_name, embedding_blob))
    conn.commit()
    conn.close()

def cosine_similarity(embedding1, embedding2):
    embedding1 = embedding1.flatten()
    embedding2 = embedding2.flatten()

    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    if norm1 == 0 or norm2 == 0: return 0.0
    
    similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
    return similarity

def find_best_match(test_embedding, embedding_db_path):
    conn = sqlite3.connect(embedding_db_path)
    cursor = conn.cursor()

    best_match_name = None
    best_similarity = -1  

    cursor.execute('SELECT * FROM Embeddings')
    embeddings_db = cursor.fetchall()

    for entry in embeddings_db:

        per_id, name, embedding_blob = entry
        stored_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
        similarity = cosine_similarity(test_embedding, stored_embedding)  

        if similarity > best_similarity:  
            best_similarity = similarity
            best_match_name = name
            best_match_id = per_id
    
    print(best_match_name)
    return best_match_id, best_match_name

def add_new_embedding_from_path(image_path, person_id, person_name, embedding_db_path):
    image = cv2.imread(image_path)
    add_new_embedding_from_image(image, person_id, person_name, embedding_db_path)

def add_new_embedding_from_image(image, person_id, person_name, embedding_db_path):
    embedding = get_embeddings(image)
    save_embeddings(person_id, person_name, embedding, embedding_db_path)

def initialize_attendance_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
            CREATE TABLE IF NOT EXISTS Attendance(
                person_id INTEGER,
                name TEXT NOT NULL,
                date DATE NOT NULL,
                time TIME NOT NULL,
                FOREIGN KEY (person_id) REFERENCES Embeddings(id)
            )
        '''
        )
    conn.commit()
    conn.close()

def mark_attendance(person_id, name, date, time, db_path):

    initialize_attendance_db(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('INSERT into Attendance (person_id, name, date, time) VALUES(?, ?, ?, ?)', (person_id, name, date, time))

    conn.commit()
    conn.close()

def print_attendance_record(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('SELECT * from Attendance')
    records = cursor.fetchall()
    for record in records: print(record)

    conn.commit()
    conn.close()

def delete_attendance_record(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('DELETE from Attendance')

    conn.commit()
    conn.close()