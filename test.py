import numpy as np
import cv2
from util import get_embeddings
import sqlite3

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

        name, embedding_blob = entry
        stored_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
        similarity = cosine_similarity(test_embedding, stored_embedding)  

        if similarity > best_similarity:  
            best_similarity = similarity
            best_match_name = name
    
    print(best_match_name)
    return best_match_name

def test_from_path(image_path, embedding_db_path):
    image = cv2.imread(image_path)
    embedding = get_embeddings(image)
    if embedding is None: return None
    return find_best_match(embedding, embedding_db_path)

def test_from_image(image, embedding_db_path):
    embedding = get_embeddings(image)
    if embedding is None: return None
    return find_best_match(embedding, embedding_db_path)