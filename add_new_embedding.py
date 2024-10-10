import json
import cv2
from util import load_embeddings, get_embeddings

def save_embeddings(person_name, embedding, embedding_db_path):
    embeddings_db = load_embeddings(embedding_db_path)

    found = 0
    for entry in embeddings_db:
        if entry['name'] == person_name: 
            found = 1
            entry['embedding'] = embedding.tolist()
    
    if not found: embeddings_db.append({"name": person_name, "embedding": embedding.tolist()})
    with open(embedding_db_path, 'w') as f:
        json.dump(embeddings_db, f) 

def add_new_embedding(image_path, person_name, embedding_db_path):
    face_image = cv2.imread(image_path)
    embedding = get_embeddings(face_image)
    save_embeddings(person_name, embedding, embedding_db_path)
