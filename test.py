import numpy as np
import cv2
from util import load_embeddings, get_embeddings

def cosine_similarity(embedding1, embedding2):
    embedding1 = embedding1.flatten()
    embedding2 = embedding2.flatten()

    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    if norm1 == 0 or norm2 == 0: return 0.0
    
    similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
    return similarity

def find_best_match(test_embedding, embedding_db_path):
    embeddings_db = load_embeddings(embedding_db_path)

    best_match_name = None
    best_similarity = -1  

    for entry in embeddings_db:
        stored_embedding = np.array(entry["embedding"]) 
        similarity = cosine_similarity(test_embedding, stored_embedding)  

        if similarity > best_similarity:  
            best_similarity = similarity
            best_match_name = entry["name"]
    
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