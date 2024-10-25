import cv2
from util import get_embeddings, save_embeddings

def add_new_embedding_from_path(image_path, person_name, embedding_db_path):
    face_image = cv2.imread(image_path)
    embedding = get_embeddings(face_image)
    save_embeddings(person_name, embedding, embedding_db_path)

def add_new_embedding_from_image(image, person_name, embedding_db_path):
    embedding = get_embeddings(image)
    save_embeddings(person_name, embedding, embedding_db_path)
