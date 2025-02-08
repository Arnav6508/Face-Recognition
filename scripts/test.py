import cv2
from utils import get_embeddings, find_best_match
from ultralytics import YOLO

model_path = 'best.pt'
embedding_db_path = 'test.db'

model = YOLO(model_path)

def test_from_path(image_path, embedding_db_path):
    image = cv2.imread(image_path)
    return test_from_image(image, embedding_db_path)

def test_from_image(image, embedding_db_path):
    embedding = get_embeddings(image)
    if embedding is None: return None
    return find_best_match(embedding, embedding_db_path)

def test_from_webcam(embedding_db_path = embedding_db_path, threshold = 0.5):
    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()
    while ret == True:

        results = model(frame)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)

            if score > threshold:

                cropped_image = frame[int(y1): int(y2), int(x1): int(x2)]
                label = test_from_image(cropped_image, embedding_db_path)

                if label != None:
                    cv2.putText(frame, label, (int(x1), int(y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                
                else:
                    cv2.putText(frame, 'Unknown', (int(x1), int(y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

        ret, frame = cap.read()
        
    cap.release()
    cv2.destroyAllWindows()