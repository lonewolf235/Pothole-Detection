from deepface import DeepFace
from mtcnn import MTCNN
import cv2
import os
from datetime import datetime

known_faces='D:\pothole_dataset\known_faces'
img_path='D:\pothole_dataset\Linkin-Park.jpg'
result=DeepFace.find(img_path,known_faces,detector_backend='retinaface',model_name='Facenet',enforce_detection=False)

print(result)
print(type(result[0]))