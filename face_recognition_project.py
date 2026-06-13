import cv2
import face_recognition
import numpy as np
from google.colab.patches import cv2_imshow
from google.colab import files
import os
known_face_encodings = []
known_face_names = []
uploaded = files.upload()
for filename in uploaded.keys():
    image = face_recognition.load_image_file(filename)
    encodings = face_recognition.face_encodings(image)
    if encodings:
        known_face_encodings.append(encodings[0])
        known_face_names.append("Aravind")
        print(f"Face loaded: Aravind")
test_upload = files.upload()
for filename in test_upload.keys():
    test_image = face_recognition.load_image_file(filename)
    face_locations = face_recognition.face_locations(test_image)
    face_encodings_list = face_recognition.face_encodings(test_image, face_locations)
    print(f"Found {len(face_locations)} face(s)")
    test_image_bgr = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings_list):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        if known_face_encodings:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match = np.argmin(face_distances)
            if matches[best_match]:
                name = known_face_names[best_match]
                accuracy = (1 - face_distances[best_match]) * 100
                print(f"Recognized: {name} with {accuracy:.1f}% accuracy")
        cv2.rectangle(test_image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(test_image_bgr, (left, bottom-35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(test_image_bgr, name, (left+6, bottom-6),
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
    cv2_imshow(test_image_bgr)
    cv2.imwrite("result.jpg", test_image_bgr)
    files.download("result.jpg")
