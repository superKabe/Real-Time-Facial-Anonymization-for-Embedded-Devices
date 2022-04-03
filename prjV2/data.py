import face_recognition
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

known_face_names = []
known_face_encodings = []

for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith("jpg"):
			path = os.path.join(root, file)
			
			image = face_recognition.load_image_file(path)
			image_encode = face_recognition.face_encodings(image)[0]
			known_face_encodings.append(image_encode)

			lables = (os.path.basename(path).replace(".jpg", "").replace("-", " "))
			known_face_names.append(lables)
