# import warnings
# warnings.filterwarnings('ignore', message='GStreamer')

import face_recognition
import cv2
import numpy as np
import data_loader

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

#Getting the images from another file
known_face_encodings = data_loader.known_face_encodings
known_face_names = data_loader.known_face_names

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

Do_track = True
Do_recognition = True
cooldown = 0
prev_face_location = []
prev_face_names = []

while True:
	# Grab a single frame of video
	ret, frame = video_capture.read()

	# Resize frame of video to 1/4 size for faster face recognition processing
	small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

	# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
	rgb_small_frame = small_frame[:, :, ::-1]

	# Only with track if toggled
	if Do_track:

		# Only process every other frame of video to save time
		if process_this_frame:

			
			# Find all the faces and face encodings in the current frame of video
			face_locations = face_recognition.face_locations(rgb_small_frame)
			face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

			face_names = []
			for face_encoding in face_encodings:
				# See if the face is a match for the known face(s)
				matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
				name = "Unknown"

				# Use the known face with the smallest distance to the new face
				face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
				best_match_index = np.argmin(face_distances)
				if matches[best_match_index]:
					name = known_face_names[best_match_index]

				face_names.append(name)

			# Gives the application a buffer if it misses a few frames
			if (face_locations == []):
				if (cooldown < 6):
					cooldown = cooldown + 1
					# print(cooldown)
					
					face_locations = prev_face_location
					face_names = prev_face_names
			else:
				cooldown=0
				# print("Resting cooldown")

		
			
	
		process_this_frame = not process_this_frame
		# process_this_frame = True
			
		# print(name)
		
		# Display the results
		for (top, right, bottom, left), name in zip(face_locations, face_names):
			
			# Will just bypass the Recontion
			if Do_recognition == False:
				name = "Unknown"
				
			# Creating a buffer
			prev_face_location = face_locations
			prev_face_names = face_names

			# Scale back up face locations since the frame we detected in was scaled to 1/4 size
			top *= 4
			right *= 4
			bottom *= 4
			left *= 4

			# print(name)

			# Blurring the face if is labled as unknown
			if (name == "Unknown"):
				for i in range(4):
					frame[top:bottom, left:right] = cv2.GaussianBlur(frame[top:bottom, left:right], (99, 99), 30)

			# Draw a box around the face
			cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)

			# Draw a label with a name below the face
			cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
			font = cv2.FONT_HERSHEY_DUPLEX
			cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
		

	# Display the resulting image
	cv2.imshow('Live feed', frame)

	# Hit 'q' on the keyboard to quit!
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	if cv2.waitKey(1) & 0xFF == ord('w'):
		Do_track = False
		print("Tracking: " + str(Do_track))
	
	if cv2.waitKey(1) & 0xFF == ord('e'):
		Do_track = True
		print("Tracking: " + str(Do_track))

	if cv2.waitKey(1) & 0xFF == ord('t'):
		Do_recognition = False
		print("Recognition: " + str(Do_recognition))
		
	if cv2.waitKey(1) & 0xFF == ord('r'):
		Do_recognition = True
		if Do_track:
			print("Recognition: " + str(Do_recognition))
		else:
			Do_track = True
			print("Recognition: " + str(Do_recognition) + " and setting tracking to True")

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
