import pickle
import cv2
import time

face_cascade_front = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt.xml')
face_cascade_profile = cv2.CascadeClassifier('data/haarcascades/haarcascade_profileface.xml')
face_cascade_extended = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalcatface_extended.xml')
face_cascade_eye = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye.xml')
face_cascade_glasses = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
face_cascade_leftEye = cv2.CascadeClassifier('data/haarcascades/haarcascade_lefteye_2splits.xml')
face_cascade_rightEye = cv2.CascadeClassifier('data/haarcascades/haarcascade_righteye_2splits.xml')

body_cascade_full = cv2.CascadeClassifier('data/haarcascades/haarcascade_fullbody.xml')


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face-trainner.yml")

lables = {}
with open("face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

# frameWidth = 640
# frameHeight = 480

Do_track = True

cap = cv2.VideoCapture(0)

# color_val=(255, 0, 0)

def track(roi_part, color_val, stroke_val):
	for (x, y, w, h) in roi_part:
		end_cord_x = x + w
		end_cord_y = y + h
		roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)

		for i in range(4):
			frame[y:y+h,x:x+w] = cv2.GaussianBlur(frame[y:y+h,x:x+w],(15,15),1000,cv2.BORDER_DEFAULT)
		
		cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color_val, stroke_val)

		id_, conf = recognizer.predict(roi_gray)
		if conf>=4 and conf <= 85:
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_].replace("-", " ")
			color = (0, 255, 0)
			stroke = 2
			cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)


# def blur()

#time.sleep(2)

# cap.set(3, frameWidth)
# cap.set(4, frameHeight)


while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	if	Do_track == True:

		#BGR 0-255
		
		# Tracking face
		roi_part = face_cascade_front.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
		track(roi_part, (0, 255, 0), 1)

		# Tracking face profile
		roi_part = face_cascade_profile.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
		track(roi_part, (0, 255, 0), 1)

		# Tracking face extended
		roi_part = face_cascade_extended.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
		track(roi_part, (0, 255, 0), 1)
		
		# Tracking eyes
		roi_part = face_cascade_eye.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
		track(roi_part, (255, 0, 0), 1)

		# Tracking L eye
		roi_part = face_cascade_leftEye.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
		track(roi_part, (255, 0, 0), 1)

		# Tracking R eye
		roi_part = face_cascade_rightEye.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
		track(roi_part, (255, 0, 0), 1)

		# Tracking full body
		roi_part = body_cascade_full.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
		track(roi_part, (0, 0, 255), 1)

	# Display the resulting frame
	cv2.imshow('frame',frame)
	#time.sleep(0.1)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

	if cv2.waitKey(20) & 0xFF == ord('w'):
		Do_track = True

	if cv2.waitKey(20) & 0xFF == ord('e'):
		Do_track = False

# When everything done, reease the capture
cap.release()
cv2.destroyAllWindows()


