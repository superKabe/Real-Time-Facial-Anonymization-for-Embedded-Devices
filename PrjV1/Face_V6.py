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

# frameWidth = 640
# frameHeight = 480

Do_track = True

cap = cv2.VideoCapture(0)
#time.sleep(2)

# cap.set(3, frameWidth)
# cap.set(4, frameHeight)

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	if	Do_track == True:
	
		faces_eye = face_cascade_eye.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
		for (x, y, w, h) in faces_eye:

			#color = (0, 0, 0) #BGR 0-255
			#stroke = -1
			end_cord_x = x + w
			end_cord_y = y + h
			#cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)
			frame[y:y+h,x:x+w] = cv2.GaussianBlur(frame[y:y+h,x:x+w],(15,15),40,cv2.BORDER_DEFAULT)
			color = (255, 0, 0) #BGR 0-255
			stroke = 1
			cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)

		faces_leftEye = face_cascade_leftEye.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
		for (x, y, w, h) in faces_leftEye:
			
			#color = (0, 0, 0) #BGR 0-255
			#stroke = -1
			end_cord_x = x + w
			end_cord_y = y + h
			#cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)
			frame[y:y+h,x:x+w] = cv2.GaussianBlur(frame[y:y+h,x:x+w],(15,15),40,cv2.BORDER_DEFAULT)
			color = (255, 0, 0) #BGR 0-255
			stroke = 1
			cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)

		faces_rightEye = face_cascade_rightEye.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
		for (x, y, w, h) in faces_rightEye:
			
			#color = (0, 0, 0) #BGR 0-255
			#stroke = -1
			end_cord_x = x + w
			end_cord_y = y + h
			#cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)
			frame[y:y+h,x:x+w] = cv2.GaussianBlur(frame[y:y+h,x:x+w],(15,15),40,cv2.BORDER_DEFAULT)
			color = (255, 0, 0) #BGR 0-255
			stroke = 1
			cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)	
			
		faces_front = face_cascade_front.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
		for (x, y, w, h) in faces_front:
			
			#color = (0, 0, 0) #BGR 0-255
			#stroke = -1
			end_cord_x = x + w
			end_cord_y = y + h
			#cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)
			frame[y:y+h,x:x+w] = cv2.GaussianBlur(frame[y:y+h,x:x+w],(15,15),100000,cv2.BORDER_DEFAULT)
			
			color = (0, 255, 0) #BGR 0-255
			stroke = 1
			cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)

		faces_profile = face_cascade_profile.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
		for (x, y, w, h) in faces_profile:
			
			#color = (0, 0, 0) #BGR 0-255
			#stroke = -1
			end_cord_x = x + w
			end_cord_y = y + h
			#cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)
			frame[y:y+h,x:x+w] = cv2.GaussianBlur(frame[y:y+h,x:x+w],(15,15),100000,cv2.BORDER_DEFAULT)

			color = (0, 255, 0) #BGR 0-255
			stroke = 1
			cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)

		faces_extended = face_cascade_extended.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
		for (x, y, w, h) in faces_extended:
			
			#color = (0, 0, 0) #BGR 0-255
			#stroke = 2
			end_cord_x = x + w
			end_cord_y = y + h
			#cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)
			frame[y:y+h,x:x+w] = cv2.GaussianBlur(frame[y:y+h,x:x+w],(15,15),100000,cv2.BORDER_DEFAULT)

			color = (0, 255, 0) #BGR 0-255
			stroke = 1
			cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)

		bodies_full = body_cascade_full.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
		for (x, y, w, h) in bodies_full:
			
			#color = (0, 0, 0) #BGR 0-255
			#stroke = 2
			end_cord_x = x + w
			end_cord_y = y + h
			#cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)
			frame[y:y+h,x:x+w] = cv2.GaussianBlur(frame[y:y+h,x:x+w],(15,15),40,cv2.BORDER_DEFAULT)

			color = (0, 0, 255) #BGR 0-255
			stroke = 1
			cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)	

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