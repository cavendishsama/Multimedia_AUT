import cv2
import time
import imutils
from imutils.video import VideoStream
from matplotlib import pyplot

detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

vs = VideoStream(src=0).start()
time.sleep(2.0)


c = 0
start = time.time()

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=480)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	rects = detector.detectMultiScale(
                gray, scaleFactor=1.5, minNeighbors=5, minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE)
	
	for (x, y, w, h) in rects:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
	

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	
	if key == ord("q"):
		break

	
	if c == 100: # Print average fps for the last 100 frames
		end = time.time()
		print(frame.shape)
		print("FPS: ", 100/(end-start))
		c = 0
		start = time.time()
	
	c += 1

cv2.destroyAllWindows()
vs.stop()