import cv2 as cv
import handDetector

capture = cv.VideoCapture(1)

detector = handDetector.HandDetector()

while True : 
	playing, frame = capture.read()

	frame, handPositions = detector.findHands(frame, draw=False)

	fingerCount = 0
	for hand in handPositions : 
		if hand[4][1] < hand[5][1] : 
			fingerCount += 1
		if hand[8][1] < hand[6][1] : 
			fingerCount += 1
		if hand[12][1] < hand[10][1] : 
			fingerCount += 1
		if hand[16][1] < hand[14][1] : 
			fingerCount += 1
		if hand[20][1] < hand[18][1] : 
			fingerCount += 1

	cv.putText(frame, f'Finger count : {fingerCount}', (40, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (124, 56, 289))

	cv.imshow('Frame', frame)
	if cv.waitKey(1) & 0xFF == ord('d') : 
		break

capture.release()

cv.destroyAllWindows()