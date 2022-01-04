import mediapipe as mp
import cv2 as cv

class HandDetector() : 
	def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5) : 
		self.mode = mode
		self.maxHands = maxHands
		self.detectionCon = detectionCon
		self.trackCon = trackCon

		self.mpHands = mp.solutions.hands
		self.hands = self.mpHands.Hands()
		self.mpDraw = mp.solutions.drawing_utils

	def findHands(self, frame, draw=True) : 
		frameRgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

		results = self.hands.process(frameRgb)
		handPositions = []
		if results.multi_hand_landmarks : 
			for landmarks in results.multi_hand_landmarks : 
				handPositions.append([])
				for id, landmark in enumerate(landmarks.landmark) : 
					h, w, c = frame.shape
					handPositions[-1].append([int(landmark.x * w), int(landmark.y * h)])
				if draw : 
					self.mpDraw.draw_landmarks(frame, landmarks, self.mpHands.HAND_CONNECTIONS)
					
		return (frame, handPositions)