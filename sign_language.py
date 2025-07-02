import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow import keras
import math
import time

class HandTracking:
    def __init__(self, mode = False, maxHands = 1, detectionCon=0.5, trackCon=0.5 ):
        self.__mode__ = mode
        self.__maxHands__ = maxHands
        self.__detectionCon__ = detectionCon
        self.__trackCon__ = trackCon

        self.handsMp = mp.solutions.hands
        self.hands = self.handsMp.Hands()
        self.mpDraw = mp.solutions.drawing_utils


#Detect hand keypoints:

    def findfingers(self, frame, draw = True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.handsMp.HAND_CONNECTIONS)
        return frame
    def findposition(self, frame, handNo = 0, draw = True):
        xList = []
        yList = []
        zList = []
        box = []
        self.lmsList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h,w,c = frame.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z)
                xList.append(cx)
                yList.append(cy)
                zList.append(cz)
                self.lmsList.append([id, cx, cy, cz])
                if draw:
                    cv2.circle(frame, (cx,cy), 5, (255,0,255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            box = xmin, ymin, xmax, ymax
            if draw:
                cv2.rectangle(frame,(xmin-20, ymin-20),(xmax+20, ymax+20), (0,255,0),2)
        return self.lmsList, box

    def normalize(self, lmsList, box):
        normalized = []
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin if xmax - xmin != 0 else 1
        height = ymax - ymin if ymax - ymin != 0 else 1
        for _,x,y,z in lmsList:

            lm_x = (x - xmin)/width
            lm_y = (y - ymin)/height
            normalized.append([lm_x, lm_y, z])

        return normalized

    def global_angle(self, lmsList):
        wrist_x, wrist_y = lmsList[0][1], lmsList[0][2]
        middle_base_x, middle_base_y = lmsList[9][1], lmsList[9][2]
        dx = middle_base_x - wrist_x
        dy = middle_base_y - wrist_y
        angle = math.atan2(dy, dx)
        angle_normalized = angle / math.pi
        return angle_normalized


model = keras.models.load_model('/Users/macbookair/Desktop/my_model_box_normalization.keras')
class_names = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']


def main():

    cap = cv2.VideoCapture(0)
    detector = HandTracking()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    start_time = time.time()
    delay = 3
    if not cap.isOpened():
        print('Camera cannot be opened')

        exit()
    word = ''
    letter = ''
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Cannot receive frame')
            break
        frame = detector.findfingers(frame)
        if detector.results.multi_hand_landmarks:
            lmslist, box = detector.findposition(frame, draw = True)
            if lmslist and len(lmslist) == 21:
                normalized_lms = detector.normalize(lmslist, box)
                angle = detector.global_angle(lmslist)
                input_data = np.array(normalized_lms).flatten()
                input_data = np.append(input_data, angle)
                input_data = input_data.reshape(1, -1)
                preds = model.predict(input_data)
                letter_index = np.argmax(preds)
                letter = class_names[letter_index]
                # cv2.putText(frame, letter, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            current_time = time.time()
            if current_time - start_time>=delay:
                if letter:
                    if letter =='0' and ( len(word)==0 or word[-1].isalpha() or word[-1].isspace()):
                        letter = 'o'
                        word += letter
                    else:
                        word += letter
                start_time = current_time


        cv2.imshow('frame', frame)
        word_image = 255 * np.ones((200, 640, 3), dtype=np.uint8)
        cv2.putText(word_image, f"{word}", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
        cv2.imshow('Word Window', word_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            word+=' '
        elif key == 127:
            word = word[:-1]
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
        main()
