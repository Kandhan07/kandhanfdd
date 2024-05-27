import argparse
import os
import time
from threading import Thread
import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils
from imutils.video import VideoStream
from scipy.spatial import distance as dist
import playsound
import math
import face_recognition

# Main class to run the program
class DrowsinessDetection:
    def __init__(self):
        self.EYE_AR_THRESH = 0.3
        self.EYE_AR_CONSEC_FRAMES = 10
        self.YAWN_THRESH = 30
        self.alarm_status = False
        self.alarm_status2 = False
        self.saying = False
        self.COUNTER = 0
        self.detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.vs = VideoStream(src=args["webcam"]).start()
        time.sleep(1.0)
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.known_face_encodings = []
        self.known_face_names= []
        self.process_current_frame = True
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f'faces/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)
        
        print(self.known_face_names)
    
    # Define face confidence function
    def face_confidence_value(self, face_distance_val, face_match_threshold=0.6):
        confidence_range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance_val) / (confidence_range*2.0)

        if face_distance_val > face_match_threshold:
            return str(round(linear_val * 100, 2)) + '%'
        else:
            value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5)*2, 0.2))) * 100
            return str(round(value, 2))+ '%'

    def alarm(self, msg):
        while self.alarm_status:
            print('call')
            s = 'espeak "'+msg+'"'
            os.system(s)

        if self.alarm_status2:
            print('call')
            self.saying = True
            s = 'espeak "' + msg + '"'
            os.system(s)
            self.saying = False

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def final_ear(self, shape):
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = self.eye_aspect_ratio(leftEye)
        rightEAR = self.eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0
        return (ear, leftEye, rightEye)

    def lip_distance(self, shape):
        top_lip = shape[50:53]
        top_lip = np.concatenate((top_lip, shape[61:64]))

        low_lip = shape[56:59]
        low_lip = np.concatenate((low_lip, shape[65:68]))

        top_mean = np.mean(top_lip, axis=0)
        low_mean = np.mean(low_lip, axis=0)

        distance = abs(top_mean[1] - low_mean[1])
        return distance

    def run(self):
        while True:
            frame = self.vs.read()
            frame = imutils.resize(frame, width=850)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
            
            for (x, y, w, h) in rects:
                rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                eye = self.final_ear(shape)
                ear = eye[0]
                leftEye = eye[1]
                rightEye = eye[2]

                distance = self.lip_distance(shape)

                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                lip = shape[48:60]
                cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

                if ear < self.EYE_AR_THRESH:
                    self.COUNTER += 1

                    if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                        if self.alarm_status == False:
                            self.alarm_status = True
                            t = Thread(target=self.alarm, args=('wake up sir',))
                            t.deamon = True
                            t.start()
                        
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    self.COUNTER = 0
                    self.alarm_status = False

                if (distance > self.YAWN_THRESH):
                        cv2.putText(frame, "YAWN ALERT", (20, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        if self.alarm_status2 == False and self.saying == False:
                            self.alarm_status2 = True
                            t = Thread(target=self.alarm, args=('take some fresh air sir',))
                            t.deamon = True
                            t.start()
                else:
                    self.alarm_status2 = False

                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            # Integrate face recognition
            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = 'Unknown'
                    confidence = 'Unknown'

                    face_distances_val =face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances_val)

                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = self.face_confidence_value(face_distances_val[best_match_index])
                    
                    self.face_names.append(f'{name} ({confidence})')
            
            self.process_current_frame = not self.process_current_frame

            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0,0,255), -1)
                cv2.putText(frame, name, (left+6, bottom-6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 1)

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

        cv2.destroyAllWindows()
        

# Instantiate the main class and run the program
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
    args = vars(ap.parse_args())

    dd = DrowsinessDetection()
    dd.run()
