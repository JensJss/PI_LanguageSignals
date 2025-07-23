import cv2 as cv
import numpy as np
import mediapipe as mp
from Function import *

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

cap = cv.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1) as hands:
    index_cord = []  
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv.flip(image, 1)  
        image.flags.writeable = False
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        image.flags.writeable = True
        image = cv.cvtColor(image_rgb, cv.COLOR_RGB2BGR)

        imgH, imgW = image.shape[:2]
        string = ''

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                hand_cordinate = []
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    x, y = int(landmark.x * imgW), int(landmark.y * imgH)
                    hand_cordinate.append([idx, x, y])
                hand_cordinate = np.array(hand_cordinate)

                
                string = persons_input(hand_cordinate)
                image = get_fram(image, hand_cordinate, string)

                
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

        
        if string.strip():
            cv.rectangle(image, (10, 30), (160, 90), (0, 0, 0), -1) 
            cv.putText(image, f'Letra: {string.strip()}', (20, 75),
                       cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        
        if string == " D":
            index_cord.append([15, hand_cordinate[8][1], hand_cordinate[8][2]])
        if string in [" I", " J"]:
            index_cord.append([15, hand_cordinate[20][1], hand_cordinate[20][2]])

        for val in index_cord[:]:
            image = cv.circle(image, (val[1], val[2]), val[0], (255, 255, 255), -1)
            val[0] -= 1
            if val[0] <= 0:
                index_cord.remove(val)

        cv.imshow('Sign Language Detection', image)

        if cv.waitKey(5) & 0xFF == ord('x'):
            break

cap.release()
cv.destroyAllWindows()
