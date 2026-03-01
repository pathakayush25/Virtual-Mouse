import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math

# Camera size
cam_width = 640
cam_height = 480

# Smoothening
smoothening = 5
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

# Setup camera
cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Get screen size
screen_width, screen_height = pyautogui.size()

click_delay = 0.3
last_click_time = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * cam_width), int(lm.y * cam_height)
                lm_list.append((id, cx, cy))

            if lm_list:

                # Index finger tip (8)
                x1, y1 = lm_list[8][1:]

                # Middle finger tip (12)
                x2, y2 = lm_list[12][1:]

                # Thumb tip (4)
                x_thumb, y_thumb = lm_list[4][1:]

                # Move mouse
                screen_x = np.interp(x1, (0, cam_width), (0, screen_width))
                screen_y = np.interp(y1, (0, cam_height), (0, screen_height))

                curr_x = prev_x + (screen_x - prev_x) / smoothening
                curr_y = prev_y + (screen_y - prev_y) / smoothening

                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y

                # Click (distance between index & middle)
                distance = math.hypot(x2 - x1, y2 - y1)

                if distance < 30:
                    if time.time() - last_click_time > click_delay:
                        pyautogui.click()
                        last_click_time = time.time()

                # Scroll
                if y_thumb < y1 - 40:
                    pyautogui.scroll(40)

                if y_thumb > y1 + 40:
                    pyautogui.scroll(-40)

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Virtual Mouse", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()