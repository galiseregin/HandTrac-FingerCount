import cv2
import mediapipe as mp
import time
import os

u_operation = "+"  # user operation default


class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_con=0.6, model_complex=1, track_con=0.5):
        self.results = None
        self.mode = mode
        self.MaxHands = max_hands  # number of hands to track
        self.detectionCon = detection_con
        self.trackCon = track_con
        self.modelComplex = model_complex

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.MaxHands, self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils  # draw option

    # find the positions of hands and fingers, store it in self.results
    # input: all self objects, img (display-screen), draw (true or false)
    # output: img
    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # make img rgb
        self.results = self.hands.process(img_rgb)  # track and find hands locations

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)  # draw hands
        return img  # return the img with drawing hands

    # process the results from find_hands() function
    # input: all self objects, img (display-screen), hands_number, draw (true or false)
    # output: two lists of hands with positions (lm_list, lm_list2)
    def find_position(self, img, hand_num=0, draw=True):

        lm_list = []  # hand1 list coordinates
        lm_list2 = []  # hand2 list coordinates

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_num]
            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
                if draw:
                    if id == 4 or id == 8 or id == 12 or id == 16 or id == 20:  # draw circle on 35 pointers (fingers tops)
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            try:
                my_hand2 = self.results.multi_hand_landmarks[1]
                for id, lm in enumerate(my_hand2.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list2.append([id, cx, cy])
                    if draw:
                        if (id == 4 or id == 8 or id == 12 or id == 16 or id == 20) and draw is True:  # draw circle on 35 pointers (fingers tops)
                            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            except:
                pass

        return lm_list, lm_list2


# count the fingers up
# input: img (display screen), two lists of hands, list of images fingers, hand_type(right, left)
# output: img (display screen), fingers_num (count-int)
def finger_count(img, lm_list, overlay_list, hand_type):
    fingers_num = -1  # default None (-1)
    some_ids = [4, 8, 12, 16, 20]  # tops of fingers

    # image width and height and positions (default)
    a = (0, 375)
    b = (0, 287)

    # if there is a hand in screen do...
    if len(lm_list) != 0:
        fingers = []

        # Thumb (check if thumb is up)
        if hand_type == "Right Hand":
            a = (0, 375)
            b = (0, 287)
            if lm_list[some_ids[0]][1] > lm_list[some_ids[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        elif hand_type == "Left Hand":
            # 1366 768 gali resolution
            a = (0, 375)
            b = (993, 1366)
            if lm_list[some_ids[0]][1] > lm_list[some_ids[0] - 1][1]:
                fingers.append(0)
            else:
                fingers.append(1)
        elif hand_type == "None":
            print("Hand Type Error!!!!!!")

        # 4 Fingers (check if 4 fingers are up)
        for id in range(1, 5):
            if lm_list[some_ids[id]][2] < lm_list[some_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        finger01 = ""
        for f in fingers:
            finger01 = finger01 + f"{f}"

        fingers_num = finger01.count('1')

        if finger01 == "01000":
            img[a[0]:a[1], b[0]:b[1]] = overlay_list[0]
        elif finger01 == "01100":
            img[a[0]:a[1], b[0]:b[1]] = overlay_list[1]
        elif finger01 == "01110":
            img[a[0]:a[1], b[0]:b[1]] = overlay_list[2]
        elif finger01 == "01111":
            img[a[0]:a[1], b[0]:b[1]] = overlay_list[3]
        elif finger01 == "11111":
            img[a[0]:a[1], b[0]:b[1]] = overlay_list[4]
        elif finger01 == "00111":
            img[a[0]:a[1], b[0]:b[1]] = overlay_list[6]
        elif finger01 == "11000":
            img[a[0]:a[1], b[0]:b[1]] = overlay_list[7]
        elif finger01 == "00000":
            img[a[0]:a[1], b[0]:b[1]] = overlay_list[8]

    return img, fingers_num


# check left-hand or right-hand
# input: list of hands with positions
# output: hand-type (right, left, None) - str
def l_r_hand(lm_list):
    hand_type = "None"
    try:
        if lm_list[4][1] < lm_list[17][1]:
            hand_type = "Left Hand"
        elif lm_list[4][1] > lm_list[17][1]:
            hand_type = "Right Hand"
    except:
        hand_type = "None"

    return hand_type


# selection user operation
# input: img (display screen), list of hands with positions
# output: u_operation ( +, -, /, * ) - default ( + )
def select(img, lm_list):
    global u_operation
    # 1366 768 gali resolution
    # 1920 = 960
    # 1366 = 683

    img = cv2.rectangle(img, (550, 5), (650, 105), (0, 255, 0), 3)  # drawing rectangle1
    img = cv2.rectangle(img, (670, 5), (770, 105), (0, 255, 0), 3)  # drawing rectangle2
    img = cv2.rectangle(img, (430, 5), (530, 105), (0, 255, 0), 3)  # drawing rectangle3
    img = cv2.rectangle(img, (790, 5), (890, 105), (0, 255, 0), 3)  # drawing rectangle4
    try:
        if 550 < lm_list[8][1] < 650 and 5 < lm_list[8][2] < 105:  # selection positions for "-"
            u_operation = "-"
            cv2.circle(img, (600, 52), 15, (255, 0, 255), cv2.FILLED)  # draw a circle at user_select
        elif 670 < lm_list[8][1] < 770 and 5 < lm_list[8][2] < 105:  # selection positions for "*"
            u_operation = "*"
            cv2.circle(img, (720, 52), 15, (255, 0, 255), cv2.FILLED)  # draw a circle at user_select
        elif 430 < lm_list[8][1] < 530 and 5 < lm_list[8][2] < 105:  # selection positions for "+"
            u_operation = "+"
            cv2.circle(img, (480, 52), 15, (255, 0, 255), cv2.FILLED)  # draw a circle at user_select
        elif 790 < lm_list[8][1] < 890 and 5 < lm_list[8][2] < 105:  # selection positions for "/"
            u_operation = "/"
            cv2.circle(img, (840, 52), 15, (255, 0, 255), cv2.FILLED)  # draw a circle at user_select
    except:
        pass

    cv2.putText(img, f"{u_operation}", (610, 200), cv2.FONT_HERSHEY_PLAIN, 8, (255, 0, 255), 7)

    return u_operation


def main():
    global u_operation
    u_operation = "+"

    folder_path = "FingerImages"
    overlay_list = []
    my_list = os.listdir(folder_path)
    for imagePath in my_list:
        image = cv2.imread(f"{folder_path}/{imagePath}")
        overlay_list.append(image)

    # wCam, hCam = 1280, 720
    w_cam, h_cam = 1366, 768
    cap = cv2.VideoCapture(0)
    cap.set(3, w_cam)
    cap.set(4, h_cam)
    p_time = 0
    detector2 = HandDetector()

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector2.find_hands(img)
        lm_list, lm_list2 = detector2.find_position(img, draw=False)
        if len(lm_list) != 0:
            u_operation = select(img, lm_list)
            hand_type1 = l_r_hand(lm_list)
            hand_type2 = l_r_hand(lm_list2)
            img, fingers_num1 = finger_count(img, lm_list, overlay_list, hand_type1)
            img, fingers_num2 = finger_count(img, lm_list2, overlay_list, hand_type2)
            print(fingers_num1, fingers_num2)

            if fingers_num1 != -1 and fingers_num2 != -1:
                if u_operation == "+":
                    cv2.putText(img, f"{fingers_num2} + {fingers_num1} = {fingers_num2 + fingers_num1}", (150, 550), cv2.FONT_HERSHEY_PLAIN, 8, (255, 0, 255), 7)
                elif u_operation == "-":
                    cv2.putText(img, f"{fingers_num2} - {fingers_num1} = {fingers_num2 - fingers_num1}", (150, 550), cv2.FONT_HERSHEY_PLAIN, 8, (255, 0, 255), 7)
                elif u_operation == "*":
                    cv2.putText(img, f"{fingers_num2} * {fingers_num1} = {fingers_num2 * fingers_num1}", (150, 550), cv2.FONT_HERSHEY_PLAIN, 8, (255, 0, 255), 7)
                elif u_operation == "/":
                    if fingers_num1 == 0:
                        cv2.putText(img, f"Error divide by 0", (50, 550), cv2.FONT_HERSHEY_PLAIN, 8, (255, 0, 255), 7)
                    else:
                        cv2.putText(img, f"{fingers_num2} / {fingers_num1} = {round(fingers_num2 / fingers_num1, 3)}", (150, 550), cv2.FONT_HERSHEY_PLAIN, 8, (255, 0, 255), 7)

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(img, f"{int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
