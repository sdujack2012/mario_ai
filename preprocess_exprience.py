import numpy as np
import cv2
import pprint
import _pickle
import random
from PIL import Image
from multiprocessing.dummy import Pool as ThreadPool

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'

death_penalty = -1
do_nothing_penalty = -0.01
progress_rewards = 0.02

point100 = np.array(cv2.imread('./resources/point100.png', 0))
point200 = np.array(cv2.imread('./resources/point200.png', 0))
point1000 = np.array(cv2.imread('./resources/point1000.png', 0))


def detect_100_reward(input_data1, input_data2):
    thresh = 0.7
    method = eval('cv2.TM_CCOEFF_NORMED')
    res1 = cv2.matchTemplate(input_data1, point100, method)
    res2 = cv2.matchTemplate(input_data2, point100, method)
    min_val, max_value1, min_loc, max_loc = cv2.minMaxLoc(res1)
    min_val, max_value2, min_loc, max_loc = cv2.minMaxLoc(res2)

    return max_value1 < thresh and max_value2 >= thresh


def detect_200_reward(input_data1, input_data2):
    thresh = 0.7
    method = eval('cv2.TM_CCOEFF_NORMED')
    res1 = cv2.matchTemplate(input_data1, point200, method)
    res2 = cv2.matchTemplate(input_data2, point200, method)
    min_val, max_value1, min_loc, max_loc = cv2.minMaxLoc(res1)
    min_val, max_value2, min_loc, max_loc = cv2.minMaxLoc(res2)

    return max_value1 < thresh and max_value2 >= thresh


def detect_1000_reward(input_data1, input_data2):
    thresh = 0.7
    method = eval('cv2.TM_CCOEFF_NORMED')
    res1 = cv2.matchTemplate(input_data1, point1000, method)
    res2 = cv2.matchTemplate(input_data2, point1000, method)
    min_val, max_value1, min_loc, max_loc = cv2.minMaxLoc(res1)
    min_val, max_value2, min_loc, max_loc = cv2.minMaxLoc(res2)

    return max_value1 < thresh and max_value2 >= thresh


def detect_1000(input_data1):
    thresh = 0.7
    method = eval('cv2.TM_CCOEFF_NORMED')
    res1 = cv2.matchTemplate(input_data1, point1000, method)
    min_val, max_value1, min_loc, max_loc = cv2.minMaxLoc(res1)

    return max_value1 > thresh


def calculate_reward(input_data1, input_data2):
    # ret,thresh1 = cv2.threshold(input_data1[16:24, 23:72],240,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # ret,thresh2 = cv2.threshold(input_data2[16:24, 23:72],240,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # thresh1 = Image.fromarray(255-thresh1, 'L').resize((100,int(100 * 7 / 48)),Image.ANTIALIAS)
    # thresh2 = Image.fromarray(255-thresh2, 'L').resize((100,int(100 * 7 / 48)),Image.ANTIALIAS)

    # score1 = 0
    # score2 = 0
    # try:
    #     score1 = int(pytesseract.image_to_string(thresh1, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789'))
    #     score2 = int(pytesseract.image_to_string(thresh2, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789'))
    # except:
    #     print("bad scores")

    # print(score1, score2, score2 - score1)

    score = 0
    if(detect_1000_reward(input_data1, input_data2)):
        score = 0.1
    elif(detect_200_reward(input_data1, input_data2)):
        score = 0.1
    elif(detect_100_reward(input_data1, input_data2)):
        score = 0.1

    method = eval('cv2.TM_CCOEFF_NORMED')
    res = cv2.matchTemplate(input_data1, input_data2[24:, 0:80], method)
    min_val, max_value, min_loc, max_loc = cv2.minMaxLoc(res)
    final_reward = score + max_loc[0] * progress_rewards
    final_reward = final_reward if final_reward > 0 else do_nothing_penalty
    return final_reward

def calculate_reward_array(experience):
    if experience["terminal"] == True:
        print("#### death penalty", death_penalty)
        return death_penalty
    else:
        final_reward = calculate_reward(
            experience["screenshot"], experience["next_screenshot"])
        print("#### normal rewards", final_reward)
        return final_reward


def calculate_rewards(experiences):
    pool = ThreadPool(20)
    rewards = pool.map(calculate_reward_array, experiences)
    for i in range(len(experiences)):
        experiences[i]["actual_reward"] = rewards[i]
