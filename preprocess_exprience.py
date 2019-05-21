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
shoot_enemy_reward = 0.1

enemy_exploded = np.array(cv2.imread('./resources/enemy_exploded.png', 0))


def detect_reward(input_data1, input_data2):
    thresh = 0.7
    method = eval('cv2.TM_CCOEFF_NORMED')
    res1 = cv2.matchTemplate(input_data1, enemy_exploded, method)
    res2 = cv2.matchTemplate(input_data2, enemy_exploded, method)
    min_val, max_value1, min_loc, max_loc = cv2.minMaxLoc(res1)
    min_val, max_value2, min_loc, max_loc = cv2.minMaxLoc(res2)
    print(max_value1, max_value2)
    if max_value1 < thresh and max_value2 >= thresh:
        print("######################################### reward")
        return True
    return False


def calculate_reward(input_data1, input_data2):
    score = 0
    if(detect_reward(input_data1, input_data2)):
        score = shoot_enemy_reward

    final_reward = score if score > 0 else do_nothing_penalty
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
