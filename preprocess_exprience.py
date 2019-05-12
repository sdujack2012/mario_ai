import numpy as np
import cv2
import pprint
import _pickle
import random
from PIL import Image
from multiprocessing.dummy import Pool as ThreadPool 

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'

death_penalty = -100
do_nothing_penalty = -10

def calculate_reward(input_data1, input_data2):  
    method = eval('cv2.TM_CCOEFF_NORMED')
    ret,thresh1 = cv2.threshold(input_data1[16:24, 23:72],240,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    ret,thresh2 = cv2.threshold(input_data2[16:24, 23:72],240,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    thresh1 = Image.fromarray(255-thresh1, 'L').resize((100,int(100 * 7 / 48)),Image.ANTIALIAS)
    thresh2 = Image.fromarray(255-thresh2, 'L').resize((100,int(100 * 7 / 48)),Image.ANTIALIAS)

    score1 = 0
    score2 = 0

    try:
        score1 = int(pytesseract.image_to_string(thresh1, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789'))
        score2 = int(pytesseract.image_to_string(thresh2, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789'))
    except:
        print("No score")

    print(score1, score2, score2 - score1)
    res = cv2.matchTemplate(input_data1,input_data2[:, 0:80],method)
    min_val, max_value, min_loc, max_loc = cv2.minMaxLoc(res)
    print("#### rewards", max_loc, max_value)
    final_reward = score2 - score1 + max_loc[0]
    final_reward = final_reward if final_reward > 0 else do_nothing_penalty
    print("#### final rewards", final_reward)
    return final_reward

def calculate_reward_array(experience):
    if experience["terminal"] == True:
        print("#### death rewards", death_penalty)
        return death_penalty
    else:
        return calculate_reward(experience["screenshot"], np.array(experience["next_screenshot"]))

def calculate_rewards(experiences):
    pool = ThreadPool(10) 
    rewards = pool.map(calculate_reward_array, experiences)
    print(rewards)
    for i in range(len(experiences)):
            experiences[i]["actual_reward"] = rewards[i]

# def main():
    
#     experiences = []
#     with open('./experiences', 'rb') as pickle_file:
#         experiences = _pickle.load(pickle_file, encoding='latin1')

#     pool = ThreadPool(10) 
#     reward_results = pool.map(calculate_reward_array, experiences)

#     for i in range(len(experiences)):
#         experience = experiences[i]
#         experience["final_reward"] = reward_results[i]

        
#     _pickle.dump(experiences, open('experiences', 'wb')) 


# if __name__ == "__main__":
#     main()