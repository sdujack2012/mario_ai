
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pyscreenshot as ImageGrab
import win32com.client
import win32api
import win32gui
import win32con
import time
import cv2
import pprint
from secrets import randbelow
import random
import _pickle
import os

from nes_io import IO
from agent import Agent
from preprocess_exprience import calculate_rewards
from train_model import train_with_experience, sample_distribution
discount = 0.7

sample_size = 32
epoch = 50
training_before_update_target = 10
esplison = 0.6
esplison_decay = 0.01

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
def main():
    esplison = 0.6
    io_instance = IO("FCEUX 2.2.3: mario")
    io_instance.reset()

    agent_instance = Agent((100, 150, 4), True)

    experiences = []
    if os.path.exists("experiences_collection"):
        with open('./experiences_collection', 'rb') as pickle_file:
            print("existing experiences loaded")
            
            experiences = _pickle.load(pickle_file, encoding='latin1')
    i = 1
    while True:
        print(f"experiences size: {len(experiences)}")
        experience_bacth = []
        io_instance.focus_window()
        io_instance.reset()
        is_termnial = False
        is_new_episode = True
        
        while is_termnial != True:
            experience = {}
            experience["terminal"] = False
            experience["screenshot"] = io_instance.get_screenshot()
            experience["image_state"] = io_instance.get_stacked_frames(experience["screenshot"], is_new_episode)
            experience["device_state"] = io_instance.get_device_state()
            is_new_episode = False
            
            dice = random.uniform(0.0, 1.0)
            action_index = 0
             
            if dice >= esplison:
                reward = agent_instance.model_predict([experience["image_state"].reshape(1, 100, 150, 4), experience["device_state"].reshape(1, 4)])
                action_index = np.argmax(reward)
                print("Model selected action and rewards:", action_index, reward)
            else:
                action_index = random.randint(0, 3)
                print("Random selected action:", action_index)
            io_instance.action(action_index)
            
            experience["action_index"] = action_index
            experience["next_screenshot"] = io_instance.get_screenshot()
            experience["next_image_state"] = io_instance.get_stacked_frames(experience["next_screenshot"], is_new_episode)
            experience["next_device_state"] = io_instance.get_device_state()
            experience_bacth.append(experience)

            if io_instance.is_termnial(experience["next_screenshot"]):
                experience["terminal"] = True
                is_termnial = True

        calculate_rewards(experience_bacth)

        experiences += experience_bacth
        _pickle.dump(experiences, open('experiences_collection', 'wb'))
        print("Experiences updated")
        # train_with_experience(agent_instance, experience_bacth, sample_size, epoch, discount)
        train_with_experience(agent_instance, experiences, sample_size, epoch, discount)
        agent_instance.save_model()  
        
        esplison -= esplison_decay

        if i % training_before_update_target == 0:
            agent_instance.sync_target()
        i += 1
         
        

if __name__ == "__main__":
    main()