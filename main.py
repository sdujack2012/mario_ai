
import tensorflow as tf
from tensorflow import keras
import numpy as np
import win32com.client
import win32api
import win32gui
import win32con
import time
import cv2
import pprint
from secrets import randbelow
from collections import deque
import random
import _pickle
import os

from memory_db import MemoryDB
from nes_io import IO
from agent import Agent
from preprocess_exprience import calculate_rewards
from train_model import train_with_experience

discount = 0.98

sample_size = 60
epoch = 1

training_before_update_target = 100
max_steps = 9000


def main():
    esplison = 0.7
    esplison_decay = 0.001

    episode = 0
    io_instance = IO("FCEUX 2.2.3: bee")
    memorydb_instance = MemoryDB('localhost', 'bee-ai', 'replay-memory1')
    agent_instance = Agent((120, 150, 4), False)

    i = 1
    while True:
        esplison -= esplison_decay
        episode += 1
        steps = 0
        print("episode: ", episode)
        print("experiences size: ", memorydb_instance.get_experiences_size())
        experience_bacth = []
        io_instance.focus_window()
        io_instance.reset()
        is_termnial = False

        previous_screenshot = io_instance.get_screenshot()
        previous_image_state = io_instance.get_stacked_frames(
            previous_screenshot, True)

        while is_termnial != True:
            steps += 1
            print("steps: ", steps)
            experience = {}
            experience["terminal"] = False
            experience["screenshot"] = previous_screenshot
            experience["image_state"] = previous_image_state
            dice = random.uniform(0.0, 1.0)
            action_index = 0

            if dice >= esplison:
            #if True:
                reward = agent_instance.model_predict(experience["image_state"].reshape(
                    1, 120, 150, 4))

                print("###reward: ", reward)
                action_index = np.argmax(reward).item()
                print("Model selected action and rewards:", action_index,
                      io_instance.action_mapping_name[action_index])
            else:
                action_index = random.randint(0, 2)

                print("Random selected action:", action_index,
                      io_instance.action_mapping_name[action_index])
            io_instance.action(action_index)

            experience["action_index"] = action_index

            experience["next_screenshot"] = previous_screenshot = io_instance.get_screenshot()
            experience["next_image_state"] = previous_image_state = io_instance.get_stacked_frames(
                previous_screenshot, False)

            experience_bacth.append(experience)

            if io_instance.is_termnial(experience["next_screenshot"]):
                experience["terminal"] = True
                is_termnial = True

        calculate_rewards(experience_bacth)
        memorydb_instance.add_batch(experience_bacth)

        print("New Experiences Added")

        sampled_experiences, b_idx, b_ISWeights = memorydb_instance.sample(
            sample_size)
        print("sampled_experiences: ", len(sampled_experiences))

        errors = train_with_experience(
            agent_instance, sampled_experiences, b_ISWeights, epoch, discount)

        agent_instance.save_model()
        memorydb_instance.update_batch(b_idx, errors, sampled_experiences)

        if i % training_before_update_target == 0:
            agent_instance.sync_target()
        i += 1


if __name__ == "__main__":
    main()
