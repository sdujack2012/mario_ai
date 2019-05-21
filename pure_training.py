
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
from nes_io import IO, max_occurence_last_two_screens_same
from agent import Agent
from preprocess_exprience import calculate_rewards
from train_model import train_with_experience

discount = 0.8

sample_size = 300
epoch = 1

training_before_update_target = 100
max_steps = 900

def main():
    memorydb_instance = MemoryDB('localhost', 'mario-ai', 'replay-memory')
    agent_instance = Agent((120, 120, 4), True)

    i = 1
    while True:
        print("experiences size: ", memorydb_instance.get_experiences_size())
    
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
