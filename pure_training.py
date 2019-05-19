
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
from multiprocessing.pool import ThreadPool
from memory_db import MemoryDB
from nes_io import IO
from agent import Agent
from preprocess_exprience import calculate_rewards
from train_model import train_with_experience
discount = 0.7

sample_size = 50
epoch = 1

training_before_update_target = 10
esplison = 0.6
esplison_decay = 0.01
maxlen = 1000


def main():
    esplison = 0.6
    memorydb_instance = MemoryDB('localhost', 'mario-ai', 'replay-memory')
    agent_instance = Agent((100, 150, 4), False)
    i = 1

    while True:
        print(f"experiences size: {memorydb_instance.get_experiences_size()}")
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
