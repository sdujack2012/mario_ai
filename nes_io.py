
import numpy as np
import pyscreenshot as ImageGrab
import win32api
import win32gui
import win32con
import win32ui
import time
import os
import cv2
import PIL.ImageGrab
from collections import deque
from PIL import Image

P = 0x50
Enter = 0x0D
top_cutoff = 70
frames_to_take = 4
time_per_frame = 1/15

stack_size = 4  # We stack 4 frames
frame_size = (150, 100)
screenshot_size = (224, 256)
# Initialize deque with zero-images one array for each image


class IO:
    def __init__(self, window_name):
        self.window = win32gui.FindWindow(None, window_name)
        self.action_mapping = [0x41, 0x44, 0x48, 0x53]
        self.action_mapping_name = ['Jump', 'Right', 'Right', 'Fire']
        self.dead_mario = np.array(cv2.imread('./resources/dead_mario.png', 0))
        self.blackout = np.array(cv2.imread('./resources/blackout.png', 0))
        self.pressed_key = np.array([0, 0, 0, 0])

        self.stacked_frames = deque([np.zeros(frame_size)
                                     for i in range(stack_size)], maxlen=4)

        self.last_screenshots = []
        self.i = 1

    def focus_window(self):
        win32gui.SetForegroundWindow(self.window)

    def action(self, actionIndex):
        if self.pressed_key[actionIndex] == 1:
            win32api.keybd_event(
                self.action_mapping[actionIndex], 0, win32con.KEYEVENTF_KEYUP, 0)
        else:
            win32api.keybd_event(self.action_mapping[actionIndex], 0, 0, 0)
        self.pressed_key[actionIndex] = 1 - self.pressed_key[actionIndex]
        time.sleep(time_per_frame)

    def reset(self):
        for action in self.action_mapping:
            win32api.keybd_event(action, 0, win32con.KEYEVENTF_KEYUP, 0)
        self.pressed_key[...] = 0
        time.sleep(0.5)
        win32api.keybd_event(P, 0, 0, 0)
        time.sleep(0.3)
        win32api.keybd_event(P, 0, win32con.KEYEVENTF_KEYUP, 0)
        time.sleep(4)

    def pause(self):
        win32api.keybd_event(Enter, 0, 0, 0)
        time.sleep(0.3)
        win32api.keybd_event(Enter, 0, win32con.KEYEVENTF_KEYUP, 0)

    def get_screenshot(self):
        rect = win32gui.GetClientRect(self.window)
        offset = win32gui.ClientToScreen(self.window, (0, 0))
        im = PIL.ImageGrab.grab()
        im = im.crop((rect[0]+offset[0], rect[1]+offset[1],
                      rect[2]+offset[0], rect[3]+offset[1])).convert('L')
        im.save(f"./screenshots/screenshots{self.i}.png")
        self.i = (self.i + 1) % 1000
        return np.array(im)

    def process_screenshot(self, screenshot):
        croped_screenshot = Image.fromarray(screenshot[74:224, :])
        resized_screenshot = croped_screenshot.resize(
            frame_size, Image.BICUBIC)
        normalized_screenshot = np.asarray(
            resized_screenshot).reshape(100, 150) / 255.0
        return normalized_screenshot

    def get_stacked_frames(self, screenshot, is_new_episode):
        # Preprocess frame
        frame = self.process_screenshot(screenshot)

        if is_new_episode:
            # Clear our stacked_frames
            self.stacked_frames = deque(
                [np.zeros(frame_size) for i in range(stack_size)], maxlen=stack_size)
            # Because we're in a new episode, copy the same frame 4x
            self.stacked_frames.append(frame)
            self.stacked_frames.append(frame)
            self.stacked_frames.append(frame)
            self.stacked_frames.append(frame)
        else:
            # Append frame to deque, automatically removes the oldest frame
            self.stacked_frames.append(frame)

            # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(self.stacked_frames, axis=2)
        return stacked_state

    def get_device_state(self):
        return self.pressed_key

    def is_termnial(self, input_data):
        method = eval('cv2.TM_CCOEFF_NORMED')
        res1 = cv2.matchTemplate(input_data, self.dead_mario, method)
        res2 = cv2.matchTemplate(input_data, self.blackout, method)
        threshold1 = 0.7
        threshold2 = 0.7

        min_val, max_val1, min_loc, max_loc = cv2.minMaxLoc(res1)
        min_val, max_val2, min_loc, max_loc = cv2.minMaxLoc(res2)

        print("#### dying", max_val1, max_val2)
        return max_val1 > threshold1 or max_val2 > threshold2
