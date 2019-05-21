
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
import datetime
from collections import deque
from PIL import Image

P = 0x50
Enter = 0x0D
top_cutoff = 70
frames_to_take = 4
time_per_frame = 1/30

stack_size = 4  # We stack 4 frames
frame_size = (150, 120)
screenshot_size = (224, 256)
# Initialize deque with zero-images one array for each image
S = 0x53
D = 0x44
H = 0x48

def image_difference(image1, image2):
    dif = np.sum(np.sum(np.abs(p1-p2) for p1, p2 in zip(image1, image2)))
    return (dif / 255.0 * 100) / image1.size


class IO:
    def __init__(self, window_name):
        self.window = win32gui.FindWindow(None, window_name)
        self.action_mapping = [S, D, H]
        self.action_mapping_name = ['Fire', 'Left', 'Right']
        self.exploded = np.array(cv2.imread('./resources/exploded.png', 0))

        self.stacked_frames = deque([np.zeros(frame_size)
                                     for i in range(stack_size)], maxlen=4)

    def focus_window(self):
        win32gui.SetForegroundWindow(self.window)

    def action(self, actionIndex):
        win32api.keybd_event(
                self.action_mapping[actionIndex], 0, 0, 0)
        time.sleep(time_per_frame)        
        win32api.keybd_event(
                self.action_mapping[actionIndex], 0, win32con.KEYEVENTF_KEYUP, 0)


    def reset(self):
        time.sleep(1)
        win32api.keybd_event(P, 0, 0, 0)
        time.sleep(0.3)
        win32api.keybd_event(P, 0, win32con.KEYEVENTF_KEYUP, 0)
        time.sleep(3)

    def pause(self):
        win32api.keybd_event(Enter, 0, 0, 0)
        time.sleep(0.3)
        win32api.keybd_event(Enter, 0, win32con.KEYEVENTF_KEYUP, 0)

    def get_screenshot(self, isSavingScreenshot=False):
        rect = win32gui.GetClientRect(self.window)
        offset = win32gui.ClientToScreen(self.window, (0, 0))
        im = PIL.ImageGrab.grab()
        im = im.crop((rect[0]+offset[0], rect[1]+offset[1],
                      rect[2]+offset[0], rect[3]+offset[1])).convert('L')
        if isSavingScreenshot:
            currentDT = datetime.datetime.now()
            im.save(f"./screenshots/screenshots_{currentDT.strftime('%H%M%S')}.png")

        return np.array(im)

    def process_screenshot(self, screenshot):
        croped_screenshot = Image.fromarray(screenshot[35:224, :])
        resized_screenshot = croped_screenshot.resize(
            frame_size, Image.BICUBIC)
        normalized_screenshot = np.asarray(resized_screenshot) / 255.0
        return normalized_screenshot

    def get_stacked_frames(self, screenshot, is_new_episode):
        frame = self.process_screenshot(screenshot)
        if is_new_episode:
            self.stacked_frames = deque(
                [np.zeros(frame_size) for i in range(stack_size)], maxlen=stack_size)
            self.stacked_frames.append(frame)
            self.stacked_frames.append(frame)
            self.stacked_frames.append(frame)
            self.stacked_frames.append(frame)
        else:
            self.stacked_frames.append(frame)

        stacked_state = np.stack(self.stacked_frames, axis=2)
        return stacked_state

    def is_termnial(self, input_data):
        method = eval('cv2.TM_CCOEFF_NORMED')
        res1 = cv2.matchTemplate(input_data, self.exploded, method)
        threshold1 = 0.7

        min_val, max_val1, min_loc, max_loc = cv2.minMaxLoc(res1)

        print("#### dying", max_val1)
        if max_val1 > threshold1:
            self.get_screenshot(True)
            return True

        return False