import pyautogui
from pynput.keyboard import Key, Listener
import threading
import time

running = True
clicking = False

def on_press(key):
    global clicking, running
    if key == Key.f2:
        clicking = not clicking
    if key == Key.esc:
        running = False
        return False

def clicker():
    global clicking, running
    while running:
        if clicking: pyautogui.click()
        time.sleep(0.005)

threading.Thread(target=clicker).start()

# Collect events until released
with Listener(on_press=on_press) as listener:
    listener.join()