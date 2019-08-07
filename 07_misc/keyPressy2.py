import pyautogui
from pynput.keyboard import Key, Listener, Controller
import pyautogui
import threading
import time

running = True
clicking = False
keyboard = Controller()

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
        if clicking: 
            x, y = pyautogui.position()
            pyautogui.moveTo(x+2, y+2)
            time.sleep(0.2)
            pyautogui.moveTo(x, y)
            time.sleep(0.2)
            pyautogui.click()
            time.sleep(0.2)
            keyboard.press(Key.down)
            time.sleep(0.2)
            keyboard.press(Key.down)
            time.sleep(0.2)
            keyboard.press(Key.down)
            time.sleep(0.2)
            keyboard.press(Key.down)
            time.sleep(0.2)
            keyboard.press(Key.down)
            time.sleep(0.2)
            keyboard.press(Key.down)
            time.sleep(0.2)
            keyboard.press(Key.enter)
            time.sleep(0.2)
            keyboard.press(Key.enter)
            time.sleep(0.5)
        time.sleep(0.5)

threading.Thread(target=clicker).start()

# Collect events until released
with Listener(on_press=on_press) as listener:
    listener.join()