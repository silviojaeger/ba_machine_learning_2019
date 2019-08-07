#import pyautogui
from pynput.keyboard import Key, Listener, Controller
import threading
import time

running = True
clicking = False
keyboard = Controller()

def on_press(key):
    global clicking, running
    if key == Key.f12:
        clicking = not clicking
    if key == Key.esc:
        running = False
        return False

def clicker():
    global clicking, running
    while running:
        if clicking: 
            keyboard.press(Key.enter)
            time.sleep(3)
            keyboard.press(Key.f1)
            time.sleep(0.5)
            keyboard.press(Key.enter)
            time.sleep(25)
            keyboard.press(Key.enter)
            time.sleep(3)
            keyboard.press(Key.enter)
            time.sleep(4)

            #back to folder
            with keyboard.pressed(Key.alt):
                keyboard.press(Key.tab)
                keyboard.release(Key.tab)
                time.sleep(0.5)
                keyboard.press(Key.tab)
                keyboard.release(Key.tab)
            time.sleep(2)
            keyboard.press(Key.down)
            time.sleep(0.2)
            
        

threading.Thread(target=clicker).start()

# Collect events until released
with Listener(on_press=on_press) as listener:
    listener.join()