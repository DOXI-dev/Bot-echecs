import cv2
import pyautogui
import time
import os
import numpy as np

old_name = "echiquier_ancien.png"
new_name = "echiquier_actuel.png"

def screenshot():
    screen = pyautogui.screenshot()
    screen.save(new_name)

def read_img():
    if not os.path.exists(old_name) or not os.path.exists(new_name):
        print("Erreur : Une ou les deux images n'existent pas.")
        return

    img1 = cv2.imread("echiquier_actuel.png", 1)
    img2 = cv2.imread("echiquier_ancien.png", 1)

    if img1 is None or img2 is None:
        print("Erreur : une des images n'a pas pu être chargée.")
        return

    diff = cv2.absdiff(img1, img2)

    diff_sum = np.sum(diff)

    seuil = 10000

    if diff_sum > seuil:
        print("Différence détectée !")
    else:
        print("Pas de changement.")

while True:
    if os.path.exists(new_name):
        os.rename(new_name, old_name)

    screenshot()
    time.sleep(5)
    read_img()