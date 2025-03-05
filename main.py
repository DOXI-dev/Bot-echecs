import cv2
import pyautogui
import time
import os
import numpy as np
from stockfish import Stockfish

old_name = "echiquier_ancien.png"
new_name = "echiquier_actuel.png"

chessboard_detected = False
region = None

def detect_chessboard():
    global region, chessboard_detected

    time.sleep(3)

    screenshot_detect = pyautogui.screenshot()
    screenshot_detect.save("img.png")

    img = cv2.imread("img.png", 1)
    templ = cv2.imread("Echiquier blanc.png", 1)
    templ2 = cv2.imread("Echiquier noir.png", 1)

    if img is None or templ is None or templ2 is None:
        print("Erreur de chargement des images.")
        chessboard_detected = False
        return

    match = cv2.matchTemplate(img, templ, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)

    match2 = cv2.matchTemplate(img, templ2, cv2.TM_CCOEFF_NORMED)
    min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(match2)

    if max_val >= 0.8:
        screen_width, screen_height = pyautogui.size()
        image_width, image_height = img.shape[1], img.shape[0]

        scale_x = screen_width / image_width
        scale_y = screen_height / image_height

        region = (int(max_loc[0] * scale_x), int(max_loc[1] * scale_y),
                  int(templ.shape[1] * scale_x), int(templ.shape[0] * scale_y))

        print("C'est aux blancs de jouer.")

        chessboard_detected = True

    elif max_val2 >= 0.8:
        screen_width, screen_height = pyautogui.size()
        image_width, image_height = img.shape[1], img.shape[0]

        scale_x = screen_width / image_width
        scale_y = screen_height / image_height

        region = (int(max_loc2[0] * scale_x), int(max_loc2[1] * scale_y),
                  int(templ2.shape[1] * scale_x), int(templ2.shape[0] * scale_y))

        print("C'est aux noirs de jouer.")
        chessboard_detected = True

    else:
        print("Votre image ne correspond pas.")
        chessboard_detected = False

def screenshot():
    if region is not None:
        screen = pyautogui.screenshot(region=region)
        screen.save(new_name)
    else:
        print("Erreur : la région de capture n'est pas définie.")

def read_img():
    if not os.path.exists(old_name) or not os.path.exists(new_name):
        print("Erreur : Une ou les deux images n'existent pas.")
        return

    img1 = cv2.imread("echiquier_actuel.png", 1)
    img2 = cv2.imread("echiquier_ancien.png", 1)

    if img1 is None or img2 is None:
        print("Erreur : une des images n'a pas pu être chargée.")
        return

    if img1.shape != img2.shape:
        print(f"Erreur : les tailles des images ne correspondent pas. img1: {img1.shape}, img2: {img2.shape}")
        return

    diff = cv2.absdiff(img1, img2)

    diff_sum = np.sum(diff)

    seuil = 10000

    if diff_sum > seuil:
        print("Différence détectée !")
    else:
        print("Pas de changement.")

def chess_engine():
    stockfish_path = "/opt/homebrew/bin/stockfish"

    stockfish = Stockfish(path=stockfish_path)

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    stockfish.set_fen_position(fen)

    best_move = stockfish.get_best_move()
    print("Le meilleur coup est:", best_move)

def main():
    detect_chessboard()

    if chessboard_detected:
        while True:
            if os.path.exists(new_name):
                os.rename(new_name, old_name)

            screenshot()
            time.sleep(5)
            read_img()
            chess_engine()  # Calcul du meilleur coup

    else:
        print("Échiquier non détecté, le programme s'arrête.")


if __name__ == "__main__":
    main()