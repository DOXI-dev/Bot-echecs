import cv2
import pyautogui
import time
import os
import numpy as np
from stockfish import Stockfish

# Attention, je n'ai pas mis beaucoup de commentaires dans le code si vous avez des questions posez les moi.
# Le code n'est potentiellement pas adapté à tous les PC. Je vais essayer de faire en sorte de le rendre plus adaptable.
# Imports à faire : pip3 install opencv-python, pip3 install pyautogui, pip3 install stockfish.
# Aidez-moi à l'améliorer !

class ChessboardDetector:
    def __init__(self):
        self.old_name = "echiquier_ancien.png"
        self.new_name = "echiquier_actuel.png"
        self.region = None
        self.chessboard_detected = False
        self.board = None
        self.best_move = None

    def detect_chessboard(self):
        time.sleep(3)

        screenshot_detect = pyautogui.screenshot()
        screenshot_detect.save("img.png")

        img = cv2.imread("img.png", 1)
        templ = cv2.imread("Echiquier blanc.png", 1)
        templ2 = cv2.imread("Echiquier noir.png", 1)

        if img is None or templ is None or templ2 is None:
            print("Erreur de chargement des images.")
            self.chessboard_detected = False
            return

        def calculate_region(match_loc, template):
            screen_width, screen_height = pyautogui.size()
            image_width, image_height = img.shape[1], img.shape[0]

            scale_x = screen_width / image_width
            scale_y = screen_height / image_height

            return (int(match_loc[0] * scale_x), int(match_loc[1] * scale_y),
                    int(template.shape[1] * scale_x), int(template.shape[0] * scale_y))

        match = cv2.matchTemplate(img, templ, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(match)

        match2 = cv2.matchTemplate(img, templ2, cv2.TM_CCOEFF_NORMED)
        _, max_val2, _, max_loc2 = cv2.minMaxLoc(match2)

        if max_val >= 0.8:
            self.region = calculate_region(max_loc, templ)
            print("C'est aux blancs de jouer.")
            self.chessboard_detected = True

        elif max_val2 >= 0.8:
            self.region = calculate_region(max_loc2, templ2)
            print("C'est aux noirs de jouer.")
            self.chessboard_detected = True

        else:
            print("Votre image ne correspond pas.")
            self.chessboard_detected = False

    def screenshot(self):
        if self.region is not None:
            screen = pyautogui.screenshot(region=self.region)
            screen.save(self.new_name)
        else:
            print("Erreur : la région de capture n'est pas définie.")

    def read_img(self):
        if not os.path.exists(self.old_name) or not os.path.exists(self.new_name):
            print("Erreur : Une ou les deux images n'existent pas.")
            return

        img1 = cv2.imread(self.new_name, 1)
        img2 = cv2.imread(self.old_name, 1)

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
            self.find_pieces_position()
        else:
            print("Pas de changement.")

    def find_pieces_position(self):
        image = cv2.imread(self.new_name, 1)
        if image is None:
            print("Erreur : Impossible de charger l'image.")
            return

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])

        lower_beige = np.array([20, 20, 50])
        upper_beige = np.array([35, 255, 255])

        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_beige = cv2.inRange(hsv, lower_beige, upper_beige)

        mask = cv2.bitwise_or(mask_green, mask_beige)

        image[mask > 0] = [0, 0, 0]

        cv2.imwrite("image_vert_beige_noir.png", image)

        pieces = {
            "K": "wk.png",
            "Q": "wq.png",
            "R": "wr.png",
            "B": "wb.png",
            "N": "wn.png",
            "P": "wp.png",
            "k": "bk.png",
            "q": "bq.png",
            "r": "br.png",
            "b": "bb.png",
            "n": "bn.png",
            "p": "bp.png"
        }

        chessboard_img = cv2.imread("image_vert_beige_noir.png", 1)
        if chessboard_img is None:
            print("Erreur : Impossible de charger l'image de l'échiquier.")
            return

        rows, cols = 8, 8
        cell_height, cell_width = chessboard_img.shape[0] // rows, chessboard_img.shape[1] // cols

        self.board = [["" for _ in range(cols)] for _ in range(rows)]

        for row in range(rows):
            for col in range(cols):
                x1, y1 = col * cell_width, row * cell_height
                x2, y2 = x1 + cell_width, y1 + cell_height
                cell = chessboard_img[y1:y2, x1:x2]

                cell_gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

                for piece, template_path in pieces.items():
                    if not os.path.exists(template_path):
                        print(f"Erreur : Le modèle pour {piece} est introuvable à {template_path}.")
                        continue

                    template = cv2.imread(template_path, 1)
                    if template is None:
                        print(f"Erreur : Impossible de charger le modèle {template_path}.")
                        continue
                    template = cv2.resize(template, (cell_width, cell_height))

                    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

                    match = cv2.matchTemplate(cell_gray, template_gray, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(match)

                    if max_val >= 0.989:
                        self.board[row][col] = piece
                        break

        for row in self.board:
            print(" ".join(piece if piece else "-" for piece in row))

    @staticmethod
    def board_to_fen(board, active_color="w", castling_rights="KQkq", en_passant="-", halfmove_clock=0,
                     fullmove_number=1):
        """
        Convertit un échiquier en notation FEN.
        """
        fen_rows = []
        for row in board:
            empty_count = 0
            fen_row = ""
            for cell in row:
                if cell == "":
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_row += str(empty_count)
                        empty_count = 0
                    fen_row += cell
            if empty_count > 0:
                fen_row += str(empty_count)
            fen_rows.append(fen_row)

        fen_board = "/".join(fen_rows)
        return f"{fen_board} {active_color} {castling_rights} {en_passant} {halfmove_clock} {fullmove_number}"

    def validate_fen(self, fen):
        """
        Valide une FEN pour s'assurer qu'elle est correcte avant de l'envoyer à Stockfish.
        """
        try:
            stockfish = Stockfish(path="/opt/homebrew/bin/stockfish")
            stockfish.set_fen_position(fen)
            print("La FEN est valide.")
            return True
        except Exception as e:
            print(f"FEN invalide : {e}")
            return False

    def chess_engine(self):
        stockfish_path = "/opt/homebrew/bin/stockfish"
        stockfish = Stockfish(path=stockfish_path)

        if not self.board:
            print("Erreur : L'échiquier n'a pas été détecté ou initialisé.")
            return

        fen = self.board_to_fen(self.board)
        if not self.validate_fen(fen):
            print("Erreur : La FEN générée est invalide.")
            return

        stockfish.set_fen_position(fen)
        self.best_move = stockfish.get_best_move()
        if self.best_move:
            print("Le meilleur coup est :", self.best_move)
            self.play_move(self.best_move)
        else:
            print("Erreur : Aucun coup valide trouvé.")

    def play_move(self, move):
        if not self.region:
            print("Erreur : La région de l'échiquier n'est pas définie.")
            return

        def chess_notation_to_index(notation):
            col = ord(notation[0]) - ord('a')
            row = 8 - int(notation[1])
            return row, col

        start_square = move[:2]
        end_square = move[2:]

        start_row, start_col = chess_notation_to_index(start_square)
        end_row, end_col = chess_notation_to_index(end_square)

        x, y, width, height = self.region
        cell_width = width // 8
        cell_height = height // 8

        start_x = x + start_col * cell_width + cell_width // 2
        start_y = y + start_row * cell_height + cell_height // 2
        end_x = x + end_col * cell_width + cell_width // 2
        end_y = y + end_row * cell_height + cell_height // 2

        pyautogui.click(start_x, start_y)
        time.sleep(0.2)
        pyautogui.click(end_x, end_y)
        pass

    def main(self):
        self.detect_chessboard()
        if self.chessboard_detected:
            self.find_pieces_position()
            if self.board:
                while True:
                    if os.path.exists(self.new_name):
                        os.rename(self.new_name, self.old_name)
                    self.screenshot()
                    time.sleep(5)
                    self.read_img()
                    self.chess_engine()
            else:
                print("Erreur : L'échiquier n'a pas été détecté ou initialisé.")
        else:
            print("Échiquier non détecté, le programme s'arrête.")


if __name__ == "__main__":
    detector = ChessboardDetector()
    detector.main()