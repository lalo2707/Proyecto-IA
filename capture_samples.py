import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from utils import create_folder, draw_keypoints, mediapipe_detection, save_frames, there_hand
from config import FONT, FONT_POS, FONT_SIZE, FRAME_ACTIONS_PATH, ROOT_PATH

def get_word_from_window():
    word = ""
    input_active = False

    def draw_text(image, text, position, font, font_scale, font_thickness, color):
        cv2.putText(image, text, position, font, font_scale, color, font_thickness)

    def capture_input():
        nonlocal word, input_active
        input_active = True

        video = cv2.VideoCapture(0)
        while video.isOpened():
            _, frame = video.read()
            frame = cv2.flip(frame, 1)  # Voltear la imagen horizontalmente
            draw_text(frame, "Introduce la palabra y presiona 'Enter': " + word, (50, 50), FONT, FONT_SIZE, 1, (0, 255, 0))
            cv2.imshow("Introduce la palabra", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter key
                break
            elif key == 8:  # Backspace key
                word = word[:-1]
            elif key != 255:  # Other keys
                word += chr(key)

        video.release()
        cv2.destroyAllWindows()
        input_active = False
        return word.strip()

    return capture_input()

def capture_samples(path, margin_frame=2, min_cant_frames=5):
    create_folder(path)

    cant_sample_exist = len(os.listdir(path))
    count_sample = 0
    count_frame = 0
    frames = []

    with Holistic() as holistic_model:
        video = cv2.VideoCapture(0)



        while video.isOpened():
            _, frame = video.read()
            frame = cv2.flip(frame, 1)  # Voltear la imagen horizontalmente



            image, results = mediapipe_detection(frame, holistic_model)

            if there_hand(results):
                count_frame += 1
                if count_frame > margin_frame:
                    cv2.putText(image, 'Capturando...', FONT_POS, FONT, FONT_SIZE, (255, 50, 0))
                    frames.append(np.asarray(frame))

            else:
                if len(frames) > min_cant_frames + margin_frame:
                    frames = frames[:-margin_frame]
                    output_folder = os.path.join(path, f"sample_{cant_sample_exist + count_sample + 1}")
                    create_folder(output_folder)
                    save_frames(frames, output_folder)
                    count_sample += 1

                frames = []
                count_frame = 0
                cv2.putText(image, 'Listo para capturar...', FONT_POS, FONT, FONT_SIZE, (0, 220, 100))

            draw_keypoints(image, results)

            # Muestra el frame redimensionado en la ventana
            cv2.imshow(f'Toma de muestras para "{os.path.basename(path)}"', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    word_name = get_word_from_window()
    if word_name:
        word_path = os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH, word_name)
        capture_samples(word_path)
    else:
        print("No se ingresó ninguna palabra.")
