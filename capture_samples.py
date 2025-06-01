import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from utils import create_folder, draw_keypoints, mediapipe_detection, save_frames, there_hand
from config import FONT, FONT_POS, FONT_SIZE, FRAME_ACTIONS_PATH, ROOT_PATH

def get_exercise_and_posture():
    def get_input_window(prompt_text):
        value = ""
        video = cv2.VideoCapture(0)
        while video.isOpened():
            _, frame = video.read()
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, prompt_text + value, (50, 50), FONT, FONT_SIZE, (0, 255, 0), 1)
            cv2.imshow("Input", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter
                break
            elif key == 8:  # Backspace
                value = value[:-1]
            elif key != 255:
                value += chr(key)
        video.release()
        cv2.destroyAllWindows()
        return value.strip()

    ejercicio = get_input_window("Introduce el nombre del ejercicio: ")
    postura = get_input_window("Introduce el tipo de postura: ")
    if ejercicio and postura:
        etiqueta = f"{ejercicio}_{postura}"
        return etiqueta
    return None


def capture_samples(path, margin_frame=2, min_cant_frames=5):
    create_folder(path)

    cant_sample_exist = len(os.listdir(path))
    count_sample = 0
    frames = []

    with Holistic() as holistic_model:
        video = cv2.VideoCapture(0)

        capturing = False

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)  # Voltear horizontalmente

            image, results = mediapipe_detection(frame, holistic_model)

            key = cv2.waitKey(10) & 0xFF

            if key == ord('c'):  # Empieza o para la captura con la tecla 'c'
                capturing = not capturing
                if not capturing and len(frames) > min_cant_frames:
                    output_folder = os.path.join(path, f"sample_{cant_sample_exist + count_sample + 1}")
                    create_folder(output_folder)
                    save_frames(frames, output_folder)
                    count_sample += 1
                    frames = []
                elif not capturing:
                    frames = []

            if capturing:
                cv2.putText(image, 'Capturando...', FONT_POS, FONT, FONT_SIZE, (255, 50, 0), 2)
                frames.append(np.asarray(frame))
            else:
                cv2.putText(image, 'Presiona "c" para capturar', FONT_POS, FONT, FONT_SIZE, (0, 220, 100), 2)

            draw_keypoints(image, results)

            cv2.imshow(f'Toma de muestras para "{os.path.basename(path)}"', image)

            if key == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    etiqueta = get_exercise_and_posture()
    if etiqueta:
        word_path = os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH, etiqueta)
        capture_samples(word_path)
    else:
        print("No se ingresó una etiqueta válida.")

