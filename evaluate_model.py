import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
from keras.models import load_model
from mediapipe.python.solutions.holistic import Holistic
from utils import draw_keypoints, extract_keypoints, get_actions
from config import DATA_PATH, FONT, FONT_POS, FONT_SIZE, MAX_LENGTH_FRAMES, MODELS_PATH, MODEL_NAME
from text_to_speech import text_to_speech
import time

# Diccionario de feedback por clase detectada
feedback = {
    "sentadilla_correcta": "¡Buena postura! Continúa así.",
    "sentadilla_incorrecta": "Ajusta tu postura, espalda recta.",
}

# Cargar modelo y acciones
model = load_model(os.path.join(MODELS_PATH, MODEL_NAME))
actions = get_actions(DATA_PATH)
sequence = []

last_alert_time = 0
alert_cooldown = 2  # segundos

with Holistic() as holistic:
    cap = cv2.VideoCapture(0)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        frame_count += 1
        if frame_count % 2 != 0:  # Procesa solo 1 de cada 2 frames
            continue
        frame = cv2.flip(frame, 1)
        image, results = frame.copy(), holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw_keypoints(image, results)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        if len(sequence) > MAX_LENGTH_FRAMES:
            sequence = sequence[-MAX_LENGTH_FRAMES:]

        # Solo predecir si todos los frames tienen keypoints válidos (no solo ceros)
        if len(sequence) == MAX_LENGTH_FRAMES and all(np.sum(np.abs(f)) > 0 for f in sequence):
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            pred_class = actions[np.argmax(res)]
            mensaje = feedback.get(pred_class, f"Ejercicio detectado: {pred_class}")
            color = (0, 255, 0) if "correcta" in pred_class else (0, 0, 255)
            cv2.putText(image, mensaje, FONT_POS, FONT, FONT_SIZE, color, 2, cv2.LINE_AA)

            # Sonido solo si es incorrecta y cooldown
            if "incorrecta" in pred_class:
                current_time = time.time()
                if current_time - last_alert_time > alert_cooldown:
                    text_to_speech(mensaje)
                    last_alert_time = current_time

        cv2.imshow('Posture Corrector', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()