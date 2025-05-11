import os
import cv2
import numpy as np
import pandas as pd
from mediapipe.python.solutions.holistic import Holistic, FACEMESH_CONTOURS, POSE_CONNECTIONS, HAND_CONNECTIONS
from mediapipe.python.solutions.drawing_utils import draw_landmarks, DrawingSpec

# Crea una carpeta si no existe
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Realiza la detección utilizando el modelo de Mediapipe
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Dibuja los puntos clave en la imagen
def draw_keypoints(image, results):
    draw_landmarks(image, results.face_landmarks, FACEMESH_CONTOURS, DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1), DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    draw_landmarks(image, results.pose_landmarks, POSE_CONNECTIONS, DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4), DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    draw_landmarks(image, results.left_hand_landmarks, HAND_CONNECTIONS, DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4), DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    draw_landmarks(image, results.right_hand_landmarks, HAND_CONNECTIONS, DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4), DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

# Verifica si hay una mano en los resultados de Mediapipe
def there_hand(results):
    return results.left_hand_landmarks or results.right_hand_landmarks

# Guarda los frames en una carpeta específica
def save_frames(frames, output_folder):
    for num_frame, frame in enumerate(frames):
        frame_path = os.path.join(output_folder, f"{num_frame + 1}.jpg")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA))

# Extrae los keypoints de los resultados de Mediapipe
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# Obtiene las acciones (nombres de archivos .h5) desde un directorio específico
def get_actions(path):
    return [name.split('.')[0] for name in os.listdir(path) if name.endswith('.h5')]

# Da formato a las oraciones detectadas
def format_sentences(sent, sentence, repe_sent):
    if len(sentence) > 1 and sent in sentence[1]:
        repe_sent += 1
        sentence.pop(0)
        sentence[0] = f"{sent} (x{repe_sent})"
    else:
        repe_sent = 1
    return sentence, repe_sent

# Obtiene los keypoints de una muestra
def get_keypoints(model, path):
    kp_seq = np.array([])
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        frame = cv2.imread(img_path)
        _, results = mediapipe_detection(frame, model)
        kp_frame = extract_keypoints(results)
        kp_seq = np.concatenate([kp_seq, [kp_frame]] if kp_seq.size > 0 else [[kp_frame]])
    return kp_seq

# Inserta la secuencia de keypoints en un DataFrame
def insert_keypoints_sequence(df, n_sample, kp_seq):
    for frame, keypoints in enumerate(kp_seq):
        data = {'sample': n_sample, 'frame': frame + 1, 'keypoints': [keypoints]}
        df_keypoints = pd.DataFrame(data)
        df = pd.concat([df, df_keypoints])
    return df

# Guarda contenido en un archivo .txt
def save_txt(file_name, content):
    with open(file_name, 'w') as archivo:
        archivo.write(content)

# Obtiene las secuencias y etiquetas para el entrenamiento del modelo
def get_sequences_and_labels(actions, data_path):
    sequences, labels = [], []  # Inicializar las listas como vacías
    for label, action in enumerate(actions):
        hdf_path = os.path.join(data_path, f"{action}.h5")
        if os.path.exists(hdf_path):
            data = pd.read_hdf(hdf_path, key='data')
            for _, data_filtered in data.groupby('sample'):
                sequences.append([fila['keypoints'] for _, fila in data_filtered.iterrows()])
                labels.append(label)
        else:
            print(f"Archivo {hdf_path} no encontrado.")
    print(f"Sequences: {len(sequences)}, Labels: {len(labels)}")  # Agregar una línea para depuración
    return sequences, labels
