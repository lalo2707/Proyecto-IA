import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

from model import get_model
from utils import get_actions, get_sequences_and_labels, create_folder
from config import MAX_LENGTH_FRAMES, MODEL_NAME, NUM_EPOCH, DATA_PATH, MODELS_PATH

def augment_sequence(sequence, noise_level=0.01):
    noise = np.random.normal(0, noise_level, np.array(sequence).shape)
    return (np.array(sequence) + noise).tolist()

def training_model(data_path, model_path):
    actions = get_actions(data_path)
    print(f"Clases detectadas: {actions}")

    sequences, labels = get_sequences_and_labels(actions, data_path)

    # --- Balanceo del dataset ---
    class_indices = {i: [idx for idx, l in enumerate(labels) if l == i] for i in range(len(actions))}
    max_count = max(len(idxs) for idxs in class_indices.values())
    balanced_idx = []
    for i, idxs in class_indices.items():
        if len(idxs) < max_count:
            extra_idx = np.random.choice(idxs, max_count - len(idxs), replace=True)
            balanced_idx.extend(idxs + list(extra_idx))
        else:
            balanced_idx.extend(idxs)
    np.random.shuffle(balanced_idx)
    sequences = [sequences[i] for i in balanced_idx]
    labels = [labels[i] for i in balanced_idx]

    # --- Data augmentation (opcional) ---
    augmented_sequences = [augment_sequence(seq) for seq in sequences[:len(sequences)//2]]
    augmented_labels = labels[:len(labels)//2]
    sequences += augmented_sequences
    labels += augmented_labels

    # --- Preprocesamiento ---
    sequences = pad_sequences(sequences, maxlen=MAX_LENGTH_FRAMES, padding='post', truncating='post', dtype='float32')
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)

    print("Iniciando entrenamiento del modelo...")
    model = get_model(len(actions))
    early_stop = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
    model.fit(X, y, epochs=NUM_EPOCH, callbacks=[early_stop])
    model.summary()

    create_folder(os.path.dirname(model_path))
    model.save(model_path)
    print(f"✅ Modelo guardado en {model_path}")

if __name__ == "__main__":
    data_path = DATA_PATH
    model_path = os.path.join(MODELS_PATH, MODEL_NAME)
    training_model(data_path, model_path)