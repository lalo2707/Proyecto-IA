import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
from utils import get_actions, get_sequences_and_labels
from config import MAX_LENGTH_FRAMES, MODEL_NAME, DATA_PATH, MODELS_PATH
import os

# Cargar acciones y datos
actions = get_actions(DATA_PATH)
sequences, labels = get_sequences_and_labels(actions, DATA_PATH)
sequences = pad_sequences(sequences, maxlen=MAX_LENGTH_FRAMES, padding='post', truncating='post', dtype='float32')
X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Divide en entrenamiento y validación
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=labels)

# Cargar el modelo ya entrenado
model_path = os.path.join(MODELS_PATH, MODEL_NAME)
model = load_model(model_path)

# Predice en validación
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

# Imprime la matriz de confusión y el reporte
print("Matriz de confusión:")
print(confusion_matrix(y_true, y_pred_classes))
print("\nReporte de clasificación:")
print(classification_report(y_true, y_pred_classes, target_names=actions))