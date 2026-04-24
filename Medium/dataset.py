import os
import librosa
import numpy as np

from src.config_medium import AUDIO_DIR

lengths = []

for f in os.listdir(AUDIO_DIR):
    if f.endswith(".wav"):
        path = os.path.join(AUDIO_DIR, f)
        y, sr = librosa.load(path, sr=None)
        lengths.append(len(y) / sr)

import numpy as np

lengths = np.array(lengths)  # from your previous code

count = np.sum((lengths < 10) & (lengths > 8))

print("Between 10 and 12 sec:", count)