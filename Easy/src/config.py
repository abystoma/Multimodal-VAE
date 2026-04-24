import os

# ===== Base Path (project root) =====
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ===== Data Paths =====
DATA_DIR = os.path.join(BASE_DIR, "data", "MIR-1K")
AUDIO_DIR = os.path.join(DATA_DIR, "Wavfile")
LYRICS_DIR = os.path.join(DATA_DIR, "Lyrics")

SAVE_DIR = os.path.join(BASE_DIR, "data", "mir1k_easy")

# ===== Audio Processing =====
TARGET_SR = 22050
CLIP_SECONDS = 30
MIDDLE_SECONDS = 10

# ===== Feature Extraction =====
N_MFCC = 40

# ===== Model =====
LATENT_DIM = 8
BATCH_SIZE = 32
EPOCHS = 80
LEARNING_RATE = 1e-3
TEST_SIZE = 0.2
RANDOM_SEED = 42

# ===== Clustering =====
N_CLUSTERS = 2  # since your data is EN vs non-EN

# ===== Output =====
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")
PLOT_DIR = os.path.join(BASE_DIR, "outputs", "plots")
RESULT_DIR = os.path.join(BASE_DIR, "outputs", "results")