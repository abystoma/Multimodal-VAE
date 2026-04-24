import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HARD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(BASE_DIR, "data", "MIR-1K")
AUDIO_DIR = os.path.join(DATA_DIR, "Wavfile")
LYRICS_DIR = os.path.join(DATA_DIR, "Lyrics")

SAVE_DIR = os.path.join(BASE_DIR, "data", "mir1k_hard")

TARGET_SR = 22050
MIDDLE_SECONDS = 8

N_MELS = 128
N_FRAMES = 256

LYRICS_MAX_FEATURES = 1500
LYRICS_EMBED_DIM = 128

LATENT_DIM = 32
BETA = 0.05

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4
TEST_SIZE = 0.2
RANDOM_SEED = 42

TOP_N_SINGERS = 5

OUTPUT_DIR = os.path.join(HARD_DIR, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)