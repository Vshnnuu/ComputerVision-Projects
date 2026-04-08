from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "signals.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

SEQUENCE_LENGTH = 20
BATCH_SIZE = 32
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3
BIDIRECTIONAL = True

NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
TEST_SIZE = 0.2
RANDOM_SEED = 42

FEATURE_COLUMNS = ["smile", "mouth_open", "head_turn"]
CLASS_NAMES = ["neutral", "smiling", "mouth_open", "head_left", "head_right"]