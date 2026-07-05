"""
config.py
---------
Project-wide constants and path settings.
If running on Colab, replace DATA_DIR with your own Drive folder.
"""

import os

# Google Drive (or local) data folder.
# On Colab:  '/content/drive/MyDrive/UNSW_NB15'
# Locally:   a relative path like './data' also works.
DATA_DIR = os.environ.get("UNSW_DATA_DIR", "/content/drive/MyDrive/UNSW_NB15")

RAW_FEATURE_FILE = "NUSW-NB15_features.csv"
RAW_DATA_FILES = [
    "UNSW-NB15_1.csv",
    "UNSW-NB15_2.csv",
    "UNSW-NB15_3.csv",
    "UNSW-NB15_4.csv",
]

# Checkpoint file names (feature_selection.py writes/reads these)
X_TRAIN_TOP20_FILE = "X_train_top20.parquet"
X_TEST_TOP20_FILE = "X_test_top20.parquet"
Y_TRAIN_FILE = "y_train.csv"
Y_TEST_FILE = "y_test.csv"

RANDOM_STATE = 42

# Columns dropped from the pipeline because they cause data leakage
LEAKAGE_COLS_EARLY = ["srcip", "dstip", "stcpb", "dtcpb"]
LEAKAGE_COLS_MODEL = ["stime", "ltime", "sport", "dsport"]

# Number of most-frequent categories kept during encoding
TOP_PROTO_N = 10
TOP_STATE_N = 8

# Number of features selected via Mutual Information
N_TOP_FEATURES = 20
MI_SAMPLE_SIZE = 50_000

# Train/test split ratio (chronological, based on stime)
TEST_SIZE = 0.2
