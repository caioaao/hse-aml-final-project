import os
import tempfile


DATA_DIR='../.data'

RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(DATA_DIR, 'model')
MODEL_OUTPUTS_DIR = os.path.join(DATA_DIR, 'model-outputs')
TMP_DIR=tempfile.mkdtemp()

# also load them as env variables
os.environ["RAW_DATA_DIR"] = RAW_DATA_DIR
os.environ["PROCESSED_DATA_DIR"] = PROCESSED_DATA_DIR
os.environ["MODELS_DIR"] = MODELS_DIR
os.environ["MODEL_OUTPUTS_DIR"] = MODEL_OUTPUTS_DIR
os.environ["TMP_DIR"] = TMP_DIR

# From notebook 01
RMSE_CONST = 0.4770088818840332