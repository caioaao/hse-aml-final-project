import os

DATA_DIR='../.data'

RAW_DATA_DIR = os.path.join(DATA_DIR, '01-raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, '02-processed')
MODELS_DIR = os.path.join(DATA_DIR, '03-models')
MODEL_OUTPUTS_DIR = os.path.join(DATA_DIR, '04-model-output')

# also load them as env variables
os.environ["RAW_DATA_DIR"] = RAW_DATA_DIR
os.environ["PROCESSED_DATA_DIR"] = PROCESSED_DATA_DIR
os.environ["MODELS_DIR"] = MODELS_DIR
os.environ["MODEL_OUTPUTS_DIR"] = MODEL_OUTPUTS_DIR