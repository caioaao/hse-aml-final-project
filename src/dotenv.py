import os


def data_dir():
    return os.environ.get("HSE_DATA_DIR", "./.data")


def raw_data_dir():
    return os.path.join(data_dir(), '01-raw')


def processed_data_dir():
    return os.path.join(data_dir(), '02-processed')


def models_dir():
    return os.path.join(data_dir(), '03-models')


def model_outputs_dir():
    return os.path.join(data_dir(), '04-model-output')
