import os


def data_dir():
    return os.environ.get("HSE_DATA_DIR", "./.data")


def raw_data_dir():
    return os.path.join(data_dir(), 'raw')


def processed_data_dir():
    return os.path.join(data_dir(), 'processed')


def models_dir():
    return os.path.join(data_dir(), '03-models')


def model_outputs_dir():
    return os.path.join(data_dir(), '04-model-output')


def raw_data_path(dataset_name):
    return os.path.join(raw_data_dir(), dataset_name)


def processed_data_path(dataset_name):
    return os.path.join(processed_data_dir(), dataset_name)


def model_path(model_name):
    return os.path.join(models_dir(), model_name)


def model_output_path(model_name):
    return os.path.join(model_outputs_dir(), model_name)
