from .paths import processed_data_path, raw_data_path
from zipfile import ZipFile
import pandas as pd


def raw_zipfile():
    return ZipFile(raw_data_path('competitive-data-science-predict-future-sales.zip'), 'r')


def base_datasets():
    return (pd.read_parquet(processed_data_path("train-set-base.parquet")),
            pd.read_parquet(processed_data_path("test-set-base.parquet")))
