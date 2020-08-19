import pandas as pd


def submission_from_subset(subset: pd.DataFrame, full_set: pd.DataFrame):
    return full_set.merge(subset, on=['item_id', 'shop_id'],
                          how='left', sort=False)\
                   .fillna(0)[['ID', 'item_cnt_month']]
