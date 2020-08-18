import pandas as pd
from zipfile import ZipFile


if __name__ == '__main__':
    import sys
    raw_data_path = sys.argv[1]
    output_path = sys.argv[2]

    with ZipFile(raw_data_path, 'r') as f:
        item_categories = pd.read_csv(f.open('item_categories.csv'))
        items = pd.read_csv(f.open('items.csv'))

    item_categories2 = item_categories['item_category_name']\
        .str.split(pat='-', expand=True, n=1)\
        .rename(columns={0: "category_name", 1: "subcategory_name"})
    item_categories2['item_category_id'] = item_categories['item_category_id']

    tmp_filter = item_categories2['subcategory_name'].isna()
    item_categories2.loc[tmp_filter, 'subcategory_name'] = \
        item_categories2.loc[tmp_filter, 'category_name']
    item_categories2.loc[tmp_filter, 'category_name'] = 'Other'

    item_categories_metadata = items[['item_id', 'item_category_id']]
    item_categories_metadata = item_categories_metadata.merge(
        item_categories2, on='item_category_id')

    item_categories_metadata.to_parquet(output_path)
