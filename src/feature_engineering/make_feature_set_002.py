import pandas as pd

from sklearn.preprocessing import LabelEncoder

from . import add_as_cat_features

if __name__ == '__main__':
    import sys
    input_df = pd.read_parquet(sys.argv[1])
    cats_metadata = pd.read_parquet(sys.argv[2])
    output_path = sys.argv[3]

    le_cat_name = LabelEncoder().fit(cats_metadata['category_name'])
    le_subcat_name = LabelEncoder().fit(cats_metadata['subcategory_name'])

    df = input_df.merge(cats_metadata, on='item_id')

    df['le_category_name'] = le_cat_name.transform(df['category_name'])
    df['le_subcategory_name'] = le_subcat_name.transform(
        df['subcategory_name'])

    add_as_cat_features(df, ['le_category_name', 'le_subcategory_name'],
                        inplace=True)
    df.drop(columns=['le_category_name', 'le_subcategory_name'],
            inplace=True)

    print("%s columns: %s" (output_path, str(df.columns)))
    df.to_parquet(output_path)
