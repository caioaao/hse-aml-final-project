.data/raw .data/processed:
	mkdir -p $@

.data/raw/competitive-data-science-predict-future-sales.zip: | .data/raw
	pipenv run kaggle competitions download \
		-c competitive-data-science-predict-future-sales \
		-p $(@D)

.data/processed/sales-train-by-month.parquet: src/data/make_sales_train_by_month.py .data/raw/competitive-data-science-predict-future-sales.zip | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/train-set.parquet: src/data/make_train_set.py .data/processed/sales-train-by-month.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/test-subset.parquet: src/data/make_test_subset.py .data/raw/competitive-data-science-predict-future-sales.zip | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/test-ids.parquet: src/data/make_test_ids.py .data/raw/competitive-data-science-predict-future-sales.zip | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/item-categories-metadata.parquet: src/data/make_item_categories_metadata.py .data/raw/competitive-data-science-predict-future-sales.zip | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/shop-sales-train-by-month.parquet: src/data/make_shop_sales_train_by_month.py .data/processed/sales-train-by-month.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/shop-item-cat-encoding.parquet: src/data/make_shop_item_cat_encoding.py .data/processed/sales-train-by-month.parquet .data/processed/item-categories-metadata.parquet .data/processed/shop-sales-train-by-month.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/date-ids.parquet: src/data/make_date_ids.py .data/processed/sales-train-by-month.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/base-ids-monthly-freqs.parquet: src/data/make_base_ids_monthly_freqs.py .data/processed/sales-train-by-month.parquet .data/processed/item-categories-metadata.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/train-set-base-cats.parquet: src/feature_engineering/add_base_cat_features.py .data/processed/train-set-base.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/test-set-base-cats.parquet: src/feature_engineering/add_base_cat_features.py .data/processed/test-set-base.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/train-set-base-cats-item-cnt-lagged.parquet: src/feature_engineering/add_item_cnt_lagged.py .data/processed/train-set-base-cats.parquet .data/processed/item-cnt-lagged.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/test-set-base-cats-item-cnt-lagged.parquet: src/feature_engineering/add_item_cnt_lagged.py .data/processed/test-set-base-cats.parquet .data/processed/item-cnt-lagged.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/train-set-base-cats-item-cnt-lagged-date-ids.parquet: src/feature_engineering/add_date_ids.py .data/processed/train-set-base-cats-item-cnt-lagged.parquet .data/processed/date-ids.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/test-set-base-cats-item-cnt-lagged-date-ids.parquet: src/feature_engineering/add_date_ids.py .data/processed/test-set-base-cats-item-cnt-lagged.parquet .data/processed/date-ids.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@


.PHONY: clean
clean:
	rm -rf .data
