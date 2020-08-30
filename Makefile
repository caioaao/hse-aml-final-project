.data/raw .data/processed .data/model .data/submissions .data/trials:
	mkdir -p $@

.data/trials/studies.db: | .data/trials
	touch $@

.data/raw/competitive-data-science-predict-future-sales.zip: | .data/raw
	pipenv run kaggle competitions download \
		-c competitive-data-science-predict-future-sales \
		-p $(@D)

.data/processed/sales-train-by-month.parquet: src/data/make_sales_train_by_month.py .data/raw/competitive-data-science-predict-future-sales.zip | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/test-set.parquet: src/data/make_test_set.py .data/raw/competitive-data-science-predict-future-sales.zip | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/train-set.parquet: src/data/make_train_set.py .data/processed/test-set.parquet .data/processed/sales-train-by-month.parquet | .data/processed
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

.data/processed/%-features-001.parquet: src/feature_engineering/make_dataset_001.py .data/processed/%.parquet .data/processed/sales-train-by-month.parquet .data/processed/date-ids.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/%-features-002.parquet: src/feature_engineering/make_dataset_002.py .data/processed/%-features-001.parquet .data/processed/item-categories-metadata.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/%-features-003.parquet: src/feature_engineering/make_dataset_003.py .data/processed/%-features-002.parquet .data/processed/shop-item-cat-encoding.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/model/xgb-features-%.model: src/model/make_tune_xgb.py .data/trials/studies.db .data/processed/train-set-features-%.parquet | .data/model
	pipenv run scripts/runpy.sh $^ $@

.data/submissions/xgb-features-%.csv: src/submission/make_submission.py .data/model/xgb-features-%.model .data/processed/test-set-features-%.parquet | .data/submissions
	pipenv run scripts/runpy.sh $^ $@

.PHONY: clean
clean:
	rm -rf .data
