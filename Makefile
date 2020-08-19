.data/raw .data/processed:
	mkdir -p $@

.data/raw/competitive-data-science-predict-future-sales.zip: | .data/raw
	pipenv run kaggle competitions download \
		-c competitive-data-science-predict-future-sales \
		-p $(@D)

.data/processed/sales-train-by-month.parquet: .data/raw/competitive-data-science-predict-future-sales.zip | .data/processed
	pipenv run python -m src.data.make_sales_train_by_month $< $@

.data/processed/train-set.parquet: .data/processed/sales-train-by-month.parquet | .data/processed
	pipenv run python -m src.data.make_train_set $^ $@

.data/processed/test-subset.parquet: .data/raw/competitive-data-science-predict-future-sales.zip | .data/processed
	pipenv run python -m src.data.make_test_subset $^ $@

.data/processed/test-ids.parquet: .data/raw/competitive-data-science-predict-future-sales.zip | .data/processed
	pipenv run python -m src.data.make_test_ids $^ $@

.data/processed/item-categories-metadata.parquet: .data/raw/competitive-data-science-predict-future-sales.zip | .data/processed
	pipenv run python -m src.data.make_item_categories_metadata $< $@

.data/processed/shop-sales-train-by-month.parquet: .data/processed/sales-train-by-month.parquet | .data/processed
	pipenv run python -m src.data.make_shop_sales_train_by_month $< $@

.data/processed/shop-item-cat-encoding.parquet: .data/processed/sales-train-by-month.parquet .data/processed/item-categories-metadata.parquet .data/processed/shop-sales-train-by-month.parquet | .data/processed
	pipenv run python -m src.data.make_shop_item_cat_encoding $^ $@

.data/processed/date-ids.parquet: .data/processed/sales-train-by-month.parquet | .data/processed
	pipenv run python -m src.data.make_date_ids $^ $@

.data/processed/base-ids-monthly-freqs.parquet: .data/processed/sales-train-by-month.parquet .data/processed/item-categories-metadata.parquet | .data/processed
	pipenv run python -m src.data.make_base_ids_monthly_freqs $^ $@

.data/processed/item-cnt-lagged.parquet: .data/processed/train-set-base.parquet .data/processed/test-set-base.parquet | .data/processed
	pipenv run python -m src.data.make_item_cnt_lagged $^ $@

.data/processed/train-set-base-cats.parquet: .data/processed/train-set-base.parquet | .data/processed
	pipenv run python -m src.feature_engineering.add_base_cat_features $< $@

.data/processed/test-set-base-cats.parquet: .data/processed/test-set-base.parquet | .data/processed
	pipenv run python -m src.feature_engineering.add_base_cat_features $< $@

.data/processed/train-set-base-cats-item-cnt-lagged.parquet: .data/processed/train-set-base-cats.parquet .data/processed/item-cnt-lagged.parquet | .data/processed
	pipenv run python -m src.feature_engineering.add_item_cnt_lagged $^ $@

.data/processed/test-set-base-cats-item-cnt-lagged.parquet: .data/processed/test-set-base-cats.parquet .data/processed/item-cnt-lagged.parquet | .data/processed
	pipenv run python -m src.feature_engineering.add_item_cnt_lagged $^ $@

.data/processed/train-set-base-cats-item-cnt-lagged-date-ids.parquet: .data/processed/train-set-base-cats-item-cnt-lagged.parquet .data/processed/date-ids.parquet | .data/processed
	pipenv run python -m src.feature_engineering.add_date_ids $^ $@

.data/processed/test-set-base-cats-item-cnt-lagged-date-ids.parquet: .data/processed/test-set-base-cats-item-cnt-lagged.parquet .data/processed/date-ids.parquet | .data/processed
	pipenv run python -m src.feature_engineering.add_date_ids $^ $@


.PHONY: clean
clean:
	rm -rf .data
