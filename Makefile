.data/01-raw .data/02-processed:
	mkdir -p $@

.data/01-raw/competitive-data-science-predict-future-sales.zip: .data/01-raw
	pipenv run kaggle competitions download \
		-c competitive-data-science-predict-future-sales \
		-p $(@D)

.data/02-processed/sales-train-by-month.parquet: .data/01-raw/competitive-data-science-predict-future-sales.zip .data/02-processed
	pipenv run python -m src.data.make_sales_train_by_month $@ $<

.data/02-processed/test-set-base.parquet: .data/01-raw/competitive-data-science-predict-future-sales.zip .data/02-processed
	pipenv run python -m src.data.preprocess_test_set $< $@

.data/02-processed/train-set-base.parquet: .data/02-processed/sales-train-by-month.parquet .data/02-processed
	pipenv run python -m src.data.preprocess_train_set $< $@

fetch-raw: .data/01-raw/competitive-data-science-predict-future-sales.zip
process-data: .data/02-processed/train-set-base.parquet .data/02-processed/test-set-base.parquet
