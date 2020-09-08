.data/raw .data/processed .data/model .data/submissions .data/trials .data/model-outputs reports:
	mkdir -p $@

.data/trials/studies.db: | .data/trials
	touch $@

###############################################################################
# Raw data
###############################################################################

.data/raw/competitive-data-science-predict-future-sales.zip: | .data/raw
	pipenv run kaggle competitions download \
		-c competitive-data-science-predict-future-sales \
		-p $(@D)

###############################################################################
# Canonical datasets
###############################################################################

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

.data/processed/economics-history.parquet: src/data/make_economics_history.py | .data/processed
	pipenv run scripts/runpy.sh $^ $@

###############################################################################
# Feature sets
###############################################################################

.data/processed/%-features-000.parquet: src/feature_engineering/make_feature_set_000.py .data/processed/%.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/%-features-001.parquet: src/feature_engineering/make_feature_set_001.py .data/processed/%.parquet .data/processed/date-ids.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/%-features-002.parquet: src/feature_engineering/make_feature_set_002.py .data/processed/%.parquet .data/processed/sales-train-by-month.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/%-features-003.parquet: src/feature_engineering/make_feature_set_003.py .data/processed/%.parquet .data/processed/item-categories-metadata.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/%-features-004.parquet: src/feature_engineering/make_feature_set_004.py .data/processed/%.parquet .data/processed/sales-train-by-month.parquet .data/processed/item-categories-metadata.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/%-features-005.parquet: src/feature_engineering/make_feature_set_005.py .data/processed/%.parquet .data/processed/economics-history.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/%-features-006.parquet: src/feature_engineering/make_combined_feature_set.py .data/processed/%-features-000.parquet .data/processed/%-features-001.parquet .data/processed/%-features-002.parquet .data/processed/%-features-003.parquet .data/processed/%-features-004.parquet .data/processed/%-features-005.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

###############################################################################
# Models
###############################################################################

.data/model/xgb-features-%.model: src/model/make_xgb.py .data/trials/studies.db .data/processed/train-set-features-%.parquet | .data/model
	pipenv run scripts/runpy.sh $^ $@

.data/model/lgb-features-%.model: src/model/make_lgb.py .data/trials/studies.db .data/processed/train-set-features-%.parquet | .data/model
	pipenv run scripts/runpy.sh $^ $@

###############################################################################
# Submissions
###############################################################################

.data/submissions/xgb-features-%.csv: src/submission/make_submission.py .data/model/xgb-features-%.model .data/processed/test-set-features-%.parquet | .data/submissions
	pipenv run scripts/runpy.sh $^ $@

.data/submissions/lgb-features-%.csv: src/submission/make_submission.py .data/model/lgb-features-%.model .data/processed/test-set-features-%.parquet | .data/submissions
	pipenv run scripts/runpy.sh $^ $@

###############################################################################
# Model outputs
###############################################################################

.data/model-outputs/cv-xgb-features-%.parquet: src/model/make_cv_predict.py .data/model/xgb-features-%.model .data/processed/train-set-features-%.parquet | .data/model-outputs
	pipenv run scripts/runpy.sh $^ $@

###############################################################################
# Reports
###############################################################################

reports/cv-score-%.log: src/model/make_cv_report.py .data/model-outputs/cv-%.parquet | reports
	pipenv run scripts/runpy.sh $^ $@

.PHONY: clean
clean:
	rm -rf .data

.PHONY: clean-derived
clean-derived:
	rm -rf .data/processed .data/model .data/model-outputs .data/submissions .data/trials reports

.PHONY: all-reports
all-reports: reports/cv-score-xgb-features-000.log reports/cv-score-xgb-features-001.log reports/cv-score-xgb-features-002.log reports/cv-score-xgb-features-003.log reports/cv-score-xgb-features-004.log reports/cv-score-xgb-features-005.log reports/cv-score-xgb-features-006.log

.PHONY: clean-models
clean-models:
	rm -rf .data/model .data/model-outputs .data/trials

.SECONDARY:
