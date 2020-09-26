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

.data/processed/sales-train.parquet: src/data/clean_sales_train.py .data/raw/competitive-data-science-predict-future-sales.zip | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/sales-train-by-month.parquet: src/data/make_sales_train_by_month.py .data/processed/sales-train.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/test-set.parquet: src/data/make_test_set.py .data/raw/competitive-data-science-predict-future-sales.zip | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/train-set.parquet: src/data/make_train_set_v2.py .data/processed/test-set.parquet .data/processed/sales-train-by-month.parquet | .data/processed
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

.data/processed/prices-statistics.parquet: src/data/make_prices_statistics.py .data/processed/sales-train.parquet .data/processed/item-categories-metadata.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/price-and-sales-ranks.parquet: src/data/make_ranks_df.py .data/processed/sales-train.parquet .data/processed/item-categories-metadata.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/sales-deltas.parquet: src/data/make_sales_deltas.py .data/processed/sales-train.parquet .data/processed/item-categories-metadata.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/prices-deltas.parquet: src/data/make_prices_deltas.py .data/processed/prices-statistics.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/revenues-history.parquet: src/data/make_revenue_features.py .data/processed/sales-train.parquet .data/processed/item-categories-metadata.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/release-dates.parquet: src/data/make_release_dates.py .data/processed/sales-train.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

###############################################################################
# Feature sets
###############################################################################

.data/processed/%-features-000.parquet: src/data/make_feature_set_000.py .data/processed/%.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/%-features-001.parquet: src/data/make_feature_set_001.py .data/processed/%.parquet .data/processed/date-ids.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/%-features-002.parquet: src/data/make_feature_set_002.py .data/processed/%.parquet .data/processed/sales-train-by-month.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/%-features-003.parquet: src/data/make_feature_set_003.py .data/processed/%.parquet .data/processed/item-categories-metadata.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/%-features-004.parquet: src/data/make_feature_set_004.py .data/processed/%.parquet .data/processed/sales-train-by-month.parquet .data/processed/item-categories-metadata.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/%-features-005.parquet: src/data/make_feature_set_005.py .data/processed/%.parquet .data/processed/economics-history.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/%-features-006.parquet: src/data/make_combined_feature_set.py .data/processed/%-features-000.parquet .data/processed/%-features-001.parquet .data/processed/%-features-002.parquet .data/processed/%-features-003.parquet .data/processed/%-features-004.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/%-features-007.parquet: src/data/make_feature_set_007.py .data/processed/%-features-002.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/%-features-008.parquet: src/data/make_combined_feature_set.py .data/processed/%-features-000.parquet .data/processed/%-features-001.parquet .data/processed/%-features-003.parquet .data/processed/%-features-004.parquet .data/processed/%-features-007.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/%-features-009.parquet: src/data/make_feature_set_009.py .data/processed/%.parquet  .data/processed/item-categories-metadata.parquet .data/processed/prices-statistics.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/%-features-010.parquet: src/data/make_feature_set_010.py .data/processed/%.parquet  .data/processed/price-and-sales-ranks.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/%-features-013.parquet: src/data/make_combined_feature_set.py .data/processed/%-features-008.parquet .data/processed/%-features-009.parquet .data/processed/%-features-010.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/%-features-014.parquet: src/data/make_combined_feature_set.py .data/processed/%-features-008.parquet .data/processed/%-features-009.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/%-features-015.parquet: src/data/make_combined_feature_set.py .data/processed/%-features-008.parquet .data/processed/%-features-010.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/%-features-016.parquet: src/data/make_feature_set_016.py .data/processed/%.parquet  .data/processed/sales-deltas.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/%-features-017.parquet: src/data/make_combined_feature_set.py .data/processed/%-features-013.parquet .data/processed/%-features-016.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/%-features-018.parquet: src/data/make_feature_set_018.py .data/processed/%.parquet  .data/processed/prices-deltas.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/%-features-019.parquet: src/data/make_combined_feature_set.py .data/processed/%-features-017.parquet .data/processed/%-features-018.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/%-features-020.parquet: src/data/make_feature_set_020.py .data/processed/%.parquet .data/processed/revenues-history.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/%-features-021.parquet: src/data/make_combined_feature_set.py .data/processed/%-features-019.parquet .data/processed/%-features-020.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/%-features-022.parquet: src/data/make_feature_set_022.py .data/processed/%.parquet .data/processed/release-dates.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/%-features-023.parquet: src/data/make_combined_feature_set.py .data/processed/%-features-021.parquet .data/processed/%-features-022.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/%-features-024.parquet: src/data/make_feature_set_024.py .data/processed/%.parquet .data/processed/sales-train.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/%-features-025.parquet: src/data/make_combined_feature_set.py .data/processed/%-features-023.parquet .data/processed/%-features-024.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

###############################################################################
# Models
###############################################################################

.data/model/xgb-baseline.model: src/model/make_xgb_baseline.py | .data/model
	pipenv run scripts/runpy.sh $^ $@

.data/model/xgb-baseline-features-%.model: src/model/make_xgb_baseline.py .data/processed/train-set-features-%.parquet | .data/model
	pipenv run scripts/runpy.sh $^ $@

.data/model/xgb-features-%.model: src/model/make_xgb.py .data/processed/train-set-features-%.parquet | .data/model .data/trials/studies.db
	pipenv run scripts/runpy.sh $^ .data/trials/studies.db $@

.data/model/xgb-small-features-%.model: src/model/make_xgb_small.py .data/processed/train-set-features-%.parquet | .data/model .data/trials/studies.db
	pipenv run scripts/runpy.sh $^ .data/trials/studies.db $@

.data/model/lgb-features-%.model: src/model/make_lgb.py .data/processed/train-set-features-%.parquet | .data/model .data/trials/studies.db
	pipenv run scripts/runpy.sh $^ .data/trials/studies.db $@

.data/model/linear-model-preprocessor-features-%.model: src/model/make_linear_model_preprocessor.py .data/processed/train-set-features-%.parquet .data/processed/test-set-features-%.parquet | .data/model
	pipenv run scripts/runpy.sh $^ $@

.data/model/sgd-features-%.model: src/model/make_sgd.py .data/processed/train-set-features-%.parquet .data/model/linear-model-preprocessor-features-%.model | .data/model .data/trials/studies.db
	pipenv run scripts/runpy.sh $^ .data/trials/studies.db $@

.data/model/mlp-features-%.model: src/model/make_mlp.py .data/processed/train-set-features-%.parquet .data/model/linear-model-preprocessor-features-%.model | .data/model
	pipenv run scripts/runpy.sh $^ $@

.data/model/rf-features-%.model: src/model/make_rf.py .data/processed/train-set-features-%.parquet | .data/model .data/trials/studies.db
	pipenv run scripts/runpy.sh $^ .data/trials/studies.db $@

###############################################################################
# Submissions
###############################################################################

.data/submissions/%.csv: src/submission/make_submission.py .data/model-outputs/preds-%.parquet | .data/submissions
	pipenv run scripts/runpy.sh $^ $@

.PHONY: all-submissions
all-submissions: .data/submissions/xgb-features-025.csv .data/submissions/sgd-features-025.csv .data/submissions/lgb-features-025.csv

###############################################################################
# Model outputs
###############################################################################

.data/model-outputs/cv-xgb-baseline-features-%.parquet: src/model/make_cv_predict.py .data/model/xgb-baseline.model .data/processed/train-set-features-%.parquet | .data/model-outputs
	pipenv run scripts/runpy.sh $^ $@

.data/model-outputs/cv-xgb-features-%.parquet: src/model/make_cv_predict.py .data/model/xgb-features-%.model .data/processed/train-set-features-%.parquet | .data/model-outputs
	pipenv run scripts/runpy.sh $^ $@

.data/model-outputs/cv-lgb-features-%.parquet: src/model/make_cv_predict.py .data/model/lgb-features-%.model .data/processed/train-set-features-%.parquet | .data/model-outputs
	pipenv run scripts/runpy.sh $^ $@

.data/model-outputs/cv-sgd-features-%.parquet: src/model/make_cv_predict.py .data/model/sgd-features-%.model .data/processed/train-set-features-%.parquet | .data/model-outputs
	pipenv run scripts/runpy.sh $^ $@

.data/model-outputs/cv-mlp-features-%.parquet: src/model/make_cv_predict.py .data/model/mlp-features-%.model .data/processed/train-set-features-%.parquet | .data/model-outputs
	pipenv run scripts/runpy.sh $^ $@

.data/model-outputs/preds-xgb-features-%.parquet: src/model/make_preds.py .data/model/xgb-features-%.model .data/processed/test-set-features-%.parquet | .data/model-outputs
	pipenv run scripts/runpy.sh $^ $@

.data/model-outputs/preds-xgb-small-features-%.parquet: src/model/make_preds.py .data/model/xgb-small-features-%.model .data/processed/test-set-features-%.parquet | .data/model-outputs
	pipenv run scripts/runpy.sh $^ $@

.data/model-outputs/preds-xgb-baseline-features-%.parquet: src/model/make_preds.py .data/model/xgb-baseline-features-%.model .data/processed/test-set-features-%.parquet | .data/model-outputs
	pipenv run scripts/runpy.sh $^ $@

.data/model-outputs/preds-lgb-features-%.parquet: src/model/make_preds.py .data/model/lgb-features-%.model .data/processed/test-set-features-%.parquet | .data/model-outputs
	pipenv run scripts/runpy.sh $^ $@

.data/model-outputs/preds-sgd-features-%.parquet: src/model/make_preds.py .data/model/sgd-features-%.model .data/processed/test-set-features-%.parquet | .data/model-outputs
	pipenv run scripts/runpy.sh $^ $@

###############################################################################
# Stacking
###############################################################################

.data/processed/train-set-features-layer2-%.parquet: src/model/make_stacked_features.py .data/model-outputs/cv-xgb-features-%.parquet .data/model-outputs/cv-lgb-features-%.parquet .data/model-outputs/cv-mlp-features-%.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

.data/processed/test-set-features-layer2-%.parquet: src/model/make_stacked_test_features.py .data/model-outputs/preds-xgb-features-%.parquet .data/model-outputs/preds-lgb-features-%.parquet .data/model-outputs/preds-mlp-features-%.parquet | .data/processed
	pipenv run scripts/runpy.sh $^ $@

###############################################################################
# Reports
###############################################################################

reports/cv-score-%.log: src/model/make_cv_report.py .data/model-outputs/cv-%.parquet | reports
	pipenv run scripts/runpy.sh $^ $@

.PHONY: features-baseline-reports
features-baseline-reports: reports/cv-score-xgb-baseline-features-000.log reports/cv-score-xgb-baseline-features-001.log reports/cv-score-xgb-baseline-features-002.log reports/cv-score-xgb-baseline-features-003.log reports/cv-score-xgb-baseline-features-004.log reports/cv-score-xgb-baseline-features-005.log reports/cv-score-xgb-baseline-features-006.log reports/cv-score-xgb-baseline-features-007.log reports/cv-score-xgb-baseline-features-008.log reports/cv-score-xgb-baseline-features-009.log reports/cv-score-xgb-baseline-features-010.log reports/cv-score-xgb-baseline-features-013.log reports/cv-score-xgb-baseline-features-014.log reports/cv-score-xgb-baseline-features-015.log reports/cv-score-xgb-baseline-features-016.log reports/cv-score-xgb-baseline-features-017.log reports/cv-score-xgb-baseline-features-018.log reports/cv-score-xgb-baseline-features-019.log reports/cv-score-xgb-baseline-features-020.log reports/cv-score-xgb-baseline-features-021.log reports/cv-score-xgb-baseline-features-022.log reports/cv-score-xgb-baseline-features-023.log reports/cv-score-xgb-baseline-features-024.log reports/cv-score-xgb-baseline-features-025.log

.PHONY: all-reports
all-reports: features-baseline-reports reports/cv-score-xgb-features-025.log reports/cv-score-sgd-features-025.log reports/cv-score-lgb-features-025.log

###############################################################################
# Other commands and shortcuts
###############################################################################

.PHONY: clean
clean:
	rm -rf .data

.PHONY: clean-derived
clean-derived:
	rm -rf .data/processed .data/model .data/model-outputs .data/submissions .data/trials reports

.PHONY: clean-models
clean-models:
	rm -rf .data/model .data/model-outputs .data/trials

.PHONY: all
all: all-reports all-submissions

.SECONDARY:
