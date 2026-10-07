import numpy as np

from fraud_detection.features.engineering import add_derived_features, build_preprocessor, get_feature_columns


def test_add_derived_features_adds_expected_columns(small_transactions_df):
    engineered = add_derived_features(small_transactions_df)
    for col in ["amount_to_avg_ratio", "hour_sin", "hour_cos"]:
        assert col in engineered.columns


def test_hour_cyclical_encoding_is_bounded(small_transactions_df):
    engineered = add_derived_features(small_transactions_df)
    assert engineered["hour_sin"].between(-1, 1).all()
    assert engineered["hour_cos"].between(-1, 1).all()


def test_amount_to_avg_ratio_is_nonnegative_and_capped(small_transactions_df):
    engineered = add_derived_features(small_transactions_df)
    assert (engineered["amount_to_avg_ratio"] >= 0).all()
    assert (engineered["amount_to_avg_ratio"] <= 100).all()


def test_preprocessor_transforms_to_dense_numeric_array(small_transactions_df):
    engineered = add_derived_features(small_transactions_df)
    feature_columns = get_feature_columns()
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(engineered[feature_columns])
    assert transformed.shape[0] == len(engineered)
    assert np.isfinite(transformed).all()


def test_preprocessor_handles_unseen_category_at_inference(small_transactions_df):
    engineered = add_derived_features(small_transactions_df)
    feature_columns = get_feature_columns()
    preprocessor = build_preprocessor()
    train_part = engineered.iloc[:1000]
    test_row = engineered.iloc[[1000]].copy()
    test_row["merchant_category"] = "unseen_category_xyz"

    preprocessor.fit(train_part[feature_columns])
    # Should not raise thanks to handle_unknown="ignore".
    transformed = preprocessor.transform(test_row[feature_columns])
    assert transformed.shape[0] == 1
