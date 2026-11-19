import numpy as np
import pandas as pd
from backend.features.engine import build_features, FEATURE_COLUMNS, causal_join, fractional_difference
from backend.features.labels import triple_barrier, walk_forward_splits


def test_future_price_perturbation_does_not_change_past_features(candles):
    original = build_features(candles)
    edited = candles.copy()
    edited.loc[300:, ["open", "high", "low", "close"]] *= 8
    edited.loc[300:, "volume"] *= 50
    changed = build_features(edited)
    pd.testing.assert_frame_equal(original[FEATURE_COLUMNS].iloc[:300], changed[FEATURE_COLUMNS].iloc[:300])


def test_features_reset_after_gap(candles):
    d = build_features(candles.drop(index=250))
    after = d[d.open_time > candles.open_time.iloc[250]]
    assert pd.isna(after.momentum.iloc[0])
    assert after.iloc[:47].momentum.isna().all()


def test_asof_join_cannot_see_future():
    left = pd.DataFrame({"available_at": [10, 20, 30]})
    right = pd.DataFrame({"available_at": [15, 25], "score": [1., 2.]})
    merged = causal_join(left, right, ["score"], "news_")
    assert pd.isna(merged.news_score.iloc[0])
    assert merged.news_score.tolist()[1:] == [1., 2.]


def test_labels_have_future_end_times_and_purged_folds(candles):
    features = build_features(candles)
    labels = triple_barrier(features)
    d = pd.concat([features, labels], axis=1).dropna(subset=FEATURE_COLUMNS+["label"]).reset_index(drop=True)
    assert (d.label_end > d.available_at).all()
    assert labels.label.tail(4).isna().all()
    for train, test in walk_forward_splits(d, 150, 50, 4):
        assert train.max() < test.min()-4
        assert d.label_end.iloc[train].max() < d.available_at.iloc[test].min()


def test_fractional_difference_is_causal():
    x = pd.Series(np.arange(100, dtype=float))
    a = fractional_difference(x)
    x.iloc[70:] *= 100
    b = fractional_difference(x)
    pd.testing.assert_series_equal(a.iloc[:70], b.iloc[:70])

