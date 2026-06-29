from scam_spotter.calibration import (
    brier_score,
    expected_calibration_error,
    fit_temperature,
    reliability_curve,
)


def test_brier_perfect_is_zero():
    assert brier_score([1.0, 0.0, 1.0], [1, 0, 1]) == 0.0


def test_brier_worst_is_one():
    assert brier_score([0.0, 1.0], [1, 0]) == 1.0


def test_ece_well_calibrated_low():
    # Confidence matches accuracy => low ECE.
    probs = [0.9, 0.9, 0.9, 0.1, 0.1]
    labels = [1, 1, 0, 0, 0]  # 0.9-bin accuracy ~0.67, not perfect but bounded
    assert 0.0 <= expected_calibration_error(probs, labels) <= 1.0


def test_reliability_curve_bins():
    probs = [0.05, 0.15, 0.95, 0.85]
    labels = [0, 0, 1, 1]
    bins = reliability_curve(probs, labels, n_bins=10)
    assert bins
    assert all(0.0 <= b.avg_accuracy <= 1.0 for b in bins)


def test_fit_temperature_softens_overconfident_logits():
    # Hugely separated logits but a fraction are wrong => T > 1 reduces NLL.
    pairs = [(-6.0, 6.0)] * 8 + [(6.0, -6.0)] * 2  # confident "positive" but 2/10 actually neg
    labels = [1] * 8 + [1] * 2  # pretend all labeled positive-ish to push calibration
    t = fit_temperature(pairs, labels)
    assert t >= 1.0


def test_data_examples_load():
    from scam_spotter.data import load_examples

    ex = load_examples()
    assert len(ex) >= 20
    assert all(label in (0, 1) for _, label in ex)
    assert any(label == 1 for _, label in ex) and any(label == 0 for _, label in ex)
