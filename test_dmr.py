"""
DMR の基本テスト
"""
import numpy as np
import pytest
from dmr import DynamicalModeReduction


def make_simple_data(n=100, d=10, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d))


# ── 基本動作 ─────────────────────────────────────────────────────

def test_output_shape():
    X = make_simple_data()
    emb = DynamicalModeReduction(n_components=2).fit_transform(X)
    assert emb.shape == (100, 2)


def test_output_shape_3d():
    X = make_simple_data()
    emb = DynamicalModeReduction(n_components=3).fit_transform(X)
    assert emb.shape == (100, 3)


def test_sklearn_api():
    """fit → embedding_ 属性が生成されること"""
    X = make_simple_data()
    dmr = DynamicalModeReduction(n_components=2)
    dmr.fit(X)
    assert hasattr(dmr, 'embedding_')
    assert dmr.embedding_.shape == (100, 2)


def test_no_nan_in_output():
    X = make_simple_data()
    emb = DynamicalModeReduction().fit_transform(X)
    assert not np.any(np.isnan(emb))
    assert not np.any(np.isinf(emb))


# ── 属性の存在確認 ────────────────────────────────────────────────

def test_attributes_exist():
    X = make_simple_data()
    dmr = DynamicalModeReduction(n_components=2)
    dmr.fit_transform(X)
    assert hasattr(dmr, 'eigenvalues_')
    assert hasattr(dmr, 'frequencies_')
    assert hasattr(dmr, 'masses_')
    assert hasattr(dmr, 'spring_matrix_')


def test_eigenvalues_positive():
    X = make_simple_data()
    dmr = DynamicalModeReduction(n_components=2)
    dmr.fit_transform(X)
    assert np.all(dmr.eigenvalues_ >= 0)


def test_masses_positive():
    X = make_simple_data()
    dmr = DynamicalModeReduction(n_components=2)
    dmr.fit_transform(X)
    assert np.all(dmr.masses_ > 0)


# ── パラメータ変化 ────────────────────────────────────────────────

@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0, 2.0])
def test_alpha_variants(alpha):
    X = make_simple_data()
    emb = DynamicalModeReduction(n_components=2, alpha=alpha).fit_transform(X)
    assert emb.shape == (100, 2)
    assert not np.any(np.isnan(emb))


@pytest.mark.parametrize("mass_power", [0.0, 0.5, 1.0])
def test_mass_power_variants(mass_power):
    X = make_simple_data()
    emb = DynamicalModeReduction(n_components=2, mass_power=mass_power).fit_transform(X)
    assert emb.shape == (100, 2)


def test_fixed_sigma():
    X = make_simple_data()
    emb = DynamicalModeReduction(n_components=2, sigma=1.0).fit_transform(X)
    assert emb.shape == (100, 2)


# ── 解釈性 API ───────────────────────────────────────────────────

def test_interpretability_report():
    X = make_simple_data()
    dmr = DynamicalModeReduction(n_components=3)
    dmr.fit_transform(X)
    report = dmr.interpretability_report()
    assert len(report) == 3
    for r in report:
        assert 'dimension' in r
        assert 'eigenvalue' in r
        assert 'frequency' in r
        assert 'mode_class' in r
        assert r['mode_class'] in ('global', 'meso', 'local')


def test_mass_report():
    X = make_simple_data()
    dmr = DynamicalModeReduction(n_components=2)
    dmr.fit_transform(X)
    mr = dmr.mass_report()
    assert mr['min_mass'] > 0
    assert mr['max_mass'] >= mr['min_mass']
    assert mr['mass_ratio'] >= 1.0


def test_report_raises_before_fit():
    dmr = DynamicalModeReduction()
    with pytest.raises(RuntimeError):
        dmr.interpretability_report()
