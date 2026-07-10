from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from demo.backend.policy import recommend_k


def test_awgn_policy_increases_with_snr():
    ks = [
        recommend_k(channel="awgn", snr_db=snr, max_transmitted_bits=20480).k
        for snr in [-5, 0, 2, 5, 10]
    ]
    assert ks == sorted(ks)
    assert ks[0] == 1
    assert ks[-1] == 4


def test_rayleigh_is_more_conservative_than_awgn():
    awgn = recommend_k(channel="awgn", snr_db=5, max_transmitted_bits=20480)
    rayleigh = recommend_k(channel="rayleigh", snr_db=5, max_transmitted_bits=20480)
    assert rayleigh.k < awgn.k


def test_budget_caps_selected_prefix():
    decision = recommend_k(
        channel="none",
        snr_db=20,
        protection="ldpc",
        max_transmitted_bits=10240,
    )
    assert decision.k == 2
    assert decision.source_bits == 5120
    assert decision.transmitted_bits == 10240
    assert decision.budget_limited


def test_alert_priority_always_uses_l0():
    decision = recommend_k(
        channel="none",
        snr_db=20,
        priority="alert",
        max_transmitted_bits=20480,
    )
    assert decision.k == 1


def test_hysteresis_prevents_boundary_flapping():
    held = recommend_k(
        channel="awgn",
        snr_db=4.1,
        previous_k=2,
        max_transmitted_bits=20480,
    )
    promoted = recommend_k(
        channel="awgn",
        snr_db=4.7,
        previous_k=2,
        max_transmitted_bits=20480,
    )
    assert held.k == 2
    assert held.stabilized
    assert promoted.k == 3
