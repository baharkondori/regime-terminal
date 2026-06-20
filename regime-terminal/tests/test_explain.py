"""Unit tests for explain.py"""

from explain import explain_signal, explain_confirmation_breakdown, GLOSSARY, DISCLAIMER_SHORT


def test_explain_signal_returns_nonempty_string():
    text = explain_signal(
        regime="bull", confidence=0.85, conf_count=7, conf_required=7,
        conf_total=8, bullish_now=True, action="🟢 LONG / HOLD",
    )
    assert isinstance(text, str)
    assert len(text) > 0


def test_explain_signal_bearish_mentions_staying_out():
    text = explain_signal(
        regime="bear", confidence=0.999, conf_count=4, conf_required=7,
        conf_total=8, bullish_now=False, action="🔴 EXIT / FLAT",
    )
    assert "stay out" in text or "close any open position" in text


def test_explain_signal_includes_confidence_percentage():
    text = explain_signal(
        regime="chop", confidence=0.55, conf_count=2, conf_required=7,
        conf_total=8, bullish_now=False, action="🔴 EXIT / FLAT",
    )
    assert "55%" in text


def test_explain_signal_handles_unrecognized_regime_gracefully():
    text = explain_signal(
        regime="some_future_label", confidence=0.5, conf_count=0, conf_required=7,
        conf_total=8, bullish_now=False, action="🔴 EXIT / FLAT",
    )
    assert isinstance(text, str)
    assert len(text) > 0  # should not raise, even for an unknown regime name


def test_explain_confirmation_breakdown_marks_true_and_false_distinctly():
    breakdown = {"rsi": True, "momentum": False}
    lines = explain_confirmation_breakdown(breakdown)
    assert len(lines) == 2
    assert "✅" in lines[0]
    assert "❌" in lines[1]


def test_explain_confirmation_breakdown_handles_empty_dict():
    assert explain_confirmation_breakdown({}) == []


def test_glossary_has_no_empty_definitions():
    for term, definition in GLOSSARY.items():
        assert isinstance(term, str) and len(term) > 0
        assert isinstance(definition, str) and len(definition) > 10


def test_disclaimer_short_is_nonempty_and_mentions_not_advice():
    assert len(DISCLAIMER_SHORT) > 0
    assert "not" in DISCLAIMER_SHORT.lower()


def test_explain_stop_levels_includes_concrete_prices():
    from explain import explain_stop_levels
    text = explain_stop_levels(
        last_price=60000.0, stop_loss_price=55200.0, stop_loss_pct=0.08,
        trailing_stop_price=None, trailing_stop_pct=0.04, use_trailing_stop=False,
    )
    assert "60,000.00" in text
    assert "55,200.00" in text
    assert "8%" in text


def test_explain_stop_levels_mentions_manual_setup_caveat():
    """The most important line: this must always warn that automatic
    execution only happens in the simulation, not in a real/demo app,
    since that's the exact gap the user was confused about."""
    from explain import explain_stop_levels
    text = explain_stop_levels(
        last_price=100.0, stop_loss_price=92.0, stop_loss_pct=0.08,
        trailing_stop_price=None, trailing_stop_pct=0.04, use_trailing_stop=False,
    )
    lowered = text.lower()
    assert "manually set" in lowered or "set one as an order" in lowered
    assert "nothing sells on your behalf" in lowered


def test_explain_stop_levels_trailing_stop_off_says_so():
    from explain import explain_stop_levels
    text = explain_stop_levels(
        last_price=100.0, stop_loss_price=92.0, stop_loss_pct=0.08,
        trailing_stop_price=None, trailing_stop_pct=0.04, use_trailing_stop=False,
    )
    assert "turned off" in text.lower()


def test_explain_stop_levels_trailing_stop_on_includes_price():
    from explain import explain_stop_levels
    text = explain_stop_levels(
        last_price=100.0, stop_loss_price=92.0, stop_loss_pct=0.08,
        trailing_stop_price=96.0, trailing_stop_pct=0.04, use_trailing_stop=True,
    )
    assert "trailing stop" in text.lower()
    assert "pulls back" in text.lower()


def test_explain_transition_with_data_mentions_not_a_prediction():
    from explain import explain_transition
    text = explain_transition("strong_bull", [("bear", 0.89), ("bull", 0.11)], 1885)
    assert "not a prediction" in text.lower() or "not a forecast" in text.lower()
    assert "1885" in text


def test_explain_transition_empty_history_handled_gracefully():
    from explain import explain_transition
    text = explain_transition("chop", [], 0)
    assert isinstance(text, str)
    assert len(text) > 0
    assert "not a prediction" not in text.lower()  # different message path for no-data case


def test_explain_transition_never_implies_certainty():
    """Guard against language drift: this explanation must never claim
    certainty about future outcomes, regardless of how lopsided the
    historical distribution is."""
    from explain import explain_transition
    text = explain_transition("crash", [("crash", 0.99)], 500)
    lowered = text.lower()
    forbidden_phrases = ["is guaranteed", "guaranteed to", "certain to", "definitely will", "will definitely"]
    for phrase in forbidden_phrases:
        assert phrase not in lowered
    # must explicitly hedge, not just avoid forbidden phrases
    assert "not a prediction" in lowered or "not a forecast" in lowered
