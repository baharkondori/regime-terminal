"""
explain.py
----------
Translates the raw signal (regime, confidence, confirmation count) into
plain-English sentences for people who aren't familiar with trading
terminology or HMM jargon. This is a pure presentation layer — it doesn't
change any of the underlying logic in strategies.py / backtester.py / etc.,
it just describes what those modules already computed, in plainer words.

Public API:
    explain_signal(regime, confidence, conf_count, conf_required, conf_total,
                    is_bullish, action) -> str
    explain_confirmation_breakdown(breakdown_dict) -> list[str]
    GLOSSARY: dict[str, str]   # term -> plain-English definition
"""

from __future__ import annotations

from typing import Dict, List


GLOSSARY: Dict[str, str] = {
    "regime": (
        "The market's current 'mood,' as detected by the model — things like "
        "a strong uptrend, a downtrend, or sideways chop. The model looks at "
        "patterns in price and volume and assigns the current period to one "
        "of these moods."
    ),
    "regime confidence": (
        "How sure the model is about its mood guess, from 0–100%. High "
        "confidence (e.g. 99%) means the model sees a very clear pattern. "
        "Lower confidence means it's a toss-up between two or more moods."
    ),
    "confirmations": (
        "A checklist of 8 separate technical signals (like RSI, momentum, "
        "moving averages) that either agree or disagree with making a trade "
        "right now. The strategy only considers entering a trade when enough "
        "of these checks line up — by default, 7 out of 8."
    ),
    "suggested action": (
        "What the strategy's rules would do automatically, given the regime "
        "and the confirmation count. This is not a personal recommendation — "
        "it's just what the backtested rule-set says to do."
    ),
    "bull / bear": (
        "'Bull' (and 'weak_bull', 'strong_bull') mean the model thinks prices "
        "are trending up. 'Bear' (and 'weak_bear', 'crash') mean it thinks "
        "they're trending down. 'Chop' means no clear trend either way."
    ),
}


_REGIME_DESCRIPTIONS = {
    "strong_bull": "a strong upward trend",
    "bull": "an upward trend",
    "weak_bull": "a mild upward trend",
    "chop": "no clear trend — sideways, choppy price action",
    "weak_bear": "a mild downward trend",
    "bear": "a downward trend",
    "crash": "a sharp, fast downward move",
}


def _regime_phrase(regime: str) -> str:
    return _REGIME_DESCRIPTIONS.get(regime, f"an unrecognized regime ('{regime}')")


def explain_signal(
    regime: str,
    confidence: float,
    conf_count: int,
    conf_required: int,
    conf_total: int,
    bullish_now: bool,
    action: str,
) -> str:
    """Build a 2-4 sentence plain-English explanation of the current signal.

    Parameters mirror what dashboard.py already computes — this function
    does no calculation of its own, only translation into prose.
    """
    regime_phrase = _regime_phrase(regime)
    confidence_word = (
        "very confident" if confidence >= 0.90 else
        "fairly confident" if confidence >= 0.70 else
        "not very confident" if confidence >= 0.50 else
        "quite unsure"
    )

    sentence_1 = (
        f"The model currently sees **{regime_phrase}**, and it's "
        f"**{confidence_word}** about that ({confidence:.0%})."
    )

    if conf_count >= conf_required:
        sentence_2 = (
            f"Enough of the technical checks agree right now "
            f"({conf_count} of {conf_total}, above the {conf_required} required)."
        )
    else:
        sentence_2 = (
            f"Not enough of the technical checks agree right now "
            f"({conf_count} of {conf_total}, below the {conf_required} required)."
        )

    if not bullish_now:
        sentence_3 = (
            "Because the trend isn't upward, the strategy's rules would stay "
            "out of the market (or close any open position) regardless of the "
            "confirmation count — it only looks for trades during upward trends."
        )
    elif conf_count >= conf_required:
        sentence_3 = (
            "Because the trend is upward AND enough checks agree, the "
            "strategy's rules would consider entering or holding a position."
        )
    else:
        sentence_3 = (
            "The trend is upward, but too few checks agree yet, so the "
            "strategy's rules would wait rather than enter a position."
        )

    return f"{sentence_1} {sentence_2} {sentence_3}"


_CONFIRMATION_LABELS = {
    "rsi": "RSI (overbought/oversold check)",
    "momentum": "Momentum (price speeding up or down)",
    "adx": "ADX (is there a real trend, or just noise?)",
    "macd": "MACD (trend-following signal)",
    "volatility": "Volatility (is price swinging unusually wildly?)",
    "volume_spike": "Volume spike (unusually high trading activity)",
    "price_action": "Price action (breaking out of a recent range)",
    "moving_average": "Moving average (price above/below its recent trend line)",
}


def explain_confirmation_breakdown(breakdown: Dict[str, bool]) -> List[str]:
    """Turn the raw {check_name: bool} dict into a list of plain-English
    bullet strings, e.g. '✅ RSI (overbought/oversold check): agrees'."""
    lines = []
    for key, passed in breakdown.items():
        label = _CONFIRMATION_LABELS.get(key, key)
        mark = "✅" if passed else "❌"
        verdict = "agrees" if passed else "disagrees"
        lines.append(f"{mark} {label}: {verdict}")
    return lines


def explain_stop_levels(
    last_price: float,
    stop_loss_price: float,
    stop_loss_pct: float,
    trailing_stop_price,
    trailing_stop_pct: float,
    use_trailing_stop: bool,
) -> str:
    """Explain, in concrete price terms, what the stop-loss and trailing-stop
    levels mean for a position opened at the current price — and the
    important caveat that these levels are not automatic in most apps unless
    you set them yourself."""
    lines = [
        f"If you bought at **${last_price:,.2f}** right now, the backtester's rules "
        f"would automatically sell if price fell to **${stop_loss_price:,.2f}** "
        f"(a {stop_loss_pct:.0%} loss) \u2014 that's the stop-loss."
    ]

    if use_trailing_stop and trailing_stop_price is not None:
        lines.append(
            f"There's also a trailing stop: if price rises after you buy and then "
            f"pulls back {trailing_stop_pct:.0%} from its highest point since your "
            f"purchase, the rules would sell there too \u2014 locking in whatever gain "
            f"had built up, rather than giving it all back. (Marked with * above "
            f"because this level moves upward as price rises \u2014 it isn't fixed like "
            f"the stop-loss.)"
        )
    else:
        lines.append("Trailing stop is currently turned off, so only the stop-loss above applies.")

    lines.append(
        "**Important:** these levels only execute automatically *inside this "
        "backtester's simulation*. On a real or demo trading app, a stop-loss "
        "usually only protects you if you manually set one as an order on the "
        "exchange itself \u2014 otherwise, if you step away (e.g. in a meeting) and "
        "price moves against you, nothing sells on your behalf. If your app "
        "supports stop-loss or stop-limit orders, setting one at the price shown "
        "above is how you'd replicate this protection in real life."
    )

    return " ".join(lines)


DISCLAIMER_SHORT = (
    "This is what the strategy's rules would automatically do, based only on "
    "past patterns. It is **not personal financial advice**, and is not a "
    "recommendation to buy, sell, or hold anything with real money."
)


def explain_transition(current_regime: str, next_regime_probs: List, n_observations: int) -> str:
    """Plain-English framing for the historical transition table. Deliberately
    avoids any language implying forecasting or prediction — this describes
    what happened after this regime historically, not what will happen next.

    Parameters
    ----------
    current_regime : the current regime name
    next_regime_probs : list of (regime_name, probability) tuples, as returned
                         by regimelabeler.most_likely_next_regimes()
    n_observations : how many times this regime was observed historically
                      (so the user can judge how much history backs this)
    """
    if not next_regime_probs:
        return (
            f"There isn't enough historical data on the **{current_regime}** "
            f"regime in this fitted window to show a meaningful pattern."
        )

    top_regime, top_prob = next_regime_probs[0]
    top_phrase = _regime_phrase(top_regime)

    confidence_note = (
        "a reasonable amount of" if n_observations >= 200 else
        "a modest amount of" if n_observations >= 50 else
        "very little"
    )

    lines = [
        f"Looking back at every time the market was in a **{current_regime}** regime "
        f"({n_observations} times in this dataset), the *next* period most often turned "
        f"out to be **{top_phrase}** ({top_prob:.0%} of the time)."
    ]
    if len(next_regime_probs) > 1:
        rest = ", ".join(f"{_regime_phrase(r)} ({p:.0%})" for r, p in next_regime_probs[1:])
        lines.append(f"Other outcomes seen historically: {rest}.")

    lines.append(
        f"This is based on {confidence_note} historical data, and describes what "
        f"*already happened* in the past \u2014 it is **not a prediction** of what will "
        f"happen this time. The same regime has led to different outcomes before, and "
        f"will likely do so again."
    )
    return " ".join(lines)
