"""Tests for the Spam & Phishing detector. Run: python -m pytest -q"""
import pytest

from detectors.spam import predict, red_flags


def test_obvious_scam_is_flagged():
    r = predict("Congratulations! You have won KES 50,000. Send your PIN within 10 minutes to claim your prize.")
    assert r["verdict"] == "Spam / Phishing"
    assert r["spam_probability"] > 0.5
    assert {"Urgency", "Manipulation"} <= set(r["red_flags"])


def test_normal_message_is_safe():
    r = predict("The team meeting has moved to 3pm tomorrow in the main hall.")
    assert r["verdict"] == "Looks safe"
    assert r["top_words"] == []


def test_red_flags_detects_swahili():
    assert "Manipulation" in red_flags("Umeshinda zawadi! Tuma PIN yako sasa.")


def test_empty_message_rejected():
    with pytest.raises(ValueError):
        predict("   ")
