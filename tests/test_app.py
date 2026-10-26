"""Smoke tests for the Gradio handlers (no browser needed)."""
import app


def test_spam_tab_flags_scam():
    out = app.check_spam(app.SPAM_EXAMPLES[0])
    assert "Spam / Phishing" in out and "Red flags" in out


def test_spam_tab_handles_empty_input():
    assert "⚠️" in app.check_spam("   ")


def test_image_tabs_handle_missing_upload():
    assert "⚠️" in app.verify_watermark(None)
    assert "⚠️" in app.check_ai_image(None)[0]
    assert "⚠️" in app.add_watermark(None, "RVT")[0]
