# Vitality — Detection System

Vitality is a web app that helps people check digital content. It has three detectors:

| Detector | What it tells you |
|---|---|
| **Watermark** | Whether an image carries an authentic, untampered Vitality watermark (signed with HMAC-SHA256) |
| **AI image** | How likely an image is to be AI-generated (metadata, a pretrained model and a frequency check) |
| **Spam** | Whether a message is likely spam, and which words triggered the verdict |

Results are probabilities, not proof.

## Run locally (Windows)

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python app.py
```

Built by Reuben Vitalis as the capstone project for the Wecncode AI in Cybersecurity Bootcamp.