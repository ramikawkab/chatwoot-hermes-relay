import os
import sys
from pathlib import Path

# app.py reads its settings at import time, so they must exist before the first import.
os.environ.update(
    CHATWOOT_URL="https://chat.example.com",
    CHATWOOT_BOT_TOKEN="bot-token",
    CHATWOOT_WEBHOOK_SECRET="hook-secret",
    HERMES_API_URL="http://hermes:8642",
    HERMES_API_KEY="hermes-key",
    OPENAI_API_KEY="openai-key",
    BATCH_SECONDS="0.05",
)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
