"""Central configuration. Everything is overridable via environment variables / .env."""
import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional
    pass

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///reconai.db")
ANTHROPIC_API_KEY = os.getenv("", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")

# Assumed merchant pricing plan (fraction of amount). Edit to match a real plan.
# (UPI MDR is 0% for most merchants in India, hence 0.0.)
FEE_RATES = {"card": 0.02, "netbanking": 0.019, "wallet": 0.02, "upi": 0.0}
GST_RATE = 0.18
SETTLEMENT_SLA_DAYS = 3   # captured payments should be settled within this many days
TOLERANCE_PAISE = 100     # ignore differences below Rs 1
