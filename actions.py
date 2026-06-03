import re
from typing import Optional
from nemoguardrails.actions import action


@action(is_system_action=True)
async def detect_pii_in_input(context: Optional[dict] = None):
    """Returns list of PII type names found, or empty list (falsy) if clean."""
    user_message = context.get("user_message", "") if context else ""

    patterns = {
        "email":       r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "phone":       r"\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b",
        "ssn":         r"\b\d{3}-\d{2}-\d{4}\b",
        "api_key":     r"(api[_\s-]?key|token|secret)[:\s]+[A-Za-z0-9_\-]{10,}",
        "credit_card": r"\b\d{4}[\s-]\d{4}[\s-]\d{4}[\s-]\d{4}\b",
    }
    found = [ptype for ptype, pat in patterns.items()
             if re.search(pat, user_message, re.IGNORECASE)]
    return found


@action(is_system_action=True)
async def classify_urgency(context: Optional[dict] = None):
    """Returns True if the message signals a production emergency."""
    msg = (context.get("user_message", "") if context else "").lower()
    urgent_keywords = [
        "outage", "down", "crash", "critical",
        "emergency", "not working", "urgent", "p0", "p1",
    ]
    return any(kw in msg for kw in urgent_keywords)


@action(is_system_action=True)
async def sanitize_output(context: Optional[dict] = None):
    """Intercepts bot responses containing hardcoded credentials or exploit techniques."""
    # Different NeMo versions use different context keys for the bot's response.
    bot_message = ""
    if context:
        bot_message = (
            context.get("bot_message")
            or context.get("response")
            or context.get("last_bot_message")
            or ""
        )

    sensitive_output_patterns = {
        "hardcoded_credential": r"(?i)(password|passwd|secret|api[_\-]?key|token)\s*[:=]\s*['\"]?\w{4,}",
        "private_key":          r"-----BEGIN.{0,20}PRIVATE KEY-----",
        "exploit_technique":    r"(?i)\b(reverse.?shell|bind.?shell|shellcode|meterpreter)\b",
    }
    found = [ptype for ptype, pat in sensitive_output_patterns.items()
             if re.search(pat, bot_message)]
    return found
