from nemoguardrails import RailsConfig, LLMRails

from colang_defs import (
    COLANG_TOPIC_GUARD,
    COLANG_JAILBREAK,
    COLANG_SENSITIVE,
    COLANG_DIALOG,
    COLANG_ACTIONS,
    COLANG_OUTPUT_RAIL,
    COLANG_EXP5_FULL,
)
from actions import detect_pii_in_input, classify_urgency, sanitize_output

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# ─────────────────────────────────────────────────────────────
# Colang strings per experiment (cumulative)
# ─────────────────────────────────────────────────────────────

_COLANG_MAP = {
    2: COLANG_TOPIC_GUARD,
    3: COLANG_TOPIC_GUARD + COLANG_JAILBREAK,
    4: COLANG_TOPIC_GUARD + COLANG_JAILBREAK + COLANG_SENSITIVE,
    5: COLANG_EXP5_FULL,
    6: COLANG_EXP5_FULL + COLANG_ACTIONS,
    7: COLANG_EXP5_FULL + COLANG_OUTPUT_RAIL,
}

_ACTION_MAP = {
    6: [detect_pii_in_input, classify_urgency],
    7: [sanitize_output],
}


def _build_yaml(exp_num: int, model: str, api_key: str) -> str:
    """Build YAML config with Groq's OpenAI-compatible endpoint baked in."""
    rails_section = ""
    if exp_num == 6:
        rails_section = """
rails:
  input:
    flows:
      - check input for pii
      - detect urgency
"""
    elif exp_num == 7:
        rails_section = """
rails:
  output:
    flows:
      - sanitize bot response
"""
    return f"""
models:
  - type: main
    engine: openai
    model: {model}
    parameters:
      base_url: {_GROQ_BASE_URL}
      api_key: {api_key}

instructions:
  - type: general
    content: |
      You are an Enterprise IT Assistant specialising in Kubernetes,
      Intel hardware, and enterprise networking.
      Only answer questions about these topics. Be professional and concise.
{rails_section}"""


def get_rails_config(exp_num: int, model: str, api_key: str) -> RailsConfig:
    """Return the RailsConfig for an experiment (no async state — safe to cache)."""
    return RailsConfig.from_content(
        colang_content=_COLANG_MAP[exp_num],
        yaml_content=_build_yaml(exp_num, model, api_key),
    )


def register_actions(rails: LLMRails, exp_num: int) -> None:
    """Register any custom Python actions required for the given experiment."""
    for action_fn in _ACTION_MAP.get(exp_num, []):
        rails.register_action(action_fn)


def build_rails(exp_num: int, model: str, api_key: str) -> LLMRails:
    """Build and return a fully configured LLMRails instance for the given experiment."""
    config = get_rails_config(exp_num, model, api_key)
    rails  = LLMRails(config)
    register_actions(rails, exp_num)
    return rails


# ─────────────────────────────────────────────────────────────
# New-concept Colang snippet per experiment (for the code viewer)
# Shows only what's NEW in each experiment, not the full stack
# ─────────────────────────────────────────────────────────────

COLANG_SNIPPETS = {
    1: "(No Colang — this is a raw LLM call with no NeMo rails)",
    2: COLANG_TOPIC_GUARD,
    3: COLANG_JAILBREAK,
    4: COLANG_SENSITIVE,
    5: COLANG_DIALOG,
    6: COLANG_ACTIONS,
    7: COLANG_OUTPUT_RAIL,
}
