from nemoguardrails import RailsConfig, LLMRails

from colang_defs import (
    YAML_BASE,
    YAML_WITH_INPUT_RAILS,
    YAML_WITH_OUTPUT_RAILS,
    COLANG_TOPIC_GUARD,
    COLANG_JAILBREAK,
    COLANG_SENSITIVE,
    COLANG_DIALOG,
    COLANG_ACTIONS,
    COLANG_OUTPUT_RAIL,
    COLANG_EXP5_FULL,
)
from actions import detect_pii_in_input, classify_urgency, sanitize_output

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

_YAML_MAP = {
    2: YAML_BASE,
    3: YAML_BASE,
    4: YAML_BASE,
    5: YAML_BASE,
    6: YAML_WITH_INPUT_RAILS,
    7: YAML_WITH_OUTPUT_RAILS,
}

_ACTION_MAP = {
    6: [detect_pii_in_input, classify_urgency],
    7: [sanitize_output],
}


def get_rails_config(exp_num: int) -> RailsConfig:
    """Return the RailsConfig for an experiment (no async state — safe to cache)."""
    return RailsConfig.from_content(
        colang_content=_COLANG_MAP[exp_num],
        yaml_content=_YAML_MAP[exp_num],
    )


def build_rails(exp_num: int, guard_llm) -> LLMRails:
    """Build and return a fully configured LLMRails instance for the given experiment."""
    config = get_rails_config(exp_num)
    rails  = LLMRails(config, llm=guard_llm)

    for action_fn in _ACTION_MAP.get(exp_num, []):
        rails.register_action(action_fn)

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
