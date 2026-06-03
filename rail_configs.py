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

# engine/model are placeholders — overridden by llm= passed to LLMRails
_YAML_BASE = """
models:
  - type: main
    engine: openai
    model: gpt-3.5-turbo

instructions:
  - type: general
    content: |
      You are an Enterprise IT Assistant specialising in Kubernetes,
      Intel hardware, and enterprise networking.
      Only answer questions about these topics. Be professional and concise.
"""

# NeMo 0.9.x+ requires ALL active flows to be listed under rails.input.flows.
# Without this, intent-based Colang flows are defined but never invoked.
_YAML_EXP2 = _YAML_BASE + """
rails:
  input:
    flows:
      - handle off topic
"""

_YAML_EXP3 = _YAML_BASE + """
rails:
  input:
    flows:
      - handle off topic
      - jailbreak protection
"""

_YAML_EXP4 = _YAML_BASE + """
rails:
  input:
    flows:
      - handle off topic
      - jailbreak protection
      - sensitive topic protection
"""

_YAML_EXP5 = _YAML_BASE + """
rails:
  input:
    flows:
      - greeting
      - capabilities
      - farewell
      - handle off topic
      - jailbreak protection
      - sensitive topic protection
"""

_YAML_INPUT_RAILS = _YAML_BASE + """
rails:
  input:
    flows:
      - check input for pii
      - detect urgency
      - greeting
      - capabilities
      - farewell
      - handle off topic
      - jailbreak protection
      - sensitive topic protection
"""

_YAML_OUTPUT_RAILS = _YAML_BASE + """
rails:
  output:
    flows:
      - sanitize bot response
"""

_YAML_MAP = {
    2: _YAML_EXP2,
    3: _YAML_EXP3,
    4: _YAML_EXP4,
    5: _YAML_EXP5,
    6: _YAML_INPUT_RAILS,
    7: _YAML_OUTPUT_RAILS,
}


def get_rails_config(exp_num: int) -> RailsConfig:
    return RailsConfig.from_content(
        colang_content=_COLANG_MAP[exp_num],
        yaml_content=_YAML_MAP[exp_num],
    )


def register_actions(rails: LLMRails, exp_num: int) -> None:
    for action_fn in _ACTION_MAP.get(exp_num, []):
        rails.register_action(action_fn)


def build_rails(exp_num: int, llm) -> LLMRails:
    config = get_rails_config(exp_num)
    rails  = LLMRails(config, llm=llm)
    register_actions(rails, exp_num)
    return rails


COLANG_SNIPPETS = {
    1: "(No Colang — this is a raw LLM call with no NeMo rails)",
    2: COLANG_TOPIC_GUARD,
    3: COLANG_JAILBREAK,
    4: COLANG_SENSITIVE,
    5: COLANG_DIALOG,
    6: COLANG_ACTIONS,
    7: COLANG_OUTPUT_RAIL,
}
