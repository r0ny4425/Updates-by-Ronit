"""Classical message JSON helpers for the BB84 post-processing example."""

from __future__ import annotations

import json
from typing import Any

from simyuj.control.payloads import AgentMessage


def encode_body(body: dict[str, Any]) -> str:
    return json.dumps(body, separators=(",", ":"), sort_keys=True)


def decode_body(agent_msg: AgentMessage) -> dict[str, Any]:
    message = agent_msg.message
    text = (
        message.body.decode("utf-8")
        if isinstance(message.body, bytes)
        else message.body
    )
    value = json.loads(text)
    if not isinstance(value, dict):
        raise TypeError("message body must decode to a JSON object")
    return value
