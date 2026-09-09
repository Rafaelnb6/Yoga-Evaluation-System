"""Generate concise, data-grounded feedback with the DeepSeek API."""

import json
import os
from typing import Dict, Optional

from openai import OpenAI


SYSTEM_PROMPT = """You are a quantitative yoga assessment assistant. Use only the provided joint similarities and overall score. Give concise, objective, actionable feedback. Do not invent angles, durations, poses, or deviations that are absent from the input. Return plain text without Markdown and stay under 80 words."""


def get_deepseek_advice(
    joint_scores: Dict[str, float],
    overall_score: float,
    api_key: Optional[str] = None,
) -> str:
    """Call DeepSeek and return one plain-text recommendation."""
    resolved_key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not resolved_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")

    client = OpenAI(
        api_key=resolved_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        timeout=30.0,
    )
    user_prompt = (
        f"Overall score: {overall_score:.1f}\n"
        f"Joint similarities: {json.dumps(joint_scores, ensure_ascii=True)}"
    )
    response = client.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
    )
    return response.choices[0].message.content.strip()
