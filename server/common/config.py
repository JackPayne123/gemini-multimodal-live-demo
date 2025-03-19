import os

from bots.types import BotConfig

DEFAULT_LLM_CONTEXT = [
    {
        "content": {
            "role": "system",
            "content": "Act like a human, but remember that you aren't a human and that you can't do human things in the real world. Your voice and personality should be warm and engaging, with a lively and playful tone.",
        }
    }
]


DEFAULT_BOT_CONFIG = BotConfig(
    config=[
        {"options": [{"name": "params", "value": {"stop_secs": 0.5}}], "service": "vad"},
        {
            "options": [
                {"name": "run_on_config", "value": False},
            ],
            "service": "llm",
        },
    ],
)

SERVICE_API_KEYS = {
    "gemini": os.getenv("GEMINI_API_KEY"),
    "daily": os.getenv("DAILY_API_KEY"),
}
