# providers/gemini.py

import json
import google.generativeai as genai
from .base import AIProvider

SYSTEM_MSG = (
    "You are a security assistant helping with web fuzzing. "
    "Return only valid JSON. No prose."
)

class GeminiProvider(AIProvider):
    name = "gemini"

    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            "gemini-1.5-pro",
            generation_config={
                "temperature": 0,
                "response_mime_type": "application/json",
            },
        )

    def get_extensions(self, url, headers, max_extensions):
        prompt = f"""
Given the following URL and HTTP headers, suggest the most likely file extensions
for fuzzing this endpoint.

Respond with valid JSON only.
Format:
{{"extensions": [".php", ".json", ".bak"]}}

Limit to at most {max_extensions} extensions.

URL: {url}
Headers: {headers}
"""

        resp = self.model.generate_content(
            f"SYSTEM:\n{SYSTEM_MSG}\n\nUSER:\n{prompt}"
        )

        return json.loads(resp.text.strip())
