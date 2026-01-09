from google import genai

class GeminiProvider:
    def __init__(self, api_key, model="gemini-1.5-pro"):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        return response.text.strip()
