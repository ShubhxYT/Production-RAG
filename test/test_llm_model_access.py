"""Live smoke tests — verify LLM model access for Cerebras and Gemini.

These tests call real APIs and require valid keys in your .env file.
Run them with:

    uv run pytest test/test_llm_model_access.py -v -s

Each test class has three checks per provider:
  1. Can we reach the API at all and list models?
  2. Is the configured model name actually in your plan/allowed list?
  3. Does a minimal generation call work end-to-end?

If check #2 fails, the error message tells you which models ARE available
so you can update CEREBRAS_MODEL or GENERATION_MODEL in your .env.
"""

import pytest

from config.settings import (
    get_gemini_api_key,
    get_generation_model,
    get_groq_api_key,
    get_groq_base_url,
    get_groq_model,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _groq_client():
    import openai

    return openai.OpenAI(
        api_key=get_groq_api_key(),
        base_url=get_groq_base_url(),
    )


def _gemini_client():
    from google import genai

    return genai.Client(api_key=get_gemini_api_key())


# ---------------------------------------------------------------------------
# Cerebras
# ---------------------------------------------------------------------------


class TestGroqModelAccess:
    """Live tests to confirm Groq API access and model availability."""

    def test_can_list_groq_models(self):
        """Verify the Groq /openai/v1/models endpoint is reachable."""
        client = _groq_client()
        models = list(client.models.list())
        model_ids = [m.id for m in models]
        print(f"\nGroq models available on your plan ({len(model_ids)}):")
        for mid in sorted(model_ids):
            print(f"  {mid}")
        assert model_ids, "No models returned from Groq — check your API key."

    def test_configured_groq_model_is_available(self):
        """Fail with a clear message if GROQ_MODEL is not in your plan."""
        client = _groq_client()
        configured = get_groq_model()
        models = list(client.models.list())
        model_ids = [m.id for m in models]

        print(f"\nConfigured GROQ_MODEL: '{configured}'")
        print(f"Models available on your plan: {sorted(model_ids)}")

        assert configured in model_ids, (
            f"\n\nModel '{configured}' is NOT available on your Groq plan.\n"
            f"Available models: {sorted(model_ids)}\n\n"
            f"Fix: add/update this line in your .env file:\n"
            f"  GROQ_MODEL=<one of the model IDs listed above>"
        )

    def test_groq_smoke_generation(self):
        """Send a one-token prompt to confirm generation works end-to-end."""
        from generation.llm_service import GroqProvider
        from generation.models import GenerationConfig

        provider = GroqProvider()
        config = GenerationConfig(
            model_name=get_groq_model(),
            temperature=0.0,
            max_output_tokens=8,
        )
        response = provider.generate(
            system_prompt="You are a helpful assistant.",
            user_prompt="Reply with only the word: pong",
            config=config,
        )
        print(f"\nGroq smoke response: {response.text!r}  (model={response.model})")
        assert response.text.strip(), "Groq returned an empty response."


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


class TestGeminiModelAccess:
    """Live tests to confirm Gemini API access and model availability."""

    def test_can_list_gemini_models(self):
        """Verify the Gemini models list endpoint is reachable."""
        client = _gemini_client()
        models = list(client.models.list())
        # Filter to generative models only for a cleaner output
        model_names = [m.name for m in models]
        generative = [n for n in model_names if "gemini" in n.lower()]
        print(f"\nGemini generative models on your plan ({len(generative)}):")
        for name in sorted(generative):
            print(f"  {name}")
        assert model_names, "No models returned from Gemini — check your API key."

    def test_configured_gemini_model_is_available(self):
        """Fail with a clear message if GENERATION_MODEL is not in your plan.

        Gemini model names in the API are prefixed with 'models/', e.g.
        'models/gemini-2.5-flash'.  The configured value (e.g. 'gemini-2.5-flash')
        is matched against the suffix so both forms work.
        """
        client = _gemini_client()
        configured = get_generation_model()
        models = list(client.models.list())
        model_names = [m.name for m in models]  # e.g. ["models/gemini-2.5-flash", ...]

        # Accept exact match OR suffix match (with or without 'models/' prefix)
        matches = [
            n for n in model_names
            if n == configured or n.endswith(f"/{configured}")
        ]

        generative = sorted(n for n in model_names if "gemini" in n.lower())
        print(f"\nConfigured GENERATION_MODEL: '{configured}'")
        print(f"Gemini generative models on your plan: {generative}")
        print(f"Matches found: {matches}")

        assert matches, (
            f"\n\nModel '{configured}' is NOT available on your Gemini plan.\n"
            f"Generative models on your plan: {generative}\n\n"
            f"Fix: add/update this line in your .env file:\n"
            f"  GENERATION_MODEL=<model name without the 'models/' prefix>"
        )

    def test_gemini_smoke_generation(self):
        """Send a one-token prompt to confirm Gemini generation works end-to-end."""
        from generation.llm_service import GeminiGenerationProvider
        from generation.models import GenerationConfig

        provider = GeminiGenerationProvider()
        config = GenerationConfig(
            model_name=get_generation_model(),
            temperature=0.0,
            max_output_tokens=8,
        )
        response = provider.generate(
            system_prompt="You are a helpful assistant.",
            user_prompt="Reply with only the word: pong",
            config=config,
        )
        print(f"\nGemini smoke response: {response.text!r}  (model={response.model})")
        assert response.text.strip(), "Gemini returned an empty response."
