# openrouter_client.py

import os
from pathlib import Path
from openai import OpenAI

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(dotenv_path=None, override=False, *args, **kwargs) -> bool:
        if dotenv_path is None:
            return False
        path = Path(dotenv_path)
        if not path.exists():
            return False
        loaded = False
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            if override or key not in os.environ:
                os.environ[key] = value
                loaded = True
        return loaded

load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)
load_dotenv()

DEFAULT_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-oss-120b:free")
DEFAULT_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_NIM_MODEL = os.environ.get("NIM_MODEL", "meta/llama-3.1-70b-instruct")
DEFAULT_NIM_BASE_URL = os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
DEFAULT_INFERENCE_PROVIDER = os.environ.get("INFERENCE_PROVIDER", "openrouter")


class OpenRouter(OpenAI):

    def __init__(self, **kwargs):
        if "model" in kwargs:
            self.model = kwargs.pop("model")
        else:
            self.model = DEFAULT_MODEL

        if self.model is None or self.model == "":
            raise Exception("model argument must be passed or OPENROUTER_MODEL environment var must be defined.")

        api_key = kwargs.get("api_key") or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if api_key is None or api_key == "":
            raise Exception("OPENROUTER_API_KEY environment var not defined.")

        default_headers = kwargs.get("default_headers", {})
        if not isinstance(default_headers, dict):
            default_headers = dict(default_headers)
        default_headers.setdefault("HTTP-Referer", os.environ.get("OPENROUTER_HTTP_REFERER", "http://localhost"))
        default_headers.setdefault("X-Title", os.environ.get("OPENROUTER_APP_TITLE", "function-understanding"))

        kwargs["base_url"] = DEFAULT_BASE_URL
        kwargs["api_key"] = api_key
        kwargs["default_headers"] = default_headers

        super().__init__(**kwargs)

    def generate_text(self, prompt: str, max_tokens: int = 50):
        """
        Generate text using the OpenRouter client.

        Args:
            prompt (str): The input text prompt.
            max_tokens (int): The maximum number of tokens to generate. Defaults to 50.

        Returns:
            str: The generated text.
        """
        try:
            response = self.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            message = response.choices[0].message
            return (message.content or "").strip()
        except Exception as e:
            return f"Error: {str(e)}"


class NvidiaNIM(OpenAI):

    def __init__(self, **kwargs):
        if "model" in kwargs:
            self.model = kwargs.pop("model")
        else:
            self.model = DEFAULT_NIM_MODEL

        if self.model is None or self.model == "":
            raise Exception("model argument must be passed or NIM_MODEL environment var must be defined.")

        api_key = kwargs.get("api_key") or os.environ.get("NIM_API_KEY") or os.environ.get("NVIDIA_API_KEY")
        if api_key is None or api_key == "":
            raise Exception("NIM_API_KEY or NVIDIA_API_KEY environment var not defined.")

        kwargs["base_url"] = DEFAULT_NIM_BASE_URL
        kwargs["api_key"] = api_key

        super().__init__(**kwargs)

    def generate_text(self, prompt: str, max_tokens: int = 50):
        """
        Generate text using an NVIDIA NIM OpenAI-compatible endpoint.

        Args:
            prompt (str): The input text prompt.
            max_tokens (int): The maximum number of tokens to generate. Defaults to 50.

        Returns:
            str: The generated text.
        """
        try:
            response = self.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            message = response.choices[0].message
            return (message.content or "").strip()
        except Exception as e:
            return f"Error: {str(e)}"


def get_inference_client(provider: str | None = None, **kwargs):
    selected_provider = (provider or DEFAULT_INFERENCE_PROVIDER).strip().lower()
    aliases = selected_provider.replace("-", "_")

    if aliases in {"openrouter", "open_router", "or"}:
        return OpenRouter(**kwargs)
    if aliases in {"nim", "nvidia", "nvidia_nim", "nvidia_nims"}:
        return NvidiaNIM(**kwargs)

    raise ValueError(
        f"unsupported provider '{selected_provider}'. Expected one of: openrouter, nim."
    )


if __name__ == "__main__":
    provider = os.environ.get("INFERENCE_PROVIDER", "openrouter")
    client = get_inference_client(provider=provider)
    system_prompt_path = Path(__file__).parent.parent.joinpath("configs").joinpath("system_prompt.txt")
    input = system_prompt_path.read_text(encoding="utf-8")
    prompt = "Generate a big number widget that shows the sum total latency for catalogue service."
    llm_input = input.format(prompt)
    generated_text = client.generate_text(llm_input, max_tokens=1024)
    print(generated_text)
