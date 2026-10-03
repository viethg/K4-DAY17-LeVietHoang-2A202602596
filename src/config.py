import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Shared configuration for the lab.

    Paths:
    - base_dir: root directory of the repo
    - data_dir: dataset directory containing benchmark conversations
    - state_dir: state directory for persistent user profiles and memory files

    Compact-memory knobs:
    - compact_threshold_tokens: token limit before compaction is triggered
    - compact_keep_messages: number of recent messages to preserve intact

    Models:
    - model: provider configuration for main agent
    - judge_model: provider configuration for evaluation/judge
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int = 1200
    compact_keep_messages: int = 4
    model: ProviderConfig = None  # type: ignore[assignment]
    judge_model: ProviderConfig = None  # type: ignore[assignment]


def _resolve_provider_config(
    provider_name: str | None,
    model_name: str | None,
    temperature: float,
    api_key: str | None = None,
    base_url: str | None = None,
) -> ProviderConfig:
    provider = normalize_provider(provider_name or "openai")

    default_models = {
        "openai": "gpt-4o-mini",
        "custom": "custom-model",
        "gemini": "gemini-1.5-flash",
        "anthropic": "claude-3-haiku-20240307",
        "ollama": "llama3",
        "openrouter": "openai/gpt-4o-mini",
    }
    resolved_model = model_name or default_models.get(provider, "gpt-4o-mini")

    resolved_base_url = base_url
    if not resolved_base_url:
        if provider == "ollama":
            resolved_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        elif provider == "openrouter":
            resolved_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        elif provider == "custom":
            resolved_base_url = os.getenv("CUSTOM_BASE_URL", "http://localhost:8000/v1")

    resolved_api_key = api_key
    if not resolved_api_key:
        if provider == "openai":
            resolved_api_key = os.getenv("OPENAI_API_KEY")
        elif provider == "custom":
            resolved_api_key = os.getenv("CUSTOM_API_KEY")
        elif provider == "gemini":
            resolved_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        elif provider == "anthropic":
            resolved_api_key = os.getenv("ANTHROPIC_API_KEY")
        elif provider == "ollama":
            resolved_api_key = os.getenv("OLLAMA_API_KEY")
        elif provider == "openrouter":
            resolved_api_key = os.getenv("OPENROUTER_API_KEY")

    return ProviderConfig(
        provider=provider,
        model_name=resolved_model,
        temperature=temperature,
        api_key=resolved_api_key,
        base_url=resolved_base_url,
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a populated LabConfig.

    Steps:
    1. Resolve repo root or default to current file parent's parent.
    2. Load values from `.env` if available.
    3. Ensure `state/` exists.
    4. Populate and return a LabConfig instance with provider and compact settings.
    """
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    # Load from .env if present
    env_file = root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()

    # Paths
    data_dir = Path(os.getenv("DATA_DIR", root / "data")).resolve()
    state_dir = Path(os.getenv("STATE_DIR", root / "state")).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)

    # Compact memory parameters
    compact_threshold_tokens = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "1200"))
    compact_keep_messages = int(os.getenv("COMPACT_KEEP_MESSAGES", "4"))

    # Provider detection for main model
    detected_provider = os.getenv("LLM_PROVIDER")
    if not detected_provider:
        if os.getenv("OPENAI_API_KEY"):
            detected_provider = "openai"
        elif os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            detected_provider = "gemini"
        elif os.getenv("ANTHROPIC_API_KEY"):
            detected_provider = "anthropic"
        elif os.getenv("OPENROUTER_API_KEY"):
            detected_provider = "openrouter"
        elif os.getenv("OLLAMA_BASE_URL"):
            detected_provider = "ollama"
        elif os.getenv("CUSTOM_BASE_URL"):
            detected_provider = "custom"
        else:
            detected_provider = "openai"

    llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    model_config = _resolve_provider_config(
        provider_name=detected_provider,
        model_name=os.getenv("LLM_MODEL"),
        temperature=llm_temperature,
    )

    # Judge model configuration
    judge_provider = os.getenv("JUDGE_PROVIDER", detected_provider)
    judge_model_name = os.getenv("JUDGE_MODEL", os.getenv("LLM_MODEL"))
    judge_temperature = float(os.getenv("JUDGE_TEMPERATURE", "0.0"))
    judge_model_config = _resolve_provider_config(
        provider_name=judge_provider,
        model_name=judge_model_name,
        temperature=judge_temperature,
    )

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold_tokens,
        compact_keep_messages=compact_keep_messages,
        model=model_config,
        judge_model=judge_model_config,
    )

