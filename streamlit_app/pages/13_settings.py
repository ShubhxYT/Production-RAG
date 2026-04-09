"""Page 13 — Settings & Configuration (read-only)."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from streamlit_app.utils.page_config import apply_page_config
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Settings | FullRag", page_icon="⚙️", layout="wide")

apply_page_config()

st.title("⚙️ Settings & Configuration")
st.markdown("Read-only view of all environment variables and system configuration.")
st.markdown("---")

def _mask(val: str) -> str:
    """Mask a secret — show first 4 and last 4 chars."""
    if not val or len(val) < 10:
        return "****"
    return val[:4] + "****" + val[-4:]


# ── Environment variables ─────────────────────────────────────────────────────
st.subheader("🔐 Environment Variables")

try:
    import os
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")

    env_rows = []
    config_vars = [
        ("GEMINI_API_KEY", True),
        ("GROQ_API_KEY", True),
        ("DATABASE_URL", False),
        ("GENERATION_MODEL", False),
        ("GENERATION_TEMPERATURE", False),
        ("GENERATION_MAX_TOKENS", False),
        ("GENERATION_PROVIDER", False),
        ("GROQ_BASE_URL", False),
        ("GROQ_MODEL", False),
        ("LOG_LEVEL", False),
        ("LOG_FORMAT", False),
        ("LOG_FILE", False),
        ("CACHE_TTL_SECONDS", False),
        ("CACHE_MAX_SIZE", False),
        ("RESPONSE_CACHE_ENABLED", False),
        ("CONTINUOUS_EVAL_ENABLED", False),
        ("EVAL_SCHEDULE_INTERVAL_HOURS", False),
    ]

    for var, is_secret in config_vars:
        raw = os.environ.get(var, "")
        if is_secret and raw:
            display = _mask(raw)
            status = "🟢 Set"
        elif is_secret and not raw:
            display = "—"
            status = "🔴 Not set"
        else:
            display = raw or "—"
            status = "🟢 Set" if raw else "⚪ Default"
        env_rows.append({"Variable": var, "Value": display, "Status": status})

    st.dataframe(pd.DataFrame(env_rows), use_container_width=True, hide_index=True)
except Exception as e:
    st.error(f"Could not load env: {e}")

st.markdown("---")

# ── Generation config ─────────────────────────────────────────────────────────
st.subheader("🤖 Generation Configuration")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**GenerationConfig defaults:**")
    try:
        from generation.models import GenerationConfig
        gc = GenerationConfig()
        st.json({
            "model_name": gc.model_name,
            "temperature": gc.temperature,
            "max_output_tokens": gc.max_output_tokens,
            "max_context_tokens": gc.max_context_tokens,
            "context_budget_ratio": gc.context_budget_ratio,
        })
    except Exception as e:
        st.error(str(e))

with col2:
    st.markdown("**LLMConfig defaults (enrichment):**")
    try:
        from generation.models import LLMConfig
        lc = LLMConfig()
        st.json({
            "model_name": lc.model_name,
            "temperature": lc.temperature,
            "max_output_tokens": lc.max_output_tokens,
        })
    except Exception as e:
        st.error(str(e))

st.markdown("---")

# ── Embedding config ──────────────────────────────────────────────────────────
st.subheader("🧲 Embedding Configuration")

try:
    import torch
    from embeddings.models import EmbeddingConfig
    ec = EmbeddingConfig()
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.json({
            "model_name": ec.model_name,
            "dimensions": ec.dimensions,
            "batch_size": ec.batch_size,
            "max_retries": ec.max_retries,
        })
    with col_e2:
        st.metric("Compute Device", device)
        if torch.cuda.is_available():
            st.metric("GPU", torch.cuda.get_device_name(0))
            mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            st.metric("VRAM", f"{mem_gb:.1f} GB")
except Exception as e:
    st.error(str(e))

st.markdown("---")

# ── Cache config ──────────────────────────────────────────────────────────────
st.subheader("💾 Cache Configuration")

try:
    from config.settings import (
        get_cache_ttl_seconds,
        get_cache_max_size,
        get_response_cache_enabled,
        get_continuous_eval_enabled,
        get_eval_schedule_interval_hours,
    )
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Cache Enabled", "✅ Yes" if get_response_cache_enabled() else "❌ No")
    cc2.metric("TTL", f"{get_cache_ttl_seconds()}s")
    cc3.metric("Max Entries", get_cache_max_size())

    st.markdown("---")
    st.subheader("⏰ Continuous Evaluation Scheduler")
    sc1, sc2 = st.columns(2)
    sc1.metric("Enabled", "✅ Yes" if get_continuous_eval_enabled() else "❌ No")
    sc2.metric("Interval", f"{get_eval_schedule_interval_hours()}h")

except Exception as e:
    st.error(str(e))

st.markdown("---")

# ── Database URL ───────────────────────────────────────────────────────────────
st.subheader("🗄️ Database")
try:
    from config.settings import get_database_url
    db_url = get_database_url()
    if "@" in db_url:
        scheme_and_creds, rest = db_url.split("@", 1)
        scheme = scheme_and_creds.split("://")[0]
        masked = f"{scheme}://****:****@{rest}"
    else:
        masked = db_url
    st.code(masked, language="text")
except Exception as e:
    st.error(str(e))
