"""Page 11 — Test Runner (pytest with live-streaming output)."""

import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from streamlit_app.utils.page_config import apply_page_config
import streamlit as st

st.set_page_config(page_title="Test Runner | FullRag", page_icon="🧪", layout="wide")

apply_page_config()

st.title("🧪 Test Runner")
st.markdown("Select test files and run pytest with live streaming output.")
st.markdown("---")

# ── Test file discovery ───────────────────────────────────────────────────────
test_dir = _ROOT / "test"
all_test_files = sorted(test_dir.glob("test_*.py"))
test_names = [f.name for f in all_test_files]

st.subheader("📋 Select Tests")
col_btns, _ = st.columns([1, 3])
with col_btns:
    if st.button("✅ Select All", use_container_width=True):
        for name in test_names:
            st.session_state[f"test_{name}"] = True
    if st.button("⬜ Deselect All", use_container_width=True):
        for name in test_names:
            st.session_state[f"test_{name}"] = False

selected: list[Path] = []
cols = st.columns(3)
for i, (f, name) in enumerate(zip(all_test_files, test_names)):
    col = cols[i % 3]
    checked = col.checkbox(name, value=st.session_state.get(f"test_{name}", False), key=f"test_{name}")
    if checked:
        selected.append(f)

st.markdown(f"**{len(selected)}/{len(test_names)}** test files selected.")
st.markdown("---")

# ── Run button ────────────────────────────────────────────────────────────────
run_btn = st.button("▶️ Run Selected Tests", type="primary", disabled=len(selected) == 0)

if "test_output" not in st.session_state:
    st.session_state["test_output"] = []
if "test_summary" not in st.session_state:
    st.session_state["test_summary"] = None

if run_btn:
    if not selected:
        st.warning("Select at least one test file.")
        st.stop()

    st.session_state["test_output"] = []
    st.session_state["test_summary"] = None

    cmd = [
        sys.executable, "-m", "pytest",
        *[str(f) for f in selected],
        "--tb=short", "-v", "--no-header",
    ]

    st.subheader("📡 Live Output")
    output_area = st.empty()
    lines: list[str] = []

    with st.spinner("Running pytest…"):
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(_ROOT),
        )

        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                lines.append(line.rstrip())
                output_area.code("\n".join(lines[-60:]), language="text")

        proc.wait()

    st.session_state["test_output"] = lines

    # ── Parse summary ─────────────────────────────────────────────────────────
    passed = failed = errors = skipped = 0
    test_results: list[dict] = []

    for line in lines:
        if " PASSED" in line:
            passed += 1
            test_results.append({"status": "✅ PASSED", "test": line.split(" PASSED")[0].strip()})
        elif " FAILED" in line:
            failed += 1
            test_results.append({"status": "❌ FAILED", "test": line.split(" FAILED")[0].strip()})
        elif " ERROR" in line:
            errors += 1
            test_results.append({"status": "⚠️ ERROR", "test": line.split(" ERROR")[0].strip()})
        elif " SKIPPED" in line:
            skipped += 1
            test_results.append({"status": "⏭️ SKIPPED", "test": line.split(" SKIPPED")[0].strip()})

    st.session_state["test_summary"] = {
        "passed": passed, "failed": failed, "errors": errors,
        "skipped": skipped, "results": test_results,
        "returncode": proc.returncode,
    }

# ── Show previous output + summary ───────────────────────────────────────────
if st.session_state["test_output"] and not run_btn:
    st.subheader("📡 Previous Output")
    with st.expander("View raw output", expanded=False):
        st.code("\n".join(st.session_state["test_output"]), language="text")

if st.session_state["test_summary"]:
    summary = st.session_state["test_summary"]
    st.markdown("---")
    st.subheader("📊 Test Summary")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("✅ Passed", summary["passed"])
    c2.metric("❌ Failed", summary["failed"])
    c3.metric("⚠️ Errors", summary["errors"])
    c4.metric("⏭️ Skipped", summary["skipped"])
    total = summary["passed"] + summary["failed"] + summary["errors"]
    overall = "✅ All Passed" if summary["failed"] == 0 and summary["errors"] == 0 else "❌ Failures"
    c5.metric("Overall", overall)

    if summary["results"]:
        import pandas as pd
        df = pd.DataFrame(summary["results"])
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Show failure details from raw output
    if summary["failed"] > 0 or summary["errors"] > 0:
        st.markdown("---")
        st.subheader("🔍 Failure & Error Details")
        raw = "\n".join(st.session_state["test_output"])
        # Extract FAILED/ERROR sections
        in_failure = False
        failure_lines: list[str] = []
        for line in st.session_state["test_output"]:
            if "FAILED" in line or "ERROR" in line or "_ _ _" in line or "E " == line[:2]:
                in_failure = True
            if in_failure:
                failure_lines.append(line)
            if line.startswith("=") and in_failure and len(failure_lines) > 2:
                in_failure = False

        if failure_lines:
            st.code("\n".join(failure_lines[:200]), language="python")
