"""
O-Health Clinical Triage Engine — Streamlit App
================================================
Full pipeline:
  1. Patient demographics + chief complaint
  2. Iterative Q&A with KG + LLM extraction
  3. Triage result display
  4. Doctor review + RLHF feedback → pushed to GitHub

Requires Streamlit secrets:
  [ollama]
  url = "https://xxx.ngrok-free.app/api/generate"
  auth_user = "ohealth"
  auth_password = "xxx"
  model = "phi3:mini"

  [github]
  token = "ghp_xxx"
  repo = "org/triage-engine"
  branch = "main"
"""

import streamlit as st
import json
import time
from collections import defaultdict

# Engine imports
from graph_engine import KnowledgeGraph, ActivationSpreader
from question_selector import QuestionSelector
from enhanced_parser import EnhancedResponseParser
from smart_resolver import SmartSymptomResolver
from entity_mapper import EntityMapper
from question_dedup import ExtraMedicalInfo, QuestionDedupGate
from remote_llm import RemoteLLMExtractor
from github_persistence import GitHubPersistence

# ─── Config ───
MAX_QUESTIONS = 9
PRIMARY_BOOST = 2.0
VOLUNTEER_BOOST = 1.3
CONFIDENCE_THRESHOLD = 0.30
ENTROPY_THRESHOLD = 3.0
MIN_INFO_GAIN = 0.005

SEVERITY_COLORS = {
    "critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"
}

# ─── Page Config ───
st.set_page_config(
    page_title="O-Health Triage Engine",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════
# INITIALIZATION
# ═══════════════════════════════════════════════

@st.cache_resource
def init_graph():
    """Load the knowledge graph (cached across sessions)."""
    return KnowledgeGraph()


def init_llm():
    """Initialize remote LLM from Streamlit secrets."""
    try:
        cfg = st.secrets.get("ollama", {})
        return RemoteLLMExtractor(
            url=cfg.get("url", "http://localhost:11434/api/generate"),
            model=cfg.get("model", "phi3:mini"),
            auth_user=cfg.get("auth_user"),
            auth_password=cfg.get("auth_password"),
            timeout=15,
        )
    except Exception:
        return RemoteLLMExtractor()  # fallback to localhost


def init_github():
    """Initialize GitHub persistence from Streamlit secrets."""
    try:
        cfg = st.secrets.get("github", {})
        token = cfg.get("token", "")
        repo = cfg.get("repo", "")
        branch = cfg.get("branch", "main")
        if token and repo:
            gh = GitHubPersistence(token, repo, branch)
            return gh
    except Exception:
        pass
    return None


def load_weights(graph, gh):
    """Load calibrated weights from GitHub if available."""
    if gh is None:
        return
    weights, sha = gh.pull_weights()
    if weights:
        for sym, edge_list in weights.items():
            graph.edges[sym] = [(cond, w) for cond, w in edge_list]
        st.session_state["weights_sha"] = sha
        return len(weights)
    return 0


def save_weights(graph, gh, session_info=""):
    """Push updated weights to GitHub."""
    if gh is None:
        return False
    edges = {}
    for sym, edge_list in graph.edges.items():
        edges[sym] = [(cond, round(w, 4)) for cond, w in edge_list]
    sha = st.session_state.get("weights_sha")
    ok = gh.push_weights(edges, sha=sha, session_info=session_info)
    if ok:
        # Update SHA for next push
        _, new_sha = gh.pull_weights()
        st.session_state["weights_sha"] = new_sha
    return ok


def save_feedback_to_github(gh, feedback_entry):
    """Append a feedback entry to the GitHub feedback log."""
    if gh is None:
        return False
    log, sha = gh.pull_feedback_log()
    if log is None:
        log = []
    log.append(feedback_entry)
    return gh.push_feedback_log(log, sha=sha)


# ═══════════════════════════════════════════════
# SESSION STATE MANAGEMENT
# ═══════════════════════════════════════════════

def reset_session():
    """Reset for a new patient."""
    keys_to_clear = [
        "phase", "patient_age", "patient_gender",
        "initial_input", "initial_symptoms", "spreader_data",
        "iterations", "asked_ids", "asked_elicits", "asked_categories",
        "questions_log", "extra_info_data", "triage_done",
        "current_question", "current_question_ig",
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def init_session_state():
    """Initialize session state defaults."""
    if "phase" not in st.session_state:
        st.session_state["phase"] = "demographics"
    if "iterations" not in st.session_state:
        st.session_state["iterations"] = []


# ═══════════════════════════════════════════════
# APPLY RESPONSE TO SPREADER
# ═══════════════════════════════════════════════

def apply_response(parsed, spreader, graph, elicited, is_initial=False):
    """Apply parsed response to spreader. Returns log messages."""
    logs = []
    strength = PRIMARY_BOOST if is_initial else 1.0

    if parsed.get("is_null"):
        spreader.activate(elicited, present=False)
        logs.append(("deny", elicited, "skipped"))
        return logs

    if parsed.get("is_yes"):
        spreader.activate(elicited, present=True, strength=strength)
        logs.append(("confirm", elicited, "%.1fx" % strength))
    elif parsed.get("is_no"):
        spreader.activate(elicited, present=False)
        logs.append(("deny", elicited, ""))

    for sym in parsed.get("detected_symptoms", []):
        if sym == elicited:
            continue
        if sym not in spreader.active_symptoms:
            vol = VOLUNTEER_BOOST if not is_initial else PRIMARY_BOOST
            spreader.activate(sym, present=True, strength=vol)
            logs.append(("detect", sym, "%.1fx" % vol))

    for sym in parsed.get("negated_symptoms", []):
        if sym not in spreader.negative_symptoms:
            spreader.activate(sym, present=False)
            logs.append(("negate", sym, ""))

    for finding in parsed.get("matched_findings", []):
        if finding.startswith("hint_"):
            continue
        if finding not in spreader.active_symptoms:
            spreader.activate(finding, present=True)
            logs.append(("finding", finding, ""))

    dur = parsed.get("duration")
    if dur and dur.get("value") is not None:
        key = "%s_duration" % elicited
        spreader.metadata[key] = "%s %s" % (dur["value"], dur.get("unit", "days"))
        logs.append(("duration", key, spreader.metadata[key]))

    for med in parsed.get("medications_found", []):
        if med not in spreader.active_symptoms:
            spreader.activate(med, present=True)
            logs.append(("medication", med, ""))

    for hist in parsed.get("history_found", []):
        if hist not in spreader.active_symptoms:
            spreader.activate(hist, present=True)
            logs.append(("history", hist, ""))

    return logs


# ═══════════════════════════════════════════════
# UI COMPONENTS
# ═══════════════════════════════════════════════

def render_sidebar(graph, llm, gh):
    """Render the status sidebar."""
    with st.sidebar:
        st.markdown("### 🏥 System Status")

        # LLM status — with debug details
        llm_ok = llm.is_available()
        if llm_ok:
            st.success("✅ LLM: Online (%s)" % llm.model)
        else:
            st.warning("⚠️ LLM: Offline (using fast parser)")

        # Debug expander — always visible so we can diagnose
        with st.expander("🔍 LLM Debug", expanded=not llm_ok):
            st.caption("URL: `%s`" % llm.url)
            st.caption("Model: `%s`" % llm.model)
            st.caption("Auth user: `%s`" % (llm.auth[0] if llm.auth else "None"))
            st.caption("Auth password: `%s`" % ("*" * len(llm.auth[1]) if llm.auth else "None"))

            if st.button("Test Connection Now", key="test_llm"):
                test_url = llm.url.replace("/api/generate", "/api/tags")
                st.caption("Testing: `%s`" % test_url)
                try:
                    import requests as req
                    resp = req.get(
                        test_url,
                        auth=llm.auth,
                        headers=llm.headers,
                        timeout=8,
                    )
                    st.caption("HTTP Status: `%d`" % resp.status_code)
                    if resp.status_code == 200:
                        st.success("Connection OK!")
                        try:
                            data = resp.json()
                            models = [m.get("name") for m in data.get("models", [])]
                            st.caption("Models: %s" % models)
                        except Exception:
                            st.caption("Response (raw): `%s`" % resp.text[:200])
                    elif resp.status_code == 401:
                        st.error("❌ 401 Unauthorized — wrong password in Streamlit secrets")
                        st.caption("Response: `%s`" % resp.text[:300])
                    elif resp.status_code == 403:
                        st.error("❌ 403 Forbidden — auth rejected")
                    else:
                        st.warning("Unexpected status %d" % resp.status_code)
                        st.caption("Response: `%s`" % resp.text[:300])
                except Exception as e:
                    st.error("Connection error: `%s`" % str(e))

        # GitHub status
        if gh and gh.is_configured():
            st.success("✅ GitHub: Connected")
        else:
            st.warning("⚠️ GitHub: Not configured")

        # Graph stats
        st.markdown("---")
        st.markdown("### 📊 Knowledge Graph")
        st.markdown("- **%d** symptoms" % len(graph.symptoms))
        st.markdown("- **%d** conditions" % len(graph.conditions))
        st.markdown("- **%d** questions" % len(graph.questions))

        # Current session
        if "spreader_data" in st.session_state:
            st.markdown("---")
            st.markdown("### 🔬 Current Session")
            sd = st.session_state["spreader_data"]
            st.markdown("**Active:** %s" % ", ".join(sorted(sd.get("active", []))))
            st.markdown("**Denied:** %s" % ", ".join(sorted(sd.get("denied", []))))

        st.markdown("---")
        if st.button("🔄 New Patient", use_container_width=True):
            reset_session()
            st.rerun()


def render_distribution(spreader, graph, n=5):
    """Render the condition distribution as a bar chart."""
    top = spreader.get_top_conditions(n)
    H = spreader.get_entropy()

    st.markdown("**Current Assessment** (entropy=%.2f bits)" % H)

    for cond, prob in top:
        info = graph.conditions.get(cond, {})
        display = info.get("display", cond)
        severity = info.get("severity", "?")
        icon = SEVERITY_COLORS.get(severity, "⚪")
        route = info.get("route_to", "?").replace("_", " ").title()

        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.progress(min(prob, 1.0))
        with col2:
            st.markdown("**%.1f%%** %s" % (prob * 100, icon))
        with col3:
            st.caption("%s → %s" % (display, route))


def render_triage_result(spreader, graph):
    """Render the final triage result."""
    top = spreader.get_top_conditions(5)

    st.markdown("---")
    st.markdown("## 🏥 Triage Result")

    for i, (cond, prob) in enumerate(top[:3]):
        info = graph.conditions.get(cond, {})
        display = info.get("display", cond)
        severity = info.get("severity", "?")
        route = info.get("route_to", "?").replace("_", " ").title()
        icon = SEVERITY_COLORS.get(severity, "⚪")

        if i == 0:
            st.markdown("### ▸ PRIMARY: %s %s" % (icon, display))
            st.markdown("**Confidence:** %.1f%% &nbsp;|&nbsp; **Severity:** %s &nbsp;|&nbsp; **Route to:** %s" % (
                prob * 100, severity.upper(), route))
            if severity in ("critical", "high"):
                st.error("⚠️ PRIORITY: Schedule this patient promptly.")
        else:
            st.markdown("**Differential #%d:** %s %s (%.1f%%) → %s" % (
                i + 1, icon, display, prob * 100, route))

    # Evidence summary
    st.markdown("---")
    st.markdown("### 📋 Evidence Collected")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**✓ Confirmed:**")
        for sym in sorted(spreader.active_symptoms):
            display = graph.symptoms.get(sym, {}).get("display", sym)
            st.markdown("- %s" % display)
    with col2:
        st.markdown("**✗ Denied:**")
        for sym in sorted(spreader.negative_symptoms):
            st.markdown("- %s" % sym)


def render_doctor_review(spreader, graph, gh, questions_log):
    """Render the doctor review / RLHF feedback section."""
    st.markdown("---")
    st.markdown("## 👨‍⚕️ Clinician Review")

    top = spreader.get_top_conditions(5)

    st.markdown("**Engine's assessment:**")
    for i, (cond, prob) in enumerate(top[:3]):
        display = graph.conditions.get(cond, {}).get("display", cond)
        tag = "PRIMARY" if i == 0 else "#%d" % (i + 1)
        st.markdown("- **%s:** %s (%.1f%%)" % (tag, display, prob * 100))

    # Feedback buttons
    st.markdown("### Your Feedback")
    feedback_type = st.radio(
        "Is the primary diagnosis correct?",
        ["✅ Confirm — Primary is correct",
         "❌ Correct — Real diagnosis is different",
         "🔄 Rerank — Right conditions, wrong order",
         "⏭️ Skip — Cannot evaluate"],
        index=3,
        key="feedback_radio",
    )

    if "Confirm" in feedback_type:
        confidence = st.select_slider("Confidence", ["Low", "Medium", "High"], value="High")
        conf_val = {"Low": 0.3, "Medium": 0.6, "High": 1.0}[confidence]

        if st.button("Submit Confirmation", type="primary"):
            feedback = _apply_confirm(spreader, graph, top[0][0], conf_val)
            ok, msg = _push_feedback(feedback, spreader, graph, gh, questions_log)
            if ok:
                st.success(msg)
            else:
                st.warning(msg)
            render_weight_changes(feedback)

    elif "Correct" in feedback_type:
        # Show all conditions grouped by department
        all_conds = sorted(graph.conditions.items(), key=lambda x: x[1].get("route_to", ""))
        options = {}
        for key, info in all_conds:
            dept = info.get("route_to", "other").replace("_", " ").title()
            display = info.get("display", key)
            marker = ""
            for rank, (tc, _) in enumerate(top):
                if tc == key:
                    marker = " ← engine #%d" % (rank + 1)
            label = "[%s] %s%s" % (dept, display, marker)
            options[label] = key

        selected = st.selectbox("What is the correct diagnosis?", list(options.keys()))
        confidence = st.select_slider("Confidence", ["Low", "Medium", "High"], value="High", key="corr_conf")
        conf_val = {"Low": 0.3, "Medium": 0.6, "High": 1.0}[confidence]

        if st.button("Submit Correction", type="primary"):
            correct_cond = options[selected]
            feedback = _apply_correct(spreader, graph, correct_cond, top[0][0], conf_val)
            ok, msg = _push_feedback(feedback, spreader, graph, gh, questions_log)
            if ok:
                st.success(msg)
            else:
                st.warning(msg)
            render_weight_changes(feedback)

    elif "Rerank" in feedback_type:
        rerank_options = {}
        for i, (cond, prob) in enumerate(top[:3]):
            display = graph.conditions.get(cond, {}).get("display", cond)
            rerank_options["#%d: %s (%.1f%%)" % (i + 1, display, prob * 100)] = cond

        selected = st.selectbox("Which should be #1?", list(rerank_options.keys()))

        if st.button("Submit Rerank", type="primary"):
            correct_cond = rerank_options[selected]
            feedback = _apply_rerank(spreader, graph, correct_cond)
            ok, msg = _push_feedback(feedback, spreader, graph, gh, questions_log)
            if ok:
                st.success(msg)
            else:
                st.warning(msg)
            render_weight_changes(feedback)


def _apply_confirm(spreader, graph, correct_cond, confidence):
    """Apply confirm feedback to weights. Returns detail list."""
    lr = 0.02
    changes = []
    for sym in sorted(spreader.active_symptoms):
        edges = graph.edges.get(sym, [])
        for i, (cond, weight) in enumerate(edges):
            if cond == correct_cond:
                new_w = min(0.99, weight + lr * confidence)
                edges[i] = (cond, round(new_w, 4))
                changes.append({
                    "action": "strengthen",
                    "symptom": sym,
                    "condition": graph.conditions.get(cond, {}).get("display", cond),
                    "old": round(weight, 4),
                    "new": round(new_w, 4),
                    "delta": round(new_w - weight, 4),
                })
    return {"type": "confirm", "correct": correct_cond, "confidence": confidence,
            "num_changes": len(changes), "changes": changes}


def _apply_correct(spreader, graph, correct_cond, wrong_cond, confidence):
    """Apply correction feedback to weights. Returns detail list."""
    lr = 0.02
    changes = []
    for sym in sorted(spreader.active_symptoms):
        edges = graph.edges.get(sym, [])
        # Weaken wrong
        for i, (cond, weight) in enumerate(edges):
            if cond == wrong_cond:
                new_w = max(0.05, weight - lr * confidence * 0.5)
                edges[i] = (cond, round(new_w, 4))
                changes.append({
                    "action": "weaken",
                    "symptom": sym,
                    "condition": graph.conditions.get(cond, {}).get("display", cond),
                    "old": round(weight, 4),
                    "new": round(new_w, 4),
                    "delta": round(new_w - weight, 4),
                })
        # Strengthen correct
        found = False
        for i, (cond, weight) in enumerate(edges):
            if cond == correct_cond:
                found = True
                new_w = min(0.99, weight + lr * confidence)
                edges[i] = (cond, round(new_w, 4))
                changes.append({
                    "action": "strengthen",
                    "symptom": sym,
                    "condition": graph.conditions.get(cond, {}).get("display", cond),
                    "old": round(weight, 4),
                    "new": round(new_w, 4),
                    "delta": round(new_w - weight, 4),
                })
        if not found:
            edges.append((correct_cond, 0.15))
            changes.append({
                "action": "new_edge",
                "symptom": sym,
                "condition": graph.conditions.get(correct_cond, {}).get("display", correct_cond),
                "old": 0.0,
                "new": 0.15,
                "delta": 0.15,
            })
    return {"type": "correct", "correct": correct_cond, "wrong": wrong_cond,
            "num_changes": len(changes), "changes": changes}


def _apply_rerank(spreader, graph, correct_cond):
    """Apply rerank feedback. Returns detail list."""
    lr = 0.01
    changes = []
    for sym in sorted(spreader.active_symptoms):
        edges = graph.edges.get(sym, [])
        for i, (cond, weight) in enumerate(edges):
            if cond == correct_cond:
                new_w = min(0.99, weight + lr)
                edges[i] = (cond, round(new_w, 4))
                changes.append({
                    "action": "strengthen",
                    "symptom": sym,
                    "condition": graph.conditions.get(cond, {}).get("display", cond),
                    "old": round(weight, 4),
                    "new": round(new_w, 4),
                    "delta": round(new_w - weight, 4),
                })
    return {"type": "rerank", "correct": correct_cond,
            "num_changes": len(changes), "changes": changes}


def _push_feedback(feedback, spreader, graph, gh, questions_log):
    """Push feedback to GitHub. Returns (ok, message)."""
    feedback_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "feedback": {k: v for k, v in feedback.items() if k != "changes"},
        "active_symptoms": sorted(list(spreader.active_symptoms)),
        "denied_symptoms": sorted(list(spreader.negative_symptoms)),
        "questions_asked": questions_log,
    }

    if gh is None:
        return False, "GitHub not configured — weights updated in memory only."

    w_ok = save_weights(graph, gh, session_info=feedback.get("type", ""))
    f_ok = save_feedback_to_github(gh, feedback_entry)

    if w_ok and f_ok:
        return True, "✅ Pushed to GitHub successfully."
    elif w_ok:
        return True, "⚠️ Weights pushed. Feedback log push failed."
    else:
        return False, "❌ GitHub push failed. Weights updated in memory only."


def render_weight_changes(feedback):
    """Display the exact weight changes from a feedback action."""
    changes = feedback.get("changes", [])
    if not changes:
        st.info("No edge weights affected — no active symptoms connected to that condition.")
        return

    st.markdown("#### ⚖️ Weight Changes")
    st.caption("Type: **%s** | Edges adjusted: **%d**" % (
        feedback.get("type", "?").upper(), len(changes)))

    strengthened = [c for c in changes if c["action"] == "strengthen"]
    weakened = [c for c in changes if c["action"] == "weaken"]
    new_edges = [c for c in changes if c["action"] == "new_edge"]

    if strengthened:
        with st.expander("▲ Strengthened (%d edges)" % len(strengthened), expanded=True):
            for c in strengthened:
                st.markdown("- `%s` → **%s**: %.4f → %.4f (+%.4f)" % (
                    c["symptom"], c["condition"], c["old"], c["new"], c["delta"]))

    if weakened:
        with st.expander("▼ Weakened (%d edges)" % len(weakened), expanded=True):
            for c in weakened:
                st.markdown("- `%s` → **%s**: %.4f → %.4f (%.4f)" % (
                    c["symptom"], c["condition"], c["old"], c["new"], c["delta"]))

    if new_edges:
        with st.expander("✨ New edges created (%d)" % len(new_edges), expanded=True):
            for c in new_edges:
                st.markdown("- `%s` → **%s**: 0.0000 → 0.1500 (new)" % (
                    c["symptom"], c["condition"]))


# ═══════════════════════════════════════════════
# MAIN APP FLOW
# ═══════════════════════════════════════════════

def main():
    st.title("🏥 O-Health Clinical Triage Engine")
    st.caption("Knowledge Graph + LLM Extraction + Clinician Feedback")

    init_session_state()

    # Initialize components
    graph = init_graph()
    llm = init_llm()
    gh = init_github()

    # Load calibrated weights
    if "weights_loaded" not in st.session_state:
        n = load_weights(graph, gh)
        st.session_state["weights_loaded"] = True
        if n:
            st.toast("Loaded calibrated weights from %d symptom groups" % n)

    # Sidebar
    render_sidebar(graph, llm, gh)

    # Parser
    parser = EnhancedResponseParser(graph, llm_extractor=llm)
    smart = SmartSymptomResolver(graph)

    phase = st.session_state.get("phase", "demographics")

    # ─── PHASE 1: Demographics ───
    if phase == "demographics":
        st.markdown("### Step 1: Patient Demographics")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
        with col2:
            gender = st.selectbox("Gender", ["male", "female"])

        if st.button("Next →", type="primary"):
            st.session_state["patient_age"] = age
            st.session_state["patient_gender"] = gender
            st.session_state["phase"] = "chief_complaint"
            st.rerun()

    # ─── PHASE 2: Chief Complaint ───
    elif phase == "chief_complaint":
        age = st.session_state.get("patient_age", 30)
        gender = st.session_state.get("patient_gender", "male")
        st.markdown("### Step 2: Chief Complaint")
        st.markdown("**Patient:** %d years old, %s" % (age, gender))

        complaint = st.text_input(
            "Describe your main health concern:",
            placeholder="e.g., there is slight fever and cough since 3 days",
            key="complaint_input",
        )

        if st.button("Start Triage →", type="primary") and complaint:
            # Parse initial complaint
            parsed = parser.parse(complaint, is_initial=True)
            smart_resolved = smart.resolve(complaint)

            all_syms = set(parsed.get("detected_symptoms", []))
            for sym, _ in smart_resolved:
                all_syms.add(sym)
            resolved = graph.resolve_symptom(complaint)
            if resolved:
                all_syms.add(resolved)

            initial_symptoms = list(all_syms)

            if not initial_symptoms:
                st.error("Could not recognize any symptoms. Try rephrasing.")
                return

            # Create spreader and activate
            spreader = ActivationSpreader(graph)
            for sym in initial_symptoms:
                spreader.activate(sym, present=True, strength=PRIMARY_BOOST)

            # Handle duration, body parts, etc.
            dur = parsed.get("duration")
            if dur and dur.get("value"):
                spreader.metadata["initial_duration"] = "%s %s" % (dur["value"], dur.get("unit", "days"))
            for bp in parsed.get("body_parts", []):
                spreader.metadata.setdefault("body_parts", []).append(bp)
            for med in parsed.get("medications_found", []):
                spreader.activate(med, present=True, strength=PRIMARY_BOOST)
            for hist in parsed.get("history_found", []):
                spreader.activate(hist, present=True, strength=PRIMARY_BOOST)
            for sym in parsed.get("negated_symptoms", []):
                spreader.activate(sym, present=False)

            prioritized = smart.prioritize_symptoms(initial_symptoms)
            spreader.metadata["symptom_allocations"] = {s: a for s, _, a in prioritized}

            # Save to session
            st.session_state["initial_symptoms"] = initial_symptoms
            st.session_state["initial_input"] = complaint
            st.session_state["initial_parse_method"] = parsed.get("parse_method", "fast")
            st.session_state["initial_latency"] = parsed.get("metadata", {}).get("_llm_latency_ms", 0)
            st.session_state["prioritized"] = [(s, p, a) for s, p, a in prioritized]

            # Serialize spreader state
            st.session_state["spreader_data"] = {
                "activations": dict(spreader.activations),
                "active": list(spreader.active_symptoms),
                "denied": list(spreader.negative_symptoms),
                "metadata": dict(spreader.metadata),
            }

            # Initialize iteration tracking
            st.session_state["iterations"] = []
            st.session_state["asked_ids"] = set()
            st.session_state["asked_elicits"] = set()
            st.session_state["asked_categories"] = set()
            st.session_state["asked_texts"] = set()
            st.session_state["questions_log"] = []
            st.session_state["extra_info_data"] = {
                "symptoms": set(), "qualifiers": set(), "negated": set(),
                "body_parts": set(), "medications": set(),
                "family_history": set(), "personal_history": set(),
            }
            st.session_state["triage_done"] = False
            st.session_state["phase"] = "questioning"
            st.session_state["current_question"] = None
            st.rerun()

    # ─── PHASE 3: Iterative Questioning ───
    elif phase == "questioning":
        age = st.session_state.get("patient_age", 30)
        gender = st.session_state.get("patient_gender", "male")

        st.markdown("### Step 3: Follow-up Questions")
        st.markdown("**Patient:** %d years old, %s | **Chief complaint:** %s" % (
            age, gender, st.session_state.get("initial_input", "")))

        # Show initial symptoms
        prioritized = st.session_state.get("prioritized", [])
        method = st.session_state.get("initial_parse_method", "fast")
        latency = st.session_state.get("initial_latency", 0)
        with st.expander("Iteration 0 — Chief Complaint [%s%s]" % (method, ", %dms" % latency if latency else ""), expanded=False):
            for sym, pri, alloc in prioritized:
                display = graph.symptoms.get(sym, {}).get("display", sym)
                st.markdown("- **%s** (priority=%d, ~%d questions) — 2.0x boost" % (display, pri, alloc))

        # Rebuild spreader from session state
        spreader = ActivationSpreader(graph)
        sd = st.session_state["spreader_data"]
        spreader.activations = defaultdict(float, sd["activations"])
        spreader.active_symptoms = set(sd["active"])
        spreader.negative_symptoms = set(sd["denied"])
        spreader.metadata = dict(sd["metadata"])

        # Show past iterations
        for it in st.session_state["iterations"]:
            with st.expander("Iteration %d — %s [%s%s]" % (
                it["num"], it["question_id"], it["parse_method"],
                ", %dms" % it["latency"] if it["latency"] else ""
            ), expanded=False):
                st.markdown("**Q:** %s" % it["question_text"])
                st.markdown("**A:** %s" % it["response"])
                st.markdown("**Parse:** yes=%s no=%s syms=%s" % (it["is_yes"], it["is_no"], it["detected"]))
                for log_type, key, detail in it["logs"]:
                    icon = {"confirm": "✅", "deny": "❌", "detect": "🔍", "negate": "🚫",
                            "finding": "📌", "duration": "⏱", "medication": "💊", "history": "📋"}.get(log_type, "•")
                    st.markdown("  %s **%s** %s %s" % (icon, log_type, key, detail))

        # Show current distribution
        render_distribution(spreader, graph)

        # Show active/denied
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**✓ Active:** %s" % ", ".join(sorted(spreader.active_symptoms)) or "none")
        with col2:
            st.markdown("**✗ Denied:** %s" % ", ".join(sorted(spreader.negative_symptoms)) or "none")

        # Check if we should stop
        iteration_num = len(st.session_state["iterations"]) + 1
        top = spreader.get_top_conditions(1)
        should_stop = False

        if iteration_num > MAX_QUESTIONS:
            should_stop = True
        elif top and top[0][1] > CONFIDENCE_THRESHOLD:
            H = spreader.get_entropy()
            if H < ENTROPY_THRESHOLD:
                should_stop = True

        if should_stop or st.session_state.get("triage_done"):
            st.session_state["phase"] = "result"
            st.rerun()
            return

        # Select next question
        selector = QuestionSelector(graph, patient_age=age, patient_gender=gender)
        asked_ids = st.session_state.get("asked_ids", set())
        asked_elicits = st.session_state.get("asked_elicits", set())

        # Build dedup gate from session
        extra_info = ExtraMedicalInfo()
        eid = st.session_state.get("extra_info_data", {})
        extra_info.symptoms = set(eid.get("symptoms", []))
        extra_info.qualifiers = set(eid.get("qualifiers", []))
        extra_info.negated = set(eid.get("negated", []))
        gate = QuestionDedupGate(extra_info=extra_info)
        gate.asked_question_ids = set(asked_ids)
        gate.asked_elicits = set(asked_elicits)
        gate.asked_categories = set(st.session_state.get("asked_categories", []))
        gate.asked_question_texts = set(st.session_state.get("asked_texts", []))

        # Find next question (skip blocked ones)
        question = None
        ig = 0
        for _ in range(20):  # max 20 attempts to find an unblocked question
            q, score = selector.select_next(spreader, gate.asked_question_ids, asked_elicits=gate.asked_elicits)
            if q is None:
                break
            should_ask, reason, auto_fill = gate.should_ask(q, spreader)
            if should_ask:
                question = q
                ig = score
                break
            elif auto_fill:
                spreader.activate(auto_fill["symptom"], present=auto_fill["present"])

        if question is None:
            st.session_state["triage_done"] = True
            st.session_state["phase"] = "result"
            st.rerun()
            return

        # Show the question
        st.markdown("---")
        st.markdown("### Iteration %d" % iteration_num)
        st.info("**Q:** %s" % question["text"])
        st.caption("(%s, IG=%.2f)" % (question["id"], ig))

        response = st.text_input("Your answer:", key="response_%d" % iteration_num,
                                 placeholder="Type your answer here...")

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            submit = st.button("Submit Answer →", type="primary")
        with col2:
            skip = st.button("Skip (no answer)")
        with col3:
            finish = st.button("Finish Early →")

        if finish:
            st.session_state["triage_done"] = True
            st.session_state["phase"] = "result"
            st.rerun()
            return

        if submit or skip:
            answer = "" if skip else response

            # Parse
            parsed = parser.parse(answer, current_question=question)
            elicited = question["elicits"]

            # Apply
            logs = apply_response(parsed, spreader, graph, elicited)

            # Register in gate
            gate.register_question(question)

            # Store iteration
            iteration_data = {
                "num": iteration_num,
                "question_id": question["id"],
                "question_text": question["text"],
                "response": answer,
                "is_yes": parsed.get("is_yes", False),
                "is_no": parsed.get("is_no", False),
                "detected": parsed.get("detected_symptoms", []),
                "negated": parsed.get("negated_symptoms", []),
                "parse_method": parsed.get("parse_method", "fast"),
                "latency": parsed.get("metadata", {}).get("_llm_latency_ms", 0),
                "logs": logs,
            }
            st.session_state["iterations"].append(iteration_data)

            # Update session state
            st.session_state["spreader_data"] = {
                "activations": dict(spreader.activations),
                "active": list(spreader.active_symptoms),
                "denied": list(spreader.negative_symptoms),
                "metadata": dict(spreader.metadata),
            }
            st.session_state["asked_ids"] = gate.asked_question_ids
            st.session_state["asked_elicits"] = gate.asked_elicits
            st.session_state["asked_categories"] = gate.asked_categories
            st.session_state["asked_texts"] = gate.asked_question_texts
            st.session_state["questions_log"].append({
                "id": question["id"], "text": question["text"],
                "response": answer, "parse_method": parsed.get("parse_method"),
            })

            # Update extra_info
            eid = st.session_state["extra_info_data"]
            eid["symptoms"] = list(set(list(eid.get("symptoms", [])) + parsed.get("detected_symptoms", [])))
            eid["negated"] = list(set(list(eid.get("negated", [])) + parsed.get("negated_symptoms", [])))
            st.session_state["extra_info_data"] = eid

            st.rerun()

    # ─── PHASE 4: Result + Doctor Review ───
    elif phase == "result":
        st.markdown("### Step 4: Triage Result + Doctor Review")
        st.markdown("**Patient:** %d years old, %s | **Chief complaint:** %s" % (
            st.session_state.get("patient_age", 0),
            st.session_state.get("patient_gender", "?"),
            st.session_state.get("initial_input", "?"),
        ))

        # Rebuild spreader
        spreader = ActivationSpreader(graph)
        sd = st.session_state["spreader_data"]
        spreader.activations = defaultdict(float, sd["activations"])
        spreader.active_symptoms = set(sd["active"])
        spreader.negative_symptoms = set(sd["denied"])
        spreader.metadata = dict(sd["metadata"])

        # Show all iterations in expandable sections
        with st.expander("📋 Full Session Log (%d iterations)" % len(st.session_state["iterations"])):
            for it in st.session_state["iterations"]:
                st.markdown("**Iter %d** (%s) — [%s%s]" % (
                    it["num"], it["question_id"], it["parse_method"],
                    ", %dms" % it["latency"] if it["latency"] else ""))
                st.markdown("Q: %s" % it["question_text"])
                st.markdown("A: *%s* → yes=%s no=%s syms=%s" % (
                    it["response"] or "(skipped)", it["is_yes"], it["is_no"], it["detected"]))
                st.markdown("---")

        # Triage result
        render_triage_result(spreader, graph)

        # Doctor review
        render_doctor_review(spreader, graph, gh, st.session_state.get("questions_log", []))


if __name__ == "__main__":
    main()