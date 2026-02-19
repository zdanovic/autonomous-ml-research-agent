"""
Comet-Swarm Dashboard - Streamlit UI for Agent Control

Run with: uv run streamlit run app/main.py
"""

import asyncio
import sys
import threading
from datetime import datetime
from pathlib import Path
from queue import Queue

# Add parent directory to path for imports when running via Streamlit
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from app.runner import AgentRunner, AgentStatus, Checkpoint, Phase, RunConfig

# Page config must be first Streamlit command
st.set_page_config(
    page_title="Comet-Swarm Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
    .log-info { color: #2196F3; }
    .log-warning { color: #FF9800; }
    .log-error { color: #F44336; }
    .log-success { color: #4CAF50; }
    .phase-complete { background-color: #4CAF50; color: white; padding: 8px; border-radius: 4px; }
    .phase-active { background-color: #2196F3; color: white; padding: 8px; border-radius: 4px; animation: pulse 1s infinite; }
    .phase-pending { background-color: #E0E0E0; color: #757575; padding: 8px; border-radius: 4px; }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "agent_status": AgentStatus.IDLE,
        "current_phase": None,
        "logs": [],
        "experiments": [],
        "selected_platform": "wundernn",
        "selected_mode": "debug",
        "data_path": "",
        "max_iterations": 10,
        "token_budget": 10.0,
        "checkpoints": [],
        "runner": None,
        "run_task": None,
        "message_queue": Queue(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_log(message: str, level: str = "info"):
    """Add a log entry."""
    st.session_state.logs.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "level": level,
        "message": message,
    })


def process_queue():
    """Process messages from the runner queue."""
    while not st.session_state.message_queue.empty():
        msg = st.session_state.message_queue.get()
        
        if msg["type"] == "log":
            add_log(msg["message"], msg["level"])
        elif msg["type"] == "phase":
            st.session_state.current_phase = msg["phase"]
        elif msg["type"] == "status":
            st.session_state.agent_status = msg["status"]
        elif msg["type"] == "checkpoint":
            st.session_state.checkpoints.append(msg["checkpoint"])
        elif msg["type"] == "experiment":
            st.session_state.experiments.append(msg["data"])


def sidebar():
    """Render sidebar with navigation."""
    with st.sidebar:
        # Logo and title
        st.markdown("## 🤖 Comet-Swarm")
        st.caption("Autonomous ML Competition Agent")
        
        st.divider()
        
        # Status indicator
        status = st.session_state.agent_status
        status_display = {
            AgentStatus.IDLE: ("🔵", "IDLE"),
            AgentStatus.RUNNING: ("🟢", "RUNNING"),
            AgentStatus.PAUSED: ("🟡", "PAUSED"),
            AgentStatus.ERROR: ("🔴", "ERROR"),
            AgentStatus.COMPLETED: ("✅", "COMPLETED"),
        }
        icon, text = status_display.get(status, ("⚪", str(status)))
        st.markdown(f"### Status: {icon} {text}")
        
        if st.session_state.current_phase:
            st.markdown(f"**Phase:** {st.session_state.current_phase.value if hasattr(st.session_state.current_phase, 'value') else st.session_state.current_phase}")
        
        st.divider()
        
        # Navigation
        st.subheader("Navigation")
        page = st.radio(
            "Go to",
            ["🏠 Dashboard", "⚙️ Configuration", "📊 Experiments", "📜 Logs"],
            label_visibility="collapsed",
        )
        
        st.divider()
        
        # Quick stats
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Experiments", len(st.session_state.experiments))
        with col2:
            st.metric("Logs", len(st.session_state.logs))
        
        st.divider()
        
        # Links
        st.markdown("**Links:**")
        st.page_link("https://www.comet.com", label="📊 Comet ML")
        st.page_link("https://www.comet.com/opik", label="🔍 Opik Traces")
        
        return page


def dashboard_page():
    """Main dashboard page."""
    st.title("🏠 Dashboard")
    
    # Process any pending messages
    process_queue()
    
    # Agent Control Section
    st.header("🎮 Agent Control")
    
    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
    
    with col1:
        platform = st.selectbox(
            "Platform",
            ["wundernn", "solafune", "kaggle"],
            key="platform_select",
            format_func=lambda x: {
                "wundernn": "🏦 Wundernn - LOB Predictorium",
                "solafune": "🏗️ Solafune - Construction Cost",
                "kaggle": "🌊 Kaggle - Urban Flood",
            }.get(x, x),
            disabled=st.session_state.agent_status == AgentStatus.RUNNING,
        )
        st.session_state.selected_platform = platform
    
    with col2:
        mode = st.selectbox(
            "Mode",
            ["debug", "test", "full"],
            key="mode_select",
            format_func=lambda x: {
                "debug": "🐛 Debug (1 iter, 1K samples)",
                "test": "🧪 Test (3 iter, 10K samples)",
                "full": "🚀 Full (10 iter, all data)",
            }.get(x, x),
            disabled=st.session_state.agent_status == AgentStatus.RUNNING,
        )
        st.session_state.selected_mode = mode
    
    with col3:
        st.write("")  # Spacing
        status = st.session_state.agent_status
        
        if status == AgentStatus.IDLE or status == AgentStatus.COMPLETED or status == AgentStatus.ERROR:
            if st.button("▶️ Start", type="primary", use_container_width=True):
                start_agent()
        elif status == AgentStatus.RUNNING:
            if st.button("⏸️ Pause", use_container_width=True):
                pause_agent()
        elif status == AgentStatus.PAUSED:
            if st.button("▶️ Resume", type="primary", use_container_width=True):
                resume_agent()
    
    with col4:
        st.write("")  # Spacing
        if status in [AgentStatus.RUNNING, AgentStatus.PAUSED]:
            if st.button("⏹️ Stop", use_container_width=True):
                stop_agent()
    
    st.divider()
    
    # Progress Section
    st.header("📈 Progress")
    
    phases = [Phase.EDA, Phase.STRATEGY, Phase.FEATURES, Phase.TRAINING, Phase.EVALUATION, Phase.SUBMIT]
    current_phase = st.session_state.current_phase
    
    # Find current index
    current_idx = -1
    if current_phase:
        for i, p in enumerate(phases):
            if p == current_phase or (hasattr(current_phase, 'value') and p.value == current_phase.value):
                current_idx = i
                break
    
    cols = st.columns(len(phases))
    for i, (col, phase) in enumerate(zip(cols, phases)):
        with col:
            if i < current_idx:
                st.markdown(f"<div class='phase-complete'>✅ {phase.value}</div>", unsafe_allow_html=True)
            elif i == current_idx:
                st.markdown(f"<div class='phase-active'>🔄 {phase.value}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='phase-pending'>⬜ {phase.value}</div>", unsafe_allow_html=True)
    
    st.divider()
    
    # Two-column layout
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("📋 Recent Logs")
        logs_container = st.container(height=300)
        with logs_container:
            for log in reversed(st.session_state.logs[-15:]):
                level_icon = {
                    "info": "ℹ️",
                    "warning": "⚠️",
                    "error": "❌",
                    "success": "✅"
                }.get(log["level"], "•")
                st.text(f"{level_icon} [{log['timestamp']}] {log['message']}")
    
    with col_right:
        st.subheader("🛑 Checkpoints")
        
        if st.session_state.checkpoints:
            for i, cp in enumerate(st.session_state.checkpoints):
                with st.container(border=True):
                    st.markdown(f"**{cp.title}**")
                    st.write(cp.description)
                    
                    selected = st.radio(
                        "Select option",
                        cp.options,
                        key=f"cp_radio_{cp.id}",
                        horizontal=True,
                    )
                    
                    if st.button("✓ Confirm", key=f"cp_btn_{cp.id}", type="primary"):
                        handle_checkpoint_response(cp.id, selected)
        else:
            st.info("No pending checkpoints. Agent will request input when needed.")
    
    # Auto-refresh when running
    if st.session_state.agent_status == AgentStatus.RUNNING:
        st.empty()  # Force rerun check


def config_page():
    """Configuration page."""
    st.title("⚙️ Configuration")
    
    tab1, tab2, tab3 = st.tabs(["🔑 API Keys", "🧠 Agent Settings", "📁 Data"])
    
    with tab1:
        st.subheader("API Keys Status")
        st.info("API keys are loaded from `.env` file. Update the file and restart to apply changes.")
        
        try:
            from comet_swarm.config import get_settings
            settings = get_settings()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**LLM Provider:**")
                if settings.openrouter_api_key:
                    st.success("✅ OpenRouter configured")
                elif settings.openai_api_key:
                    st.success("✅ OpenAI configured")
                elif settings.anthropic_api_key:
                    st.success("✅ Anthropic configured")
                else:
                    st.error("❌ No LLM API key")
                
                st.markdown("**Comet:**")
                if settings.comet_api_key:
                    st.success("✅ Comet ML / Opik configured")
                else:
                    st.error("❌ Comet API key not set")
            
            with col2:
                st.markdown("**E2B Sandbox:**")
                if settings.e2b_api_key:
                    st.success("✅ E2B configured")
                else:
                    st.warning("⚠️ E2B not set (local fallback)")
                
                st.markdown("**Kaggle:**")
                if settings.kaggle_key:
                    st.success("✅ Kaggle configured")
                else:
                    st.info("ℹ️ Kaggle not set (optional)")
                    
        except Exception as e:
            st.error(f"Error loading settings: {e}")
            st.code(str(e))
    
    with tab2:
        st.subheader("Agent Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.max_iterations = st.slider(
                "Max Iterations",
                min_value=1,
                max_value=20,
                value=st.session_state.max_iterations,
                help="Maximum number of strategy iterations",
            )
            
            st.session_state.token_budget = st.slider(
                "Token Budget (USD)",
                min_value=1.0,
                max_value=50.0,
                value=st.session_state.token_budget,
                step=1.0,
                help="Maximum LLM token spending",
            )
        
        with col2:
            st.selectbox(
                "Premium Model",
                ["openai/gpt-4o", "anthropic/claude-3.5-sonnet"],
                help="Used for strategy and complex tasks",
            )
            
            st.selectbox(
                "Standard Model",
                ["openai/gpt-4o-mini", "anthropic/claude-3-haiku"],
                help="Used for code generation",
            )
    
    with tab3:
        st.subheader("Data Configuration")
        
        st.session_state.data_path = st.text_input(
            "Custom Data Path (optional)",
            value=st.session_state.data_path,
            placeholder="/path/to/train.csv or train.parquet",
            help="Override default data location",
        )
        
        if st.session_state.data_path:
            path = Path(st.session_state.data_path)
            if path.exists():
                st.success(f"✅ Found: {path.name} ({path.stat().st_size / 1024 / 1024:.1f} MB)")
            else:
                st.warning("⚠️ File not found")
        
        st.divider()
        
        st.markdown("**Data Directories:**")
        for platform in ["wundernn", "solafune", "kaggle"]:
            data_dir = Path(f"./data/{platform}")
            if data_dir.exists():
                files = list(data_dir.glob("*"))
                st.success(f"📁 {platform}: {len(files)} files")
            else:
                st.info(f"📁 {platform}: Not downloaded")


def experiments_page():
    """Experiments viewer page."""
    st.title("📊 Experiments")
    
    # Process queue
    process_queue()
    
    if not st.session_state.experiments:
        st.info("No experiments yet. Start an agent run to create experiments.")
        return
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Experiments", len(st.session_state.experiments))
    with col2:
        best_score = max(e.get("cv_mean", 0) for e in st.session_state.experiments)
        st.metric("Best CV Score", f"{best_score:.4f}")
    with col3:
        total_features = sum(e.get("features", 0) for e in st.session_state.experiments)
        st.metric("Total Features Tried", total_features)
    
    st.divider()
    
    # Experiments list
    for exp in reversed(st.session_state.experiments):
        with st.expander(f"🧪 {exp.get('name', 'Unknown')}", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("CV Mean", f"{exp.get('cv_mean', 0):.4f}")
            with col2:
                st.metric("CV Std", f"±{exp.get('cv_std', 0):.4f}")
            with col3:
                st.metric("Features", exp.get('features', 0))
            with col4:
                if exp.get("url"):
                    st.link_button("View in Comet", exp["url"])


def logs_page():
    """Full logs viewer page."""
    st.title("📜 Logs")
    
    # Process queue
    process_queue()
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        filter_level = st.multiselect(
            "Filter by level",
            ["info", "warning", "error", "success"],
            default=["info", "warning", "error", "success"],
        )
    with col2:
        if st.button("🔄 Refresh"):
            st.rerun()
    with col3:
        if st.button("🗑️ Clear Logs"):
            st.session_state.logs = []
            st.rerun()
    
    st.divider()
    
    filtered_logs = [l for l in st.session_state.logs if l["level"] in filter_level]
    
    if not filtered_logs:
        st.info("No logs to display")
        return
    
    for log in reversed(filtered_logs):
        level_icon = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "success": "✅"
        }.get(log["level"], "•")
        
        level_color = {
            "info": "#2196F3",
            "warning": "#FF9800",
            "error": "#F44336",
            "success": "#4CAF50",
        }.get(log["level"], "#757575")
        
        st.markdown(
            f"{level_icon} <span style='color:{level_color}'>[{log['level'].upper()}]</span> "
            f"<span style='color:#888'>{log['timestamp']}</span> {log['message']}",
            unsafe_allow_html=True,
        )


# Agent control functions
def create_runner_callbacks():
    """Create callbacks that put messages in the queue."""
    queue = st.session_state.message_queue
    
    def on_log(message: str, level: str):
        queue.put({"type": "log", "message": message, "level": level})
    
    def on_phase_change(phase: Phase):
        queue.put({"type": "phase", "phase": phase})
    
    def on_status_change(status: AgentStatus):
        queue.put({"type": "status", "status": status})
    
    def on_checkpoint(checkpoint: Checkpoint):
        queue.put({"type": "checkpoint", "checkpoint": checkpoint})
    
    def on_experiment(data: dict):
        queue.put({"type": "experiment", "data": data})
    
    return {
        "on_log": on_log,
        "on_phase_change": on_phase_change,
        "on_status_change": on_status_change,
        "on_checkpoint": on_checkpoint,
        "on_experiment": on_experiment,
    }


def run_agent_async(runner: AgentRunner, config: RunConfig):
    """Run agent in a separate thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(runner.run(config))
    finally:
        loop.close()


def start_agent():
    """Start the agent."""
    st.session_state.agent_status = AgentStatus.RUNNING
    st.session_state.current_phase = Phase.EDA
    st.session_state.checkpoints = []
    
    # Create runner with callbacks
    callbacks = create_runner_callbacks()
    runner = AgentRunner(**callbacks)
    st.session_state.runner = runner
    
    # Create config
    config = RunConfig(
        platform=st.session_state.selected_platform,
        mode=st.session_state.selected_mode,
        data_path=st.session_state.data_path if st.session_state.data_path else None,
        max_iterations=st.session_state.max_iterations,
        token_budget=st.session_state.token_budget,
    )
    
    add_log(f"Starting: {config.platform} / {config.mode}", "success")
    
    # Start in background thread
    thread = threading.Thread(target=run_agent_async, args=(runner, config), daemon=True)
    thread.start()
    
    st.rerun()


def pause_agent():
    """Pause the agent."""
    if st.session_state.runner:
        st.session_state.runner.pause()
    st.session_state.agent_status = AgentStatus.PAUSED
    add_log("Agent paused", "warning")
    st.rerun()


def resume_agent():
    """Resume the agent."""
    if st.session_state.runner:
        st.session_state.runner.resume()
    st.session_state.agent_status = AgentStatus.RUNNING
    add_log("Agent resumed", "info")
    st.rerun()


def stop_agent():
    """Stop the agent."""
    if st.session_state.runner:
        st.session_state.runner.stop()
    st.session_state.agent_status = AgentStatus.IDLE
    st.session_state.current_phase = None
    add_log("Agent stopped", "warning")
    st.rerun()


def handle_checkpoint_response(checkpoint_id: str, response: str):
    """Handle user response to a checkpoint."""
    if st.session_state.runner:
        st.session_state.runner.respond_to_checkpoint(checkpoint_id, response)
    
    # Remove from pending
    st.session_state.checkpoints = [
        cp for cp in st.session_state.checkpoints if cp.id != checkpoint_id
    ]
    
    add_log(f"Checkpoint resolved: {response}", "success")
    st.rerun()


def main():
    """Main application entry point."""
    init_session_state()
    
    # Render sidebar and get selected page
    page = sidebar()
    
    # Render selected page
    if page == "🏠 Dashboard":
        dashboard_page()
    elif page == "⚙️ Configuration":
        config_page()
    elif page == "📊 Experiments":
        experiments_page()
    elif page == "📜 Logs":
        logs_page()


if __name__ == "__main__":
    main()
