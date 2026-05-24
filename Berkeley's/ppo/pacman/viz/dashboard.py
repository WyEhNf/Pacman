"""Web metrics dashboard using Dash + Plotly, reading TensorBoard event files."""

from pathlib import Path
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _load_scalars(log_dir: str | Path, tag: str) -> tuple[list[int], list[float]]:
    """Load a scalar time-series from TensorBoard event files.

    Returns (steps, values) lists.
    """
    ea = EventAccumulator(str(log_dir))
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return [], []
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def _rolling_average(values: list[float], window: int = 50) -> list[float]:
    """Compute a rolling average over a list of values."""
    if len(values) < window:
        window = max(len(values), 1)
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(sum(values[start:i + 1]) / (i - start + 1))
    return result


def _make_empty_fig(message: str = "No data yet") -> go.Figure:
    """Create an empty dark-themed figure with a message."""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        annotations=[dict(text=message, showarrow=False,
                          xref="paper", yref="paper", x=0.5, y=0.5,
                          font=dict(size=16, color="#888"))],
    )
    return fig


def create_dashboard(log_dir: str | Path) -> Dash:
    """Create and configure the Dash web dashboard.

    Args:
        log_dir: Path to the TensorBoard log directory
                 (e.g., 'runs/2024-01-01_12-00-00/tensorboard/').
    """
    log_dir = str(log_dir)
    app = Dash(__name__)

    app.layout = html.Div([
        html.H1("Pac-Man AI Training Dashboard",
                style={"textAlign": "center", "color": "#FFD700",
                       "backgroundColor": "#1a1a2e", "padding": "20px",
                       "margin": "0", "fontFamily": "monospace"}),

        dcc.Interval(id="refresh", interval=10_000, n_intervals=0),

        html.Div([
            # Summary stats bar
            html.Div(id="summary-stats",
                     style={"padding": "10px", "backgroundColor": "#16213e",
                            "borderRadius": "8px", "margin": "10px",
                            "textAlign": "center"}),

            # Row 1: Score + Level clear rate
            html.Div([
                html.Div([dcc.Graph(id="score-chart")],
                         style={"width": "48%", "display": "inline-block",
                                "verticalAlign": "top"}),
                html.Div([dcc.Graph(id="clear-rate-chart")],
                         style={"width": "48%", "display": "inline-block",
                                "verticalAlign": "top"}),
            ]),

            # Row 2: Loss curves + Episode length
            html.Div([
                html.Div([dcc.Graph(id="loss-chart")],
                         style={"width": "48%", "display": "inline-block",
                                "verticalAlign": "top"}),
                html.Div([dcc.Graph(id="episode-length-chart")],
                         style={"width": "48%", "display": "inline-block",
                                "verticalAlign": "top"}),
            ]),

            # Row 3: Throughput + Ghosts eaten
            html.Div([
                html.Div([dcc.Graph(id="throughput-chart")],
                         style={"width": "48%", "display": "inline-block",
                                "verticalAlign": "top"}),
                html.Div([dcc.Graph(id="ghost-kills-chart")],
                         style={"width": "48%", "display": "inline-block",
                                "verticalAlign": "top"}),
            ]),

            # --- World Model Section ---
            html.H2("World Model Training",
                     style={"textAlign": "center", "color": "#BB86FC",
                            "fontFamily": "monospace", "marginTop": "30px"}),

            # Row 4: WM Reconstruction + Reward prediction
            html.Div([
                html.Div([dcc.Graph(id="wm-recon-chart")],
                         style={"width": "48%", "display": "inline-block",
                                "verticalAlign": "top"}),
                html.Div([dcc.Graph(id="wm-reward-chart")],
                         style={"width": "48%", "display": "inline-block",
                                "verticalAlign": "top"}),
            ]),

            # Row 5: KL divergence + Dream agent score
            html.Div([
                html.Div([dcc.Graph(id="wm-kl-chart")],
                         style={"width": "48%", "display": "inline-block",
                                "verticalAlign": "top"}),
                html.Div([dcc.Graph(id="dream-score-chart")],
                         style={"width": "48%", "display": "inline-block",
                                "verticalAlign": "top"}),
            ]),

        ], style={"backgroundColor": "#0a0a1a", "minHeight": "100vh",
                  "padding": "10px"}),
    ], style={"backgroundColor": "#0a0a1a"})

    @app.callback(
        [Output("summary-stats", "children"),
         Output("score-chart", "figure"),
         Output("clear-rate-chart", "figure"),
         Output("loss-chart", "figure"),
         Output("episode-length-chart", "figure"),
         Output("throughput-chart", "figure"),
         Output("ghost-kills-chart", "figure"),
         Output("wm-recon-chart", "figure"),
         Output("wm-reward-chart", "figure"),
         Output("wm-kl-chart", "figure"),
         Output("dream-score-chart", "figure")],
        [Input("refresh", "n_intervals")]
    )
    def update_charts(_):
        empty = _make_empty_fig()

        try:
            ea = EventAccumulator(log_dir)
            ea.Reload()
            available_tags = ea.Tags().get("scalars", [])
        except Exception as e:
            msg = f"Error loading logs: {e}"
            return msg, empty, empty, empty, empty, empty, empty, empty, empty, empty, empty

        if not available_tags:
            msg = "Waiting for training data..."
            return msg, empty, empty, empty, empty, empty, empty, empty, empty, empty, empty

        # --- Summary stats ---
        summary_parts = []
        for tag, label, fmt in [
            ("eval/mean_score", "Avg Score", "{:.0f}"),
            ("eval/level_clear_rate", "Clear Rate", "{:.1%}"),
            ("eval/mean_steps", "Avg Steps", "{:.0f}"),
        ]:
            steps, vals = _load_scalars(log_dir, tag)
            if vals:
                text = f"{label}: {fmt.format(vals[-1])}"
                summary_parts.append(
                    html.Span(text + "  |  ",
                              style={"color": "#fff", "fontSize": "16px",
                                     "fontFamily": "monospace"})
                )
        summary = html.Div(summary_parts) if summary_parts else "Waiting for eval data..."

        # --- Score progression ---
        score_fig = go.Figure()
        steps, vals = _load_scalars(log_dir, "eval/mean_score")
        if vals:
            score_fig.add_trace(go.Scatter(
                x=steps, y=vals, name="Mean Score",
                line=dict(color="#4caf50", width=1), opacity=0.4,
            ))
            rolling = _rolling_average(vals, 10)
            score_fig.add_trace(go.Scatter(
                x=steps, y=rolling, name="Rolling Avg",
                line=dict(color="#4caf50", width=2),
            ))
        score_fig.update_layout(title="Score Progression", template="plotly_dark",
                                xaxis_title="Update", yaxis_title="Score")

        # --- Level clear rate ---
        clear_fig = go.Figure()
        steps, vals = _load_scalars(log_dir, "eval/level_clear_rate")
        if vals:
            clear_fig.add_trace(go.Scatter(
                x=steps, y=vals, name="Clear Rate",
                line=dict(color="#FFD700", width=2),
            ))
        clear_fig.update_layout(title="Level Clear Rate", template="plotly_dark",
                                xaxis_title="Update", yaxis_title="Rate",
                                yaxis=dict(range=[0, 1]))

        # --- Loss curves (policy, value, entropy) ---
        loss_fig = go.Figure()
        loss_tags = [
            ("train/policy_loss", "Policy Loss", "#ff6b6b"),
            ("train/value_loss", "Value Loss", "#64b5f6"),
            ("train/entropy", "Entropy", "#81c784"),
        ]
        for tag, name, color in loss_tags:
            steps, vals = _load_scalars(log_dir, tag)
            if vals:
                loss_fig.add_trace(go.Scatter(
                    x=steps, y=vals, name=name,
                    line=dict(color=color, width=1),
                ))
        loss_fig.update_layout(title="Loss Curves", template="plotly_dark",
                               xaxis_title="Update", yaxis_title="Loss")

        # --- Episode length ---
        ep_fig = go.Figure()
        steps, vals = _load_scalars(log_dir, "eval/mean_steps")
        if vals:
            ep_fig.add_trace(go.Scatter(
                x=steps, y=vals, name="Mean Steps",
                line=dict(color="#ce93d8", width=2),
            ))
        ep_fig.update_layout(title="Episode Length", template="plotly_dark",
                             xaxis_title="Update", yaxis_title="Steps")

        # --- Training throughput (SPS) ---
        tp_fig = go.Figure()
        steps, vals = _load_scalars(log_dir, "train/sps")
        if vals:
            tp_fig.add_trace(go.Scatter(
                x=steps, y=vals, name="Steps/sec",
                line=dict(color="#4dd0e1", width=2),
            ))
        tp_fig.update_layout(title="Training Throughput", template="plotly_dark",
                             xaxis_title="Update", yaxis_title="SPS")

        # --- Ghost kills ---
        gk_fig = go.Figure()
        steps, vals = _load_scalars(log_dir, "eval/mean_ghosts_eaten")
        if vals:
            gk_fig.add_trace(go.Scatter(
                x=steps, y=vals, name="Ghosts Eaten",
                line=dict(color="#ff8a65", width=2),
            ))
        gk_fig.update_layout(title="Ghost Kills (Mean per Episode)",
                             template="plotly_dark",
                             xaxis_title="Update", yaxis_title="Ghosts Eaten")

        # --- World Model: Reconstruction loss ---
        wm_recon_fig = go.Figure()
        steps, vals = _load_scalars(log_dir, "world_model/recon_loss")
        if vals:
            wm_recon_fig.add_trace(go.Scatter(
                x=steps, y=vals, name="Recon Loss",
                line=dict(color="#BB86FC", width=1), opacity=0.4,
            ))
            rolling = _rolling_average(vals, 50)
            wm_recon_fig.add_trace(go.Scatter(
                x=steps, y=rolling, name="Rolling Avg",
                line=dict(color="#BB86FC", width=2),
            ))
        wm_recon_fig.update_layout(title="WM Reconstruction Loss",
                                    template="plotly_dark",
                                    xaxis_title="Step", yaxis_title="MSE")

        # --- World Model: Reward prediction loss ---
        wm_reward_fig = go.Figure()
        steps, vals = _load_scalars(log_dir, "world_model/reward_loss")
        if vals:
            wm_reward_fig.add_trace(go.Scatter(
                x=steps, y=vals, name="Reward Loss",
                line=dict(color="#03DAC6", width=1), opacity=0.4,
            ))
            rolling = _rolling_average(vals, 50)
            wm_reward_fig.add_trace(go.Scatter(
                x=steps, y=rolling, name="Rolling Avg",
                line=dict(color="#03DAC6", width=2),
            ))
        wm_reward_fig.update_layout(title="WM Reward Prediction Loss",
                                     template="plotly_dark",
                                     xaxis_title="Step", yaxis_title="MSE")

        # --- World Model: KL divergence ---
        wm_kl_fig = go.Figure()
        steps, vals = _load_scalars(log_dir, "world_model/kl")
        if vals:
            wm_kl_fig.add_trace(go.Scatter(
                x=steps, y=vals, name="KL Divergence",
                line=dict(color="#CF6679", width=1), opacity=0.4,
            ))
            rolling = _rolling_average(vals, 50)
            wm_kl_fig.add_trace(go.Scatter(
                x=steps, y=rolling, name="Rolling Avg",
                line=dict(color="#CF6679", width=2),
            ))
        wm_kl_fig.update_layout(title="WM KL Divergence",
                                 template="plotly_dark",
                                 xaxis_title="Step", yaxis_title="KL (nats)")

        # --- Dream Agent: Score vs Real Agent ---
        dream_fig = go.Figure()
        steps, vals = _load_scalars(log_dir, "dream/mean_score")
        if vals:
            dream_fig.add_trace(go.Scatter(
                x=steps, y=vals, name="Dream Agent",
                line=dict(color="#BB86FC", width=2),
            ))
        steps, vals = _load_scalars(log_dir, "eval/mean_score")
        if vals:
            dream_fig.add_trace(go.Scatter(
                x=steps, y=[vals[-1]] * len(steps), name="PPO Agent (best)",
                line=dict(color="#4caf50", width=2, dash="dash"),
            ))
        dream_fig.update_layout(title="Dream Agent vs Real Agent Score",
                                 template="plotly_dark",
                                 xaxis_title="Update", yaxis_title="Score")

        return (summary, score_fig, clear_fig, loss_fig, ep_fig, tp_fig, gk_fig,
                wm_recon_fig, wm_reward_fig, wm_kl_fig, dream_fig)

    return app
