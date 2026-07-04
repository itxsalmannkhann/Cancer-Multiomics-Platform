"""Centralized UI theme for the Cancer Multi-Omics Intelligence Platform.

This module holds the single source of truth for the app's visual language:
a clinical color palette, a shared Plotly template, global CSS injection, and
reusable layout components (page headers, metric cards, section titles,
status pills). Pages import from here so the look stays consistent and the
backend data-loading logic in ``data_utils.py`` is never touched.
"""
from __future__ import annotations

from typing import Iterable

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# --------------------------------------------------------------------------- #
# Clinical color palette
# --------------------------------------------------------------------------- #
PALETTE = {
    "primary": "#0E7C86",       # deep clinical teal
    "primary_light": "#14B8C4",  # bright teal accent
    "ink": "#0F2A43",           # deep slate for headings
    "text": "#1F3A56",          # body text
    "muted": "#5B7189",         # secondary text
    "bg": "#EEF3F8",            # app background
    "surface": "#FFFFFF",       # cards / panels
    "border": "#DCE6F0",        # hairline borders
    "healthy": "#1FA971",       # positive / alive / low-risk
    "warning": "#E8A317",       # caution
    "danger": "#E5484D",        # negative / dead / high-risk
    "violet": "#6D5AE6",        # tertiary accent
}

# Ordered categorical sequence for charts (muted, clinical, high-contrast).
CHART_SEQUENCE = [
    "#0E7C86", "#14B8C4", "#6D5AE6", "#E8A317",
    "#1FA971", "#E5484D", "#3B82F6", "#9333EA",
]

# Continuous scale used for bars/heat gradients.
CHART_CONTINUOUS = [
    [0.0, "#D6EEF0"],
    [0.5, "#3FA9B4"],
    [1.0, "#0E5A62"],
]

FONT_STACK = "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif"


# --------------------------------------------------------------------------- #
# Plotly template
# --------------------------------------------------------------------------- #
def _register_plotly_template() -> None:
    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family=FONT_STACK, size=13, color=PALETTE["text"]),
        colorway=CHART_SEQUENCE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(font=dict(size=16, color=PALETTE["ink"]), x=0.01, xanchor="left"),
        margin=dict(l=10, r=10, t=50, b=10),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor=PALETTE["border"],
            font=dict(family=FONT_STACK, size=12, color=PALETTE["text"]),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=12, color=PALETTE["muted"]),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis=dict(
            gridcolor="#E7EDF4",
            zerolinecolor="#E7EDF4",
            linecolor=PALETTE["border"],
            tickfont=dict(size=12, color=PALETTE["muted"]),
        ),
        yaxis=dict(
            gridcolor="#E7EDF4",
            zerolinecolor="#E7EDF4",
            linecolor=PALETTE["border"],
            tickfont=dict(size=12, color=PALETTE["muted"]),
        ),
        colorscale=dict(sequential=CHART_CONTINUOUS),
    )
    pio.templates["clinical"] = template
    pio.templates.default = "plotly+clinical"


def style_fig(fig: go.Figure, height: int | None = None) -> go.Figure:
    """Apply the clinical template to a figure and normalize sizing."""
    fig.update_layout(template="plotly+clinical")
    if height is not None:
        fig.update_layout(height=height)
    return fig


# --------------------------------------------------------------------------- #
# Global CSS
# --------------------------------------------------------------------------- #
_GLOBAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {{
    --primary: {PALETTE['primary']};
    --primary-light: {PALETTE['primary_light']};
    --ink: {PALETTE['ink']};
    --text: {PALETTE['text']};
    --muted: {PALETTE['muted']};
    --bg: {PALETTE['bg']};
    --surface: {PALETTE['surface']};
    --border: {PALETTE['border']};
    --healthy: {PALETTE['healthy']};
    --warning: {PALETTE['warning']};
    --danger: {PALETTE['danger']};
}}

html, body, [class*="css"], .stApp {{
    font-family: {FONT_STACK};
    color: var(--text);
}}

.stApp {{
    background:
        radial-gradient(1100px 480px at 88% -8%, rgba(20,184,196,0.10), transparent 60%),
        radial-gradient(900px 420px at -5% 0%, rgba(14,124,134,0.08), transparent 55%),
        var(--bg);
}}

/* Constrain + pad the main content area */
.block-container {{
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}}

/* Headings */
h1, h2, h3, h4 {{ color: var(--ink); font-weight: 700; letter-spacing: -0.01em; }}
h1 {{ font-weight: 800; }}

/* Hide default Streamlit chrome for a cleaner product feel */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
[data-testid="stDecoration"] {{ display: none; }}

/* -------- Sidebar -------- */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0B2C3A 0%, #0E3D48 55%, #103F4B 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}}
[data-testid="stSidebar"] * {{ color: #DCEBEF; }}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
    color: #FFFFFF;
}}
[data-testid="stSidebarNav"] a {{
    border-radius: 10px;
    margin: 2px 8px;
    padding: 6px 10px;
}}
[data-testid="stSidebarNav"] a:hover {{ background: rgba(255,255,255,0.08); }}

/* Make sidebar a full-height flex column so credits can pin to the bottom */
[data-testid="stSidebarUserContent"] {{
    display: flex;
    flex-direction: column;
    min-height: calc(100vh - 220px);
}}

/* -------- Sidebar credits -------- */
.sidebar-credits {{
    margin-top: auto;
    padding: 16px 16px 8px 16px;
    border-top: 1px solid rgba(255,255,255,0.12);
    line-height: 1.5;
}}
.sidebar-credits .sc-badge {{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 10px; border-radius: 999px;
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.14);
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase; color: #BFE3E8 !important;
    margin-bottom: 10px;
}}
.sidebar-credits .sc-built {{
    font-size: 0.82rem; color: #CFE6EA !important; margin-bottom: 2px;
}}
.sidebar-credits .sc-name {{
    font-size: 0.92rem; font-weight: 700; color: #FFFFFF !important;
}}
.sidebar-credits .sc-event {{
    font-size: 0.78rem; color: #A9CDD3 !important; margin-top: 8px;
}}
.sidebar-credits .sc-event strong {{ color: #FFFFFF !important; font-weight: 700; }}
.sidebar-credits .sc-collab {{
    font-size: 0.74rem; color: #9FC3C9 !important; margin-top: 8px;
}}
.sidebar-credits .sc-lab {{ color: #7FE3EC !important; font-weight: 700; }}
.sidebar-credits .sc-disclaimer {{
    font-size: 0.68rem; color: #7C9BA2 !important; margin-top: 10px;
    font-style: italic;
}}

/* -------- Hero header -------- */
.hero {{
    background: linear-gradient(120deg, #0E7C86 0%, #12A0AD 55%, #14B8C4 100%);
    border-radius: 20px;
    margin-top: 34px;
    padding: 26px 30px;
    color: #FFFFFF;
    box-shadow: 0 18px 40px -18px rgba(14,124,134,0.55);
    margin-bottom: 22px;
    position: relative;
    overflow: hidden;
}}
.hero::after {{
    content: "";
    position: absolute; right: -40px; top: -60px;
    width: 260px; height: 260px; border-radius: 50%;
    background: radial-gradient(circle, rgba(255,255,255,0.18), transparent 70%);
}}
.hero-eyebrow {{
    text-transform: uppercase; letter-spacing: 0.18em; font-size: 0.72rem;
    font-weight: 600; opacity: 0.85; margin-bottom: 6px;
}}
.hero-title {{ font-size: 1.9rem; font-weight: 800; margin: 0; line-height: 1.15; }}
.hero-sub {{ font-size: 1rem; opacity: 0.92; margin-top: 8px; max-width: 760px; }}
.hero-icon {{ font-size: 2.4rem; line-height: 1; }}

/* -------- Section title -------- */
.section-title {{
    display: flex; align-items: center; gap: 10px;
    font-size: 1.18rem; font-weight: 700; color: var(--ink);
    margin: 6px 0 4px 0;
}}
.section-title .bar {{
    width: 5px; height: 22px; border-radius: 4px;
    background: linear-gradient(180deg, var(--primary), var(--primary-light));
}}
.section-sub {{ color: var(--muted); font-size: 0.9rem; margin: 0 0 6px 15px; }}

/* -------- Metric cards (st.metric restyle) -------- */
[data-testid="stMetric"] {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 16px 18px;
    box-shadow: 0 10px 24px -20px rgba(15,42,67,0.45);
    transition: transform .15s ease, box-shadow .15s ease;
}}
[data-testid="stMetric"]:hover {{
    transform: translateY(-2px);
    box-shadow: 0 16px 30px -20px rgba(14,124,134,0.6);
}}
[data-testid="stMetricLabel"] p {{
    color: var(--muted); font-weight: 600; font-size: 0.82rem;
    text-transform: uppercase; letter-spacing: 0.04em;
}}
[data-testid="stMetricValue"] {{ color: var(--ink); font-weight: 800; }}

/* -------- Generic card container (st.container(border=True)) -------- */
[data-testid="stVerticalBlockBorderWrapper"] {{
    background: var(--surface);
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
    box-shadow: 0 12px 28px -24px rgba(15,42,67,0.5);
}}

/* -------- Tabs -------- */
.stTabs [data-baseweb="tab-list"] {{
    gap: 6px;
    background: rgba(255,255,255,0.6);
    padding: 6px;
    border-radius: 14px;
    border: 1px solid var(--border);
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 10px;
    padding: 8px 16px;
    font-weight: 600;
    color: var(--muted);
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(120deg, var(--primary), var(--primary-light)) !important;
    color: #FFFFFF !important;
}}

/* -------- Buttons -------- */
.stButton > button, .stDownloadButton > button {{
    border-radius: 10px;
    border: 1px solid var(--primary);
    background: linear-gradient(120deg, var(--primary), var(--primary-light));
    color: #fff;
    font-weight: 600;
    padding: 0.5rem 1.1rem;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    filter: brightness(1.05);
    border-color: var(--primary);
    color: #fff;
}}

/* -------- Inputs -------- */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-baseweb="input"] > div {{
    border-radius: 10px !important;
}}

/* -------- Dataframe -------- */
[data-testid="stDataFrame"] {{
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid var(--border);
}}

/* -------- Status pill -------- */
.pill {{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 999px;
    font-size: 0.8rem; font-weight: 600;
}}
.pill-ok {{ background: rgba(31,169,113,0.12); color: #0E7A50; }}
.pill-warn {{ background: rgba(232,163,23,0.14); color: #9A6B08; }}
.pill-bad {{ background: rgba(229,72,77,0.12); color: #B02328; }}
.pill-info {{ background: rgba(14,124,134,0.12); color: #0B5A62; }}

/* -------- Info/alert restyle -------- */
[data-testid="stAlert"] {{ border-radius: 14px; }}

/* -------- Footer note -------- */
.app-footer {{
    color: var(--muted); font-size: 0.82rem; text-align: center;
    margin-top: 26px; padding-top: 14px; border-top: 1px solid var(--border);
}}
</style>
"""


def sidebar_credits() -> None:
    """Render the project credits pinned to the bottom of the sidebar."""
    st.sidebar.markdown(
        """
        <div class="sidebar-credits">
            <div class="sc-badge">🧬 Multi-Omics Platform</div>
            <div class="sc-built">Built by</div>
            <div class="sc-name">Salman Khan &amp; Team</div>
            <div class="sc-event">
                For <strong>NeoHack — GIKI Edition 2025</strong>
            </div>
            <div class="sc-collab">
                In collaboration with<br>
                <span class="sc-lab">Precision Medicine Lab</span><br>
                Real-world healthcare data
            </div>
            <div class="sc-disclaimer">
                Research preview — not for clinical use.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_base(page_title: str, page_icon: str = "🧬") -> None:
    """Set page config + inject theme. Call once at the top of every page."""
    st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide")
    _register_plotly_template()
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)
    sidebar_credits()


# --------------------------------------------------------------------------- #
# Reusable components
# --------------------------------------------------------------------------- #
def hero(title: str, subtitle: str, eyebrow: str = "Cancer Multi-Omics Intelligence", icon: str = "🧬") -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div style="display:flex; align-items:center; gap:18px;">
                <div class="hero-icon">{icon}</div>
                <div>
                    <div class="hero-eyebrow">{eyebrow}</div>
                    <div class="hero-title">{title}</div>
                    <div class="hero-sub">{subtitle}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, subtitle: str | None = None, icon: str = "") -> None:
    icon_html = f"<span>{icon}</span>" if icon else ""
    st.markdown(
        f'<div class="section-title"><span class="bar"></span>{icon_html}{title}</div>',
        unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)


def pill(text: str, kind: str = "info") -> str:
    """Return HTML for a status pill. kind: ok | warn | bad | info."""
    return f'<span class="pill pill-{kind}">{text}</span>'


def footer(text: str) -> None:
    st.markdown(f'<div class="app-footer">{text}</div>', unsafe_allow_html=True)
