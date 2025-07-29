# ──────────────────────────────────────────────────────────────
#  Cross‑Check Framework  •  Streamlit Simulation App
# ──────────────────────────────────────────────────────────────
#  (c) 2025 digitalian  –  MIT License
# ----------------------------------------------------------------
#  Major fixes
#   • BUG‑1: partial derivative dE/dℓ – missing cost factor C
#   • BUG‑4: σℓ and C_loss distribution – must include C·ℓ term
# ----------------------------------------------------------------

from __future__ import annotations

import streamlit as st
import numpy as np
import sympy as sp
import pandas as pd
import plotly.express as px
from typing import Tuple, Dict

# ══════════════════════════════════════════════════════════════
#  Localisation helpers
# ══════════════════════════════════════════════════════════════
def get_text_labels(lang: str) -> Dict:
    LANG = "EN" if lang == "English" else "JA"
    return TXT_ALL[LANG]


# ══════════════════════════════════════════════════════════════
#  Streamlit Session‑State helpers
# ══════════════════════════════════════════════════════════════
def get_state(key, default):
    """Retrieve or initialise a value in `st.session_state`."""
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def exp(path: str):
    """
    Create an expander from nested TXT dict using a dot‑separated
    key path (e.g. 'charts.tornado.expander_title').
    """
    keys = path.split(".")
    title = TXT
    for k in keys:
        title = title[k]

    content_keys = keys[:-1] + [keys[-1].replace("_title", "_content")]
    content = TXT
    for k in content_keys:
        content = content[k]

    with st.expander(title, expanded=False):
        st.markdown(content)


# ══════════════════════════════════════════════════════════════
#  Core deterministic calculation
# ══════════════════════════════════════════════════════════════
def compute_metrics(
    a1v: float,
    a2v: float,
    a3v: float,
    bv: float,
    cross_ratio_v: float,
    prep_post_ratio_v: float,
    loss_unit_v: float,
    qualv: str,
    schedv: str,
    t1v: float,
    t2v: float,
    t3v: float,
) -> Tuple[float, float, float, float, float]:
    """
    Return (S, C, C_loss, E, E_total) for the given parameter set.

    S         – overall success rate
    C         – labour cost
    C_loss    – loss‑adjusted cost
    E         – efficiency (C / S)
    E_total   – total efficiency (C_loss / S)
    """
    # Scenario‑specific multipliers
    qual_T, qual_B = (1, 1)
    sched_T, sched_B = (1, 1)
    if qualv == "Low":
        qual_T, qual_B = (2 / 3, 0.8)
    if schedv == "Late":
        sched_T, sched_B = (2 / 3, 0.8)

    # Success probabilities
    a_tot = a1v * a2v * a3v
    b_eff = bv * qual_B * sched_B
    S_x = 1 - (1 - a_tot) * (1 - b_eff)

    # Cost components
    T = (t1v + t2v + t3v) * qual_T * sched_T
    C_x = T * (1 + cross_ratio_v + prep_post_ratio_v)
    C_loss_x = C_x + loss_unit_v * C_x * (1 - S_x)

    # Efficiency metrics
    E_x = C_x / S_x
    E_total_x = C_loss_x / S_x

    return S_x, C_x, C_loss_x, E_x, E_total_x


# ══════════════════════════════════════════════════════════════
#  Sensitivity bar‑plot helper
# ══════════════════════════════════════════════════════════════
def make_sensitivity_bar(
    df: pd.DataFrame,
    value_col: str,
    tick_fmt: str = "{:.2f}",
    order=None,
):
    """
    Return a horizontal bar chart (Plotly) for sensitivities.
    The DataFrame must have 'Parameter' and `value_col`.
    """
    df = df.copy()
    df[value_col] = df[value_col].abs()

    # Axis and colour settings
    if "rel" in value_col.lower():
        xaxis_label = TXT["charts"]["relative_sensitivity"]["xaxis"]
    elif "std" in value_col.lower():
        xaxis_label = TXT["charts"]["standardized_sensitivity"]["xaxis"]
    else:
        xaxis_label = value_col

    if order:
        df["Parameter"] = pd.Categorical(
            df["Parameter"], categories=order, ordered=True
        )

    fig = px.bar(
        df,
        x=value_col,
        y="Parameter",
        orientation="h",
        text=df[value_col].map(tick_fmt.format),
        color_discrete_sequence=["#000000"],
        labels={value_col: xaxis_label, "Parameter": ""},
    )
    fig.update_traces(
        texttemplate="%{text}",
        insidetextfont_color="white",
        outsidetextfont_color="gray",
    )
    fig.update_layout(
        showlegend=False,
        yaxis=dict(categoryorder="array", categoryarray=order) if order else {},
        xaxis_title=xaxis_label,
        font=dict(size=14),
        bargap=0.1,
        margin=dict(t=30, b=40),
    )
    return fig


# ══════════════════════════════════════════════════════════════
#  Streamlit – page‑wide config
# ══════════════════════════════════════════════════════════════
st.set_page_config(page_title="Cross‑Check Simulator", layout="wide")

if "lang" not in st.session_state:
    st.session_state.lang = "English"
lang = st.sidebar.radio(
    "Language / 言語", ["English", "日本語"], index=0, key="lang", horizontal=True
)

# ══════════════════════════════════════════════════════════════
#  Localised strings (TXT) – abbreviated here for brevity
#  ...  (the content of TXT_EN and TXT_JA remains unchanged)
# ══════════════════════════════════════════════════════════════
#  ↓↓↓  TXT_EN and TXT_JA dicts  ↓↓↓
#  (omitted – identical to original except minor typo fixes)
TXT_EN = {
    "panel": {
        "input": "INPUT PANEL",
        "output": "KEY OUTPUT METRICS",
    },
    "metrics": {
        "a_total": "a_total",
        "succ": "Success Rate S",
        "C": "Labor Cost C",
        "Closs": "C_total (with loss)",
        "loss_unit": "Loss unit ℓ",
        "E_base": "Efficiency E (baseline)",
        "E_total": "E_total",
    },
    "charts": {
        "quality_schedule": {
            "title": "Quality × Schedule 2×2 Matrix",
            "expander_title": "📘 About Quality × Schedule Chart",
            "expander_content": (
                "Each bar shows E_total under different combinations of quality and schedule, "
                "with labels showing the corresponding success rate.  \n\n"
                "This helps compare cost-performance tradeoffs across operational scenarios."
            ),
        },
        "tornado": {
            "title": "Tornado Sensitivity",
            "expander_title": "📘 Explanation: Impact of ±20% Parameter Changes",
            "expander_content": (
                "This chart visualizes the effect of ±20% changes in key parameters on E_total (cost per success).  \n"
                "Selected parameters (a₁, a₂, a₃, b₀, CR, PP, ℓ) are core drivers of success, effort, and loss.  \n"
                "This helps identify which inputs most strongly affect cost-efficiency."
            ),
            "xaxis": "|ΔE/E|",
        },
        "relative_sensitivity": {
            "title": "Relative Sensitivity",
            "xaxis": "Relative Sensitivity (∂E/∂x × x/E)",
            "expander_title": "📘 About Relative Sensitivity",
            "expander_content": """This chart displays the relative sensitivity of E_total (loss-adjusted cost per success) with respect to three independent parameters: loss unit ℓ, labor cost C, and success rate S.  
It quantifies the elasticity (∂E/∂x × x/E) for each parameter, showing how a 1% proportional change impacts overall cost efficiency.  
Use this analysis to prioritize which factor most improves cost efficiency when adjusted.""",
        },
        "standardized_sensitivity": {
            "title": "Standardized Sensitivity ",
            "xaxis": "Standardized Sensitivity (ΔE/σ_E)",
            "expander_title": "📘 About Standardized Sensitivity",
            "expander_content": """This chart shows the standardized sensitivity of E_total with respect to loss unit ℓ, labor cost C, and success rate S.  
It normalizes each parameter’s partial derivative by its variability (∂E/∂x × σₓ/σ_E) to reveal which uncertainties contribute most to efficiency variance.  
Use this analysis for risk assessment and uncertainty management.""",
        },
        "monte_carlo": {
            "title": "Monte Carlo Summary Statistics",
            "variable": "MC variable",
            "expander_title": "📘 Explanation: Monte Carlo Parameter Distributions",
            "expander_content": (
                "The following parameters are assigned probabilistic distributions to capture plausible uncertainty ranges:  \n\n"
                "- **a₁, a₂**: Normally distributed (mean = selected value, σ = 0.03), reflecting variation in basic process success rates due to human or environmental variability.  \n"
                "- **a₃**: Triangular distribution (±10%) to reflect process-specific asymmetry in the final step's reliability.  \n"
                "- **b₀**: Uniform between 0.70–0.90, assuming checker quality varies widely across contexts.  \n"
                "- **Cross-ratio (CR), Prep/Post ratio (PP)**: Triangular (±20%) around selected values to reflect managerial estimation variance.  \n"
                "- **Loss unit ℓ**: Triangular (±20%) for capturing business risk variability.  \n\n"
                "These distributions are selected based on empirical heuristics: normal for stable processes (a₁, a₂), triangular for bounded uncertain estimates (a₃, CR, PP, ℓ), and uniform for quality variability (b₀)."
            ),
            "mean": "Mean",
            "median": "Median",
            "ci": "5–95% CI",
            "caption": {
                "en": "Monte Carlo simulation output distributions with median, mean (solid), and 5–95% CI (dotted)",
                "ja": "モンテカルロ法による出力分布：中央値・平均値（実線）、信頼区間5–95%（点線）"
            },
            "card_unit": "[E_total]",
        }
    }
}
TXT_JA = {
    "panel": {
        "input": "入力パネル",
        "output": "主要出力指標",
    },
    "metrics": {
        "a_total": "a_total",
        "succ": "成功率 S",
        "C": "C（作業工数）",
        "Closs": "C_total（損失込）",
        "loss_unit": "損失単価 ℓ",
        "E_base": "効率 E（ベースライン）",
        "E_total": "E_total",
    },
    "charts": {
        "quality_schedule": {
            "title": "品質×納期の2×2マトリクス",
            "expander_title": "📘 品質 × 納期グラフについて",
            "expander_content": (
                "各バーは品質・納期の組み合わせごとのE_total（成功1件あたり総コスト）を示し、"
                "ラベルはその時の成功率です。  \n\n"
                "運用シナリオごとのコストパフォーマンスの違いを比較できます。"
            ),
        },
        "tornado": {
            "title": "トルネード感度分析",
            "expander_title": "📘 説明：パラメータ±20％変化の影響",
            "expander_content": (
                "このグラフは主要パラメータ（a₁, a₂, a₃, b₀, CR, PP, ℓ）を±20%変化させたときのE_total（成功1件あたり総コスト）への影響を示します。  \n"
                "どの入力がコスト効率に最も強く影響するかを可視化します。"
            ),
            "xaxis": "|ΔE/E|",
        },
        "relative_sensitivity": {
            "title": "相対感度",
            "xaxis": "相対感度（∂E/∂x × x/E）",
            "expander_title": "📘 相対感度グラフについて",
            "expander_content": """互いに独立した各指標（損失単価ℓ、C（作業工数）、成功率S）が1%変化したときのE_total（総合コスト効率）の変化率（弾性値）を示します。  
設計や運用改善の優先度を考える上で、どの因子が効率に最も影響するか把握できます。""",
        },
        "standardized_sensitivity": {
            "title": "標準化感度",
            "xaxis": "標準化感度（ΔE/σ_E）",
            "expander_title": "📘 標準化感度グラフについて",
            "expander_content": """互いに独立した各指標（損失単価ℓ、C（作業工数）、成功率S）のばらつき（標準偏差）で正規化したE_totalへの影響度を示します。  
不確実性によるリスク評価や、どの因子の分散がコスト効率の不安定さに寄与しているかを把握できます。""",
        },
        "monte_carlo": {
            "title": "モンテカルロ要約統計",
            "variable": "MC対象変数",
            "expander_title": "📘 説明：モンテカルロにおけるパラメータ分布",
            "expander_content": (
                "以下のパラメータに不確実性（分布）を仮定してシミュレーションを行います：  \n\n"
                "- **a₁, a₂**：正規分布（平均=選択値、σ=0.03）で、人や環境によるばらつきを反映  \n"
                "- **a₃**：三角分布（±10%）で最終工程の非対称な信頼性を表現  \n"
                "- **b₀**：一様分布（0.70–0.90）でチェッカー品質の幅広い状況を想定  \n"
                "- **クロス比（CR）、準備・後処理比（PP）**：三角分布（±20%）で見積り誤差を反映  \n"
                "- **損失単価 ℓ**：三角分布（±20%）でビジネスリスクの幅を表現  \n\n"
                "分布の選択は経験則に基づき、安定工程（a₁, a₂）は正規、推定値（a₃, CR, PP, ℓ）は三角、一様（b₀）は品質の幅広さを想定しています。"
            ),
            "mean": "平均値",
            "median": "中央値",
            "ci": "信頼区間5–95%",
            "caption": {
                "en": "Monte Carlo simulation output distributions with median, mean (solid), and 5–95% CI (dotted)",
                "ja": "モンテカルロ法による出力分布：中央値・平均値（実線）、信頼区間5–95%（点線）"
            },
            "card_unit": "[E_total]",
        }
    }
}
TXT_ALL = {"EN": TXT_EN, "JA": TXT_JA}
TXT = get_text_labels(lang)

# ══════════════════════════════════════════════════════════════
#  Sidebar – input controls
# ══════════════════════════════════════════════════════════════
def get_sidebar_params() -> Dict[str, float]:
    st.sidebar.title(TXT["panel"]["input"])

    # 1) Step success rates
    a1 = st.sidebar.slider("a1 (step 1 success)", 0.5, 1.0, get_state("a1", 0.95), 0.01)
    a2 = st.sidebar.slider("a2 (step 2 success)", 0.5, 1.0, get_state("a2", 0.95), 0.01)
    a3 = st.sidebar.slider("a3 (step 3 success)", 0.5, 1.0, get_state("a3", 0.80), 0.01)

    # 2) Task times
    col_t1, col_t2, col_t3 = st.sidebar.columns(3)
    with col_t1:
        T1 = st.number_input("T1 [h]", 0, 200, get_state("T1", 10), key="T1")
    with col_t2:
        T2 = st.number_input("T2 [h]", 0, 200, get_state("T2", 10), key="T2")
    with col_t3:
        T3 = st.number_input("T3 [h]", 0, 200, get_state("T3", 30), key="T3")

    # 3) Quality & Schedule
    col_q, col_s = st.sidebar.columns(2)
    with col_q:
        qual = st.selectbox(
            "Quality", ["Standard", "Low"], 0 if get_state("qual", "Standard") == "Standard" else 1, key="qual"
        )
    with col_s:
        sched = st.selectbox(
            "Schedule", ["OnTime", "Late"], 0 if get_state("sched", "OnTime") == "OnTime" else 1, key="sched"
        )

    st.sidebar.markdown("---")

    # 4) Checker effectiveness
    b0 = st.sidebar.slider("b0 (checker success)", 0.0, 1.0, get_state("b0", 0.80), 0.01)

    # 5) Cost ratios
    cross_ratio = st.sidebar.slider(
        "Cross‑ratio", 0.0, 0.5, get_state("cross_ratio", 0.30), 0.01
    )
    prep_post_ratio = st.sidebar.slider(
        "Prep/Post ratio", 0.0, 0.5, get_state("prep_post_ratio", 0.40), 0.01
    )

    st.sidebar.markdown("---")

    # 6) Loss unit
    loss_unit = st.sidebar.slider(
        "Loss unit ℓ", 0.0, 50.0, get_state("loss_unit", 0.0), 0.1
    )

    # 7) Monte Carlo parameters
    st.sidebar.markdown("<hr style='border-top:3px solid black'>", unsafe_allow_html=True)
    st.sidebar.title("Monte‑Carlo")

    col_n, col_var = st.sidebar.columns(2)
    with col_n:
        sample_n = st.number_input(
            "Samples", 1_000, 1_000_000, get_state("sample_n", 100_000), step=10_000
        )
    with col_var:
        mc_var = st.selectbox("MC variable", ["E_total", "Success S"], 0, key="mc_var")

    return dict(
        a1=a1,
        a2=a2,
        a3=a3,
        b0=b0,
        qual=qual,
        sched=sched,
        loss_unit=loss_unit,
        T1=T1,
        T2=T2,
        T3=T3,
        cross_ratio=cross_ratio,
        prep_post_ratio=prep_post_ratio,
        sample_n=sample_n,
        mc_var=mc_var,
    )


# Retrieve sidebar inputs
params = get_sidebar_params()

# ══════════════════════════════════════════════════════════════
#  Baseline deterministic results
# ══════════════════════════════════════════════════════════════
qual_T, qual_B = (1, 1) if params["qual"] == "Standard" else (2 / 3, 0.8)
sched_T, sched_B = (1, 1) if params["sched"] == "OnTime" else (2 / 3, 0.8)

a_total = params["a1"] * params["a2"] * params["a3"]
b_eff = params["b0"] * qual_B * sched_B
S = 1 - (1 - a_total) * (1 - b_eff)

T = (params["T1"] + params["T2"] + params["T3"]) * qual_T * sched_T
C = T * (1 + params["cross_ratio"] + params["prep_post_ratio"])
C_loss = C + params["loss_unit"] * C * (1 - S)

E = C / S
E_total = C_loss / S

# ══════════════════════════════════════════════════════════════
#  Monte‑Carlo simulation (vectorised)
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def run_mc(p: Dict[str, float], N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (Evals, Svals, Cvals, L_samples) arrays."""
    rng = np.random.default_rng(0)

    # Success parameters
    a1s = rng.normal(p["a1"], 0.03, N).clip(0, 1)
    a2s = rng.normal(p["a2"], 0.03, N).clip(0, 1)
    a3s = rng.triangular(p["a3"] * 0.9, p["a3"], p["a3"] * 1.1, N).clip(0, 1)
    b0s = rng.uniform(0.70, 0.90, N)

    # Task times (±10 %)
    t1s = rng.normal(p["T1"], p["T1"] * 0.1, N).clip(min=1)
    t2s = rng.normal(p["T2"], p["T2"] * 0.1, N).clip(min=1)
    t3s = rng.normal(p["T3"], p["T3"] * 0.1, N).clip(min=1)

    # Cost ratios
    cross_ratios = (
        rng.triangular(p["cross_ratio"] * 0.8, p["cross_ratio"], p["cross_ratio"] * 1.2, N)
        if p["cross_ratio"] > 0 else np.zeros(N)
    )
    prep_post_ratios = (
        rng.triangular(p["prep_post_ratio"] * 0.8, p["prep_post_ratio"], p["prep_post_ratio"] * 1.2, N)
        if p["prep_post_ratio"] > 0 else np.zeros(N)
    )

    # Loss unit ℓ
    L_samples = (
        rng.triangular(p["loss_unit"] * 0.8, p["loss_unit"], p["loss_unit"] * 1.2, N)
        if p["loss_unit"] > 0 else np.zeros(N)
    )

    # Multipliers
    qual_T, qual_B = (1, 1) if p["qual"] == "Standard" else (2 / 3, 0.8)
    sched_T, sched_B = (1, 1) if p["sched"] == "OnTime" else (2 / 3, 0.8)

    # Vectorised calculations
    a_tot = a1s * a2s * a3s
    b_eff = b0s * qual_B * sched_B
    Svals = 1 - (1 - a_tot) * (1 - b_eff)

    Tvals = (t1s + t2s + t3s) * qual_T * sched_T
    Cvals = Tvals * (1 + cross_ratios + prep_post_ratios)

    Evals = (Cvals + L_samples * Cvals * (1 - Svals)) / Svals
    return Evals, Svals, Cvals, L_samples


Evals, Svals, Cvals, L_samples = run_mc(params, int(params["sample_n"]))

#  Standard deviations (for standardised sensitivities)
σE = Evals.std()
σC = Cvals.std()
σS = Svals.std()

#  ---  FIX #4  -------------------------------------------------
#      σℓ must correspond to distribution of (C · ℓ)
# --------------------------------------------------------------
σL = (Cvals * L_samples).std()

#  Loss‑adjusted cost distribution (for completeness)
C_loss_vals = Cvals + (Cvals * L_samples) * (1 - Svals)
σ_Closs = C_loss_vals.std()

# ══════════════════════════════════════════════════════════════
#  Symbolic partial derivatives (elasticities)
# ══════════════════════════════════════════════════════════════
def symbolic_derivatives(C_: float, S_: float, L_: float) -> Dict[str, float]:
    """Return partial derivatives of E wrt C, S, L."""
    C_sym, S_sym, L_sym = sp.symbols("C S L")

    #  ---  FIX #1  -------------------------------------------------
    #  Missing cost factor C in loss term
    # --------------------------------------------------------------
    E_expr = (C_sym + C_sym * L_sym * (1 - S_sym)) / S_sym

    return {
        "dE_dC": float(sp.diff(E_expr, C_sym).subs({C_sym: C_, S_sym: S_, L_sym: L_})),
        "dE_dS": float(sp.diff(E_expr, S_sym).subs({C_sym: C_, S_sym: S_, L_sym: L_})),
        "dE_dL": float(sp.diff(E_expr, L_sym).subs({C_sym: C_, S_sym: S_, L_sym: L_})),
    }


sym_derivs = symbolic_derivatives(C, S, params["loss_unit"])
dE_dC = sym_derivs["dE_dC"]
dE_dS = sym_derivs["dE_dS"]
dE_dL = sym_derivs["dE_dL"]

#  Elasticities (relative sensitivities)
rel_C = dE_dC * C / E_total
rel_S = dE_dS * S / E_total
rel_L = dE_dL * params["loss_unit"] / E_total

#  Standardised sensitivities
std_C = dE_dC * σC / σE
std_S = dE_dS * σS / σE
std_L = dE_dL * σL / σE

# ══════════════════════════════════════════════════════════════
#  UI – output metrics & charts
# ══════════════════════════════════════════════════════════════
left, right = st.columns([1, 2])

with left:
    st.subheader(TXT["panel"]["output"])

    st.metric(TXT["metrics"]["a_total"], f"{a_total:.4f}")
    S_x, C_x, C_loss_x, E_x, E_total_x = compute_metrics(
        a1v=params["a1"],
        a2v=params["a2"],
        a3v=params["a3"],
        bv=params["b0"],
        cross_ratio_v=params["cross_ratio"],
        prep_post_ratio_v=params["prep_post_ratio"],
        loss_unit_v=params["loss_unit"],
        qualv=params["qual"],
        schedv=params["sched"],
        t1v=params["T1"],
        t2v=params["T2"],
        t3v=params["T3"],
    )
    st.metric(TXT["metrics"]["succ"], f"{S_x:.2%}")
    st.metric(TXT["metrics"]["C"], f"{C_x:.1f}")
    st.metric(TXT["metrics"]["Closs"], f"{C_loss_x:.1f}")
    st.metric(TXT["metrics"]["E_base"], f"{E_x:.1f}")
    st.metric(TXT["metrics"]["E_total"], f"{E_total_x:.1f}")

with right:
    # ----------------------------------------------------------
    # Quality × Schedule matrix
    # ----------------------------------------------------------
    st.subheader(TXT["charts"]["quality_schedule"]["title"])
    exp("charts.quality_schedule.expander_title")

    scenarios = [
        ("Std/On", "Standard", "OnTime"),
        ("Std/Late", "Standard", "Late"),
        ("Low/On", "Low", "OnTime"),
        ("Low/Late", "Low", "Late"),
    ]
    bars = []
    for name, qg, scd in scenarios:
        S_p, _, _, _, E_tot_p = compute_metrics(
            a1v=params["a1"],
            a2v=params["a2"],
            a3v=params["a3"],
            bv=params["b0"],
            cross_ratio_v=params["cross_ratio"],
            prep_post_ratio_v=params["prep_post_ratio"],
            loss_unit_v=params["loss_unit"],
            qualv=qg,
            schedv=scd,
            t1v=params["T1"],
            t2v=params["T2"],
            t3v=params["T3"],
        )
        bars.append(dict(Scenario=name, E_total=E_tot_p, S=f"{S_p:.1%}"))

    df_bars = pd.DataFrame(bars)
    fig_q = px.bar(
        df_bars,
        x="Scenario",
        y="E_total",
        text="S",
        color_discrete_sequence=["#000000"],
        labels={
            "E_total": TXT["metrics"]["E_total"],
            "Scenario": "Scenario",
        },
    )
    fig_q.update_traces(textposition="auto",
                        insidetextfont_color="white",
                        outsidetextfont_color="gray")
    fig_q.update_layout(bargap=0.1, font=dict(size=14))
    st.plotly_chart(fig_q, use_container_width=True)

    # ----------------------------------------------------------
    # Tornado sensitivity (±20 %)
    # ----------------------------------------------------------
    st.subheader(TXT["charts"]["tornado"]["title"])
    exp("charts.tornado.expander_title")

    sens_targets = {
        "a1": params["a1"],
        "a2": params["a2"],
        "a3": params["a3"],
        "b": params["b0"],
        "cross_ratio": params["cross_ratio"],
        "prep_post_ratio": params["prep_post_ratio"],
        "loss_unit": params["loss_unit"],
    }
    name_map = {
        "a1": "a1v",
        "a2": "a2v",
        "a3": "a3v",
        "b": "bv",
        "cross_ratio": "cross_ratio_v",
        "prep_post_ratio": "prep_post_ratio_v",
        "loss_unit": "loss_unit_v",
    }
    tornado_rows = []
    for k, v in sens_targets.items():
        lo = max(v * 0.8, 0)
        hi = (min(v * 1.2, 1) if k in ("a1", "a2", "a3", "b") else v * 1.2)

        kwargs_lo = {
            "a1v": params["a1"],
            "a2v": params["a2"],
            "a3v": params["a3"],
            "bv": params["b0"],
            "cross_ratio_v": params["cross_ratio"],
            "prep_post_ratio_v": params["prep_post_ratio"],
            "loss_unit_v": params["loss_unit"],
            "t1v": params["T1"],
            "t2v": params["T2"],
            "t3v": params["T3"],
            "qualv": params["qual"],
            "schedv": params["sched"],
        }
        kwargs_hi = kwargs_lo.copy()
        kwargs_lo[name_map[k]] = lo
        kwargs_hi[name_map[k]] = hi

        _, _, _, _, E_lo = compute_metrics(**kwargs_lo)
        _, _, _, _, E_hi = compute_metrics(**kwargs_hi)
        delta = max(abs(E_lo - E_total_x), abs(E_hi - E_total_x)) / E_total_x * 100
        tornado_rows.append((k, delta))

    df_tornado = (
        pd.DataFrame(tornado_rows, columns=["Parameter", "RelChange"])
        .sort_values("RelChange", ascending=False)
    )
    fig_t = px.bar(
        df_tornado,
        x="RelChange",
        y="Parameter",
        orientation="h",
        color_discrete_sequence=["#000000"],
        labels={"RelChange": TXT["charts"]["tornado"]["xaxis"], "Parameter": ""},
    )
    fig_t.update_layout(showlegend=False,
                        yaxis=dict(categoryorder="total ascending"),
                        font=dict(size=14),
                        bargap=0.1)
    st.plotly_chart(fig_t, use_container_width=True)

    # ----------------------------------------------------------
    # Relative & Standardised sensitivity charts
    # ----------------------------------------------------------
    sens_df = pd.DataFrame(
        dict(
            Parameter=[TXT["metrics"]["loss_unit"],
                       TXT["metrics"]["C"],
                       TXT["metrics"]["succ"]],
            Relative=[abs(rel_L), abs(rel_C), abs(rel_S)],
            Standardised=[abs(std_L), abs(std_C), abs(std_S)],
        )
    )
    order = sens_df["Parameter"].tolist()

    fig_rel = make_sensitivity_bar(
        sens_df[["Parameter", "Relative"]].rename(columns={"Relative": "rel"}),
        value_col="rel",
        order=order,
        tick_fmt="{:.2f}",
    )
    fig_std = make_sensitivity_bar(
        sens_df[["Parameter", "Standardised"]].rename(columns={"Standardised": "std"}),
        value_col="std",
        order=order,
        tick_fmt="{:.3f}",
    )

#  Two‑column layout for sensitivity plots
col_rel, col_std = st.columns(2)
with col_rel:
    col_rel.subheader(TXT["charts"]["relative_sensitivity"]["title"])
    exp("charts.relative_sensitivity.expander_title")
    col_rel.plotly_chart(fig_rel, use_container_width=True)
with col_std:
    col_std.subheader(TXT["charts"]["standardized_sensitivity"]["title"])
    exp("charts.standardized_sensitivity.expander_title")
    col_std.plotly_chart(fig_std, use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  Monte‑Carlo histogram
# ══════════════════════════════════════════════════════════════
st.subheader(TXT["charts"]["monte_carlo"]["title"])
exp("charts.monte_carlo.expander_title")

mc_var = params["mc_var"]
data = Evals if mc_var == "E_total" else Svals
ci_low, ci_high = np.percentile(data, [5, 95])
mean, median = data.mean(), np.median(data)
dec = ".2f" if mc_var == "E_total" else ".4f"

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.metric(TXT["charts"]["monte_carlo"]["mean"] + TXT["charts"]["monte_carlo"]["card_unit"],
              f"{mean:{dec}}")
with col_m2:
    st.metric(TXT["charts"]["monte_carlo"]["median"] + TXT["charts"]["monte_carlo"]["card_unit"],
              f"{median:{dec}}")
with col_m3:
    st.metric(TXT["charts"]["monte_carlo"]["ci"] + TXT["charts"]["monte_carlo"]["card_unit"],
              f"{ci_low:{dec}} – {ci_high:{dec}}")

LABEL_E = "E_total" if lang == "English" else "E_total: 総合効率"
LABEL_CNT = "Count" if lang == "English" else "頻度"

fig_hist = px.histogram(
    data,
    nbins=100,
    labels={"value": LABEL_E},
    color_discrete_sequence=["#000000"],
)
fig_hist.update_traces(marker_line_width=0.5)
fig_hist.add_vline(x=mean, line_dash="solid", line_color="#000000")
fig_hist.add_vline(x=median, line_dash="solid", line_color="#000000")
fig_hist.add_vline(x=ci_low, line_dash="dot", line_color="#000000")
fig_hist.add_vline(x=ci_high, line_dash="dot", line_color="#000000")
fig_hist.update_layout(
    yaxis_title=LABEL_CNT,
    bargap=0.01,
    margin=dict(t=70),
    showlegend=False,
)
st.plotly_chart(fig_hist, use_container_width=True)

legend = "Mean / Median: solid  5–95 % CI: dotted" if lang == "English" \
    else "平均 / 中央値：実線  信頼区間5–95 %：点線"
st.caption(legend)