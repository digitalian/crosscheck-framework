def get_text_labels(lang):
    LANG = "EN" if lang == "English" else "JA"
    return TXT_ALL[LANG]

import streamlit as st
import numpy as np
import sympy as sp
import pandas as pd
import plotly.express as px
from typing import Tuple, Dict

# ───────────────────────────────
# Utility helpers for session state and standardized expanders
def get_state(key, default):
    """Retrieve or initialize a value in session_state."""
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

def exp(path: str):
    """Create an expander from nested TXT using dot-separated key path."""
    keys = path.split(".")
    d = TXT
    for k in keys:
        d = d[k]
    content_keys = keys[:-1] + [keys[-1].replace("_title", "_content")]
    c = TXT
    for k in content_keys:
        c = c[k]
    with st.expander(d, expanded=False):
        st.markdown(c)

# ───────────────────────────────
# Centralized core calculation function for the model
def compute_metrics(
    a1v: float, a2v: float, a3v: float, bv: float,
    cross_ratio_v: float, prep_post_ratio_v: float, loss_unit_v: float,
    qualv: str, schedv: str, t1v: float, t2v: float, t3v: float
) -> Tuple[float, float, float, float, float]:
    """
    Compute key output metrics:
    - S_x: overall success rate
    - C_x: labor cost
    - C_loss_x: loss-adjusted cost
    - E_x: efficiency (C/S)
    - E_total_x: total efficiency (C_loss/S)
    """
    # Fill in defaults from current sidebar inputs if not provided
    defaults = {
        "a1v": a1, "a2v": a2, "a3v": a3, "bv": b0,
        "cross_ratio_v": cross_ratio, "prep_post_ratio_v": prep_post_ratio, "loss_unit_v": loss_unit,
        "qualv": "Standard", "schedv": "OnTime", "t1v": T1, "t2v": T2, "t3v": T3
    }
    vals = locals()
    for k, v in defaults.items():
        if vals[k] is None:
            vals[k] = v
    # Adjust quality and schedule multipliers based on scenario
    qual_T, qual_B = (1,1)
    sched_T, sched_B = (1,1)
    if vals["qualv"] == "Low":
        qual_T, qual_B = (2/3, 0.8)
    if vals["schedv"] == "Late":
        sched_T, sched_B = (2/3, 0.8)
    # Calculate total process success and checker effectiveness
    a_tot = vals["a1v"] * vals["a2v"] * vals["a3v"]
    b_eff = vals["bv"] * qual_B * sched_B
    # Compute overall success rate
    S_x    = 1 - (1 - a_tot) * (1 - b_eff)
    # Compute labor time, cost, and loss-adjusted cost
    T  = (vals["t1v"] + vals["t2v"] + vals["t3v"]) * qual_T * sched_T
    C_x    = T * (1 + vals["cross_ratio_v"] + vals["prep_post_ratio_v"])
    C_loss_x = C_x + vals["loss_unit_v"] * C_x * (1 - S_x)
    # Compute efficiency and total efficiency
    E_x    = C_x / S_x
    E_total_x = C_loss_x / S_x
    return S_x, C_x, C_loss_x, E_x, E_total_x

# ───────────────────────────────
# Sensitivity bar plot utility
def make_sensitivity_bar(
    df,
    value_col,
    tick_fmt="{:.2f}",
    order=None
):
    """
    Create a horizontal bar chart for sensitivities (relative or standardized).
    The input DataFrame should have a 'Parameter' column and the value_col to plot.
    The function infers the type (relative or standardized) from value_col,
    and sets appropriate color, xaxis label, and bar order.
    """
    df = df.copy()
    df[value_col] = df[value_col].apply(abs)
    # Infer type and set properties
    if "rel" in value_col.lower():
        color = "#000000"
        xaxis_label = TXT["charts"]["relative_sensitivity"]["xaxis"]
        bar_order = order
    elif "std" in value_col.lower():
        color = "#000000"
        xaxis_label = TXT["charts"]["standardized_sensitivity"]["xaxis"]
        bar_order = order
    else:
        color = "#000000"
        xaxis_label = value_col
        bar_order = order
    if bar_order is not None:
        df["Parameter"] = pd.Categorical(df["Parameter"], categories=bar_order, ordered=True)
    fig = px.bar(
        df, x=value_col, y="Parameter", orientation="h",
        text=df[value_col].map(tick_fmt.format),
        color_discrete_sequence=[color],
        labels={value_col: xaxis_label, "Parameter": ""}
    )
    fig.update_traces(
        texttemplate='%{text}',
        insidetextfont_color="white",
        outsidetextfont_color="gray"
    )
    fig.update_layout(
        showlegend=False,
        yaxis=dict(categoryorder="array", categoryarray=bar_order) if bar_order else {},
        xaxis_title=xaxis_label,
        font=dict(size=14),
        bargap=0.1,
        margin=dict(t=30, b=40)
    )
    return fig

st.set_page_config(page_title="Cross-Check Simulator (B/W)", layout="wide")

if "lang" not in st.session_state:
    st.session_state.lang = "English"
lang = st.sidebar.radio("Language / 言語", ["English", "日本語"], index=0, key="lang", horizontal=True)

# ─────────────────────────── Nested text definitions
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
        "E_total": "E_total (cost per success)",
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
            "xaxis": "|ΔE/E| (%)",
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
        "E_total": "E_total（成功1件あたりの総コスト）",
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
            "xaxis": "|ΔE/E| (%)",
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

# ──────────────────────────────────────────────────
# Sidebar input encapsulation with session state
def get_sidebar_params():
    st.sidebar.title(TXT["panel"]["input"])
    # a1
    a1 = st.sidebar.slider("a1 (step 1 success rate)", 0.5, 1.0, get_state("a1", 0.95), 0.01, key="a1")
    # a2
    a2 = st.sidebar.slider("a2 (step 2 success rate)", 0.5, 1.0, get_state("a2", 0.95), 0.01, key="a2")
    # a3
    a3 = st.sidebar.slider("a3 (step 3 success rate)", 0.5, 1.0, get_state("a3", 0.80), 0.01, key="a3")
    # b0
    b0 = st.sidebar.slider("b0 (checker success rate)", 0.0, 1.0, get_state("b0", 0.80), 0.01, key="b0")
    # Group Quality and Schedule side-by-side
    col_qs1, col_qs2 = st.sidebar.columns(2)
    with col_qs1:
        qual = st.selectbox("Qual-Grade", ["Standard", "Low"], index=0 if get_state("qual", "Standard") == "Standard" else 1, key="qual")
    with col_qs2:
        sched = st.selectbox("Schedule", ["OnTime", "Late"], index=0 if get_state("sched", "OnTime") == "OnTime" else 1, key="sched")
    # loss_unit
    loss_unit = st.sidebar.slider("Loss unit ℓ", 0.0, 50.0, get_state("loss_unit", 0.0), 0.1, key="loss_unit")
    # T1, T2, T3 grouped in a single row
    col_t1, col_t2, col_t3 = st.sidebar.columns(3)
    with col_t1:
        T1 = st.number_input("T1 (h)", 0, 200, get_state("T1", 10), key="T1")
    with col_t2:
        T2 = st.number_input("T2 (h)", 0, 200, get_state("T2", 10), key="T2")
    with col_t3:
        T3 = st.number_input("T3 (h)", 0, 200, get_state("T3", 30), key="T3")
    # cross_ratio
    cross_ratio = st.sidebar.slider("Cross-ratio", 0.0, 0.5, get_state("cross_ratio", 0.30), 0.01, key="cross_ratio")
    # prep_post_ratio
    prep_post_ratio = st.sidebar.slider("Prep+Post ratio", 0.0, 0.5, get_state("prep_post_ratio", 0.40), 0.01, key="prep_post_ratio")
    return {
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "b0": b0,
        "qual": qual,
        "sched": sched,
        "loss_unit": loss_unit,
        "T1": T1,
        "T2": T2,
        "T3": T3,
        "cross_ratio": cross_ratio,
        "prep_post_ratio": prep_post_ratio,
    }

# Retrieve all sidebar input values as a dictionary (now session-state aware)
params = get_sidebar_params()
a1 = params["a1"]
a2 = params["a2"]
a3 = params["a3"]
b0 = params["b0"]
qual = params["qual"]
sched = params["sched"]
loss_unit = params["loss_unit"]
T1 = params["T1"]
T2 = params["T2"]
T3 = params["T3"]
cross_ratio = params["cross_ratio"]
prep_post_ratio = params["prep_post_ratio"]

# ───────────────────────────────
# Deterministic calculation of all output metrics for the current scenario
# This forms the baseline for scenario comparison and sensitivity analysis.
qual_T, qual_B  = (1,1) if params["qual"]=="Standard" else (2/3,0.8)
sched_T, sched_B= (1,1) if params["sched"]=="OnTime"   else (2/3,0.8)
a_total = params["a1"] * params["a2"] * params["a3"]
b_eff   = params["b0"] * qual_B * sched_B
S       = 1 - (1 - a_total) * (1 - b_eff)  # Overall success rate
T = (params["T1"] + params["T2"] + params["T3"]) * qual_T * sched_T      # Total labor time after quality/schedule multipliers
C       = T * (1 + params["cross_ratio"] + params["prep_post_ratio"])  # Labor cost including cross and prep/post
C_loss  = C + params["loss_unit"] * C * (1 - S)      # Loss-adjusted total cost
E       = C / S                            # Efficiency (cost per success, baseline)
E_total = C_loss / S                       # Total efficiency (loss-adjusted cost per success)

# ───────────────────────────────
@st.cache_data(show_spinner=False)
def run_mc(params: Dict[str, float], N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run vectorized Monte Carlo simulation and return arrays (Evals, Svals, Cvals)."""
    rng = np.random.default_rng(0)
    # Vectorized parameter sampling
    a1s = rng.normal(params["a1"], 0.03, N).clip(0, 1)
    a2s = rng.normal(params["a2"], 0.03, N).clip(0, 1)
    a3s = rng.triangular(params["a3"]*0.9, params["a3"], params["a3"]*1.1, N).clip(0,1)
    b0s = rng.uniform(0.70, 0.90, N).clip(0,1)
    # Time distributions (10% variability for T1-T3)
    t1s = rng.normal(params["T1"], params["T1"] * 0.1, N).clip(min=1)
    t2s = rng.normal(params["T2"], params["T2"] * 0.1, N).clip(min=1)
    t3s = rng.normal(params["T3"], params["T3"] * 0.1, N).clip(min=1)
    cross_ratios = (
        rng.triangular(params["cross_ratio"]*0.8, params["cross_ratio"], params["cross_ratio"]*1.2, N)
        if params["cross_ratio"] > 0 else np.zeros(N)
    )
    prep_post_ratios = (
        rng.triangular(params["prep_post_ratio"]*0.8, params["prep_post_ratio"], params["prep_post_ratio"]*1.2, N)
        if params["prep_post_ratio"] > 0 else np.zeros(N)
    )
    loss_units = (
        rng.triangular(params["loss_unit"]*0.8, params["loss_unit"], params["loss_unit"]*1.2, N)
        if params["loss_unit"] > 0 else np.zeros(N)
    )
    # Compute metrics vectorized
    a_tot = a1s * a2s * a3s
    qual_T, qual_B = (1,1) if params["qual"]=="Standard" else (2/3,0.8)
    sched_T, sched_B = (1,1) if params["sched"]=="OnTime" else (2/3,0.8)
    b_eff = b0s * qual_B * sched_B
    Svals = 1 - (1 - a_tot) * (1 - b_eff)
    # Use sampled times for cost calculation
    Tvals = (t1s + t2s + t3s) * qual_T * sched_T
    Cvals = Tvals * (1 + cross_ratios + prep_post_ratios)
    Evals = (Cvals + loss_units * Cvals * (1 - Svals)) / Svals
    return Evals, Svals, Cvals

# Place Sample Size and MC Variable selection side-by-side
col_mc1, col_mc2 = st.sidebar.columns(2)
with col_mc1:
    sample_n = col_mc1.number_input(
        "Sample Size", 1000, 1000000, get_state("sample_n", 100000), step=10000, key="sample_n"
    )

# Run vectorized and cached MC simulation
Evals, Svals, Cvals = run_mc(params, sample_n)

# Calculate standard deviations for use in standardized sensitivity
σE = Evals.std()
σC = Cvals.std()
σS = Svals.std()
# For σL (loss_units), use the same distribution as in MC
if params["loss_unit"] > 0:
    mc_loss_units = np.random.triangular(
        params["loss_unit"]*0.8,
        params["loss_unit"],
        params["loss_unit"]*1.2,
        sample_n
    )
else:
    mc_loss_units = np.zeros(sample_n)
σL = np.std(mc_loss_units)

# Compute C_loss_vals for standardized sensitivity using actual C_loss distribution
if params["loss_unit"] > 0:
    C_loss_vals = Cvals + mc_loss_units * Cvals * (1 - Svals)
else:
    C_loss_vals = Cvals
σ_Closs = np.std(C_loss_vals)

# ───────────────────────────────
# Symbolic derivatives utility function
def symbolic_derivatives(C: float, S: float, L: float) -> Dict[str, float]:
    """Return partial derivatives of E with respect to C, S, and L."""
    C_sym, S_sym, L_sym = sp.symbols("C S L")
    E = (C_sym + L_sym * (1 - S_sym)) / S_sym
    return {
        "dE_dC": float(sp.diff(E, C_sym).subs({C_sym: C, S_sym: S, L_sym: L})),
        "dE_dS": float(sp.diff(E, S_sym).subs({C_sym: C, S_sym: S, L_sym: L})),
        "dE_dL": float(sp.diff(E, L_sym).subs({C_sym: C, S_sym: S, L_sym: L}))
    }

# Standardized sensitivities (∂E/∂x × σₓ/σ_E)
C_base = C
S_base = S
Lunit = loss_unit
sym_derivs = symbolic_derivatives(C_base, S_base, Lunit)
dE_dC = sym_derivs["dE_dC"]
dE_dS = sym_derivs["dE_dS"]
dE_dL = sym_derivs["dE_dL"]
std_C = dE_dC * σC / σE
std_S = dE_dS * σS / σE
std_L = dE_dL * σL / σE

# ───────────────────────────────
# Output metrics and scenario analysis
left, right = st.columns([1, 2])
with left:
    # Output metrics for current scenario (main panel)
    st.subheader(TXT["panel"]["output"])
    st.metric(TXT["metrics"]["a_total"], f"{a_total:.4f}")
    S_x, C_x, C_loss_x, E_x, E_total_x = compute_metrics(
        a1v=params["a1"], a2v=params["a2"], a3v=params["a3"], bv=params["b0"],
        cross_ratio_v=params["cross_ratio"], prep_post_ratio_v=params["prep_post_ratio"],
        loss_unit_v=params["loss_unit"], qualv=params["qual"], schedv=params["sched"],
        t1v=params["T1"], t2v=params["T2"], t3v=params["T3"]
    )
    st.metric(TXT["metrics"]["succ"],    f"{S_x:.2%}")
    st.metric(TXT["metrics"]["C"],       f"{C_x:.1f}")
    st.metric(TXT["metrics"]["Closs"],   f"{C_loss_x:.1f}")
    st.metric(TXT["metrics"]["E_base"],  f"{E_x:.1f}")
    st.metric(TXT["metrics"]["E_total"], f"{E_total_x:.1f}")


with right:

    # ───────────────────────────────
    # Quality × Schedule scenario matrix
    st.subheader(TXT["charts"]["quality_schedule"]["title"])
    exp("charts.quality_schedule.expander_title")
    scenarios = [("Std/On","Standard","OnTime"),
                 ("Std/Late","Standard","Late"),
                 ("Low/On","Low","OnTime"),
                 ("Low/Late","Low","Late")]
    bars=[]
    for name, qg, scd in scenarios:
        # For each scenario, compute the output metrics
        S_x, C_x, C_loss_x, E_x, E_total_x = compute_metrics(
            a1v=params["a1"], a2v=params["a2"], a3v=params["a3"], bv=params["b0"],
            cross_ratio_v=params["cross_ratio"], prep_post_ratio_v=params["prep_post_ratio"],
            loss_unit_v=params["loss_unit"], qualv=qg, schedv=scd, t1v=params["T1"], t2v=params["T2"], t3v=params["T3"]
        )
        bars.append(dict(Scenario=name,
                         E_total=E_total_x,
                         S=f"{S_x:.1%}"))

    # Standardize y/column names for clarity
    df_bars = pd.DataFrame(bars)
    df_bars.columns = ["Scenario", "E_total", "S"]  # Ensure correct order
    fig_q = px.bar(
        df_bars, x="Scenario", y="E_total", text="S",
        color_discrete_sequence=["#000000"],
        labels={
            "E_total": TXT["metrics"]["E_total"],
            "S": TXT["metrics"]["succ"],
            "Scenario": "Scenario"
        }
    )
    fig_q.update_traces(
        textposition="auto",
        insidetextfont_color="white",
        outsidetextfont_color="gray"
    )
    fig_q.update_layout(bargap=0.1, font=dict(size=14))
    st.plotly_chart(fig_q, use_container_width=True)

    # ───────────────────────────────
    # Tornado Sensitivity Analysis (±20% parameter changes)
    st.subheader(TXT["charts"]["tornado"]["title"])
    # Relative Impact Chart expander above the chart
    exp("charts.tornado.expander_title")
    params_for_sens = {
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
        "loss_unit": "loss_unit_v"
    }
    rows=[]
    for k, v in params_for_sens.items():
        lo = max(v * 0.8, 0)
        hi = (min(v * 1.2, 1) if k in ("a1", "a2", "a3", "b") else v * 1.2)
        # Build kwargs for compute_metrics, filling all required arguments
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
            "t3v": params["T3"]
        }
        kwargs_hi = kwargs_lo.copy()
        kwargs_lo[name_map.get(k, k)] = lo
        kwargs_hi[name_map.get(k, k)] = hi
        # Add qualv and schedv to both kwargs
        kwargs_lo["qualv"] = params["qual"]
        kwargs_lo["schedv"] = params["sched"]
        kwargs_hi["qualv"] = params["qual"]
        kwargs_hi["schedv"] = params["sched"]
        _, _, _, _, E_var_lo = compute_metrics(**kwargs_lo)
        _, _, _, _, E_var_hi = compute_metrics(**kwargs_hi)
        rel_delta_lo = abs(E_var_lo - E_total_x) / E_total_x * 100
        rel_delta_hi = abs(E_var_hi - E_total_x) / E_total_x * 100
        rows.append((k, max(rel_delta_lo, rel_delta_hi)))
    df_t=pd.DataFrame(rows, columns=["Parameter","RelChange"])\
           .sort_values("RelChange", ascending=False)
    # Standardize tornado axis/label
    fig_t = px.bar(
        df_t, x="RelChange", y="Parameter", orientation="h",
        color_discrete_sequence=["#000000"],
        labels={
            "RelChange": TXT["charts"]["tornado"]["xaxis"],
            "Parameter": ""
        }
    )
    fig_t.update_traces(text=df_t["RelChange"].map("{:.1f}%".format), textposition="auto", insidetextfont_color="white", outsidetextfont_color="gray")
    fig_t.update_layout(showlegend=False,
                        yaxis=dict(categoryorder="total ascending"),
                        font=dict(size=14),
                        bargap=0.1)
    st.plotly_chart(fig_t, use_container_width=True)

    # ───────────────────────────────
    # Compute elasticity-based relative and standardized sensitivities for E_total using unified config
    sens_config = [
        {
            "key": TXT["metrics"]["loss_unit"],
            "rel_val": dE_dL * loss_unit / E_total_x,   # ℓ の弾性
            "std_val": dE_dL * σL / σE
        },
        {
            "key": TXT["metrics"]["C"],
            "rel_val": dE_dC * C_x / E_total_x,         # C の弾性
            "std_val": dE_dC * σC / σE
        },
        {
            "key": TXT["metrics"]["succ"],
            "rel_val": dE_dS * S_x / E_total_x,         # S の弾性
            "std_val": dE_dS * σS / σE
        }
    ]
    order = [d["key"] for d in sens_config]
    df_rel = pd.DataFrame({
        "Parameter": [d["key"] for d in sens_config],
        "Relative Sensitivity": [abs(d["rel_val"]) for d in sens_config]
    })
    # Sensitivity Bar Chart Titles and Captions
    fig_rel = make_sensitivity_bar(
        df_rel,
        value_col="Relative Sensitivity",
        tick_fmt="{:.2f}",
        order=order
    )

    df_std = pd.DataFrame({
        "Parameter": [d["key"] for d in sens_config],
        "Std Sens": [abs(d["std_val"]) for d in sens_config]
    })
    fig_std = make_sensitivity_bar(
        df_std,
        value_col="Std Sens",
        tick_fmt="{:.3f}",
        order=order
    )

# Display Relative and Standardized Sensitivity charts side by side using st.columns(2)
col1, col2 = st.columns(2)
with col1:
    col1.subheader(TXT["charts"]["relative_sensitivity"]["title"])
    exp("charts.relative_sensitivity.expander_title")
    col1.plotly_chart(fig_rel, use_container_width=True)
with col2:
    col2.subheader(TXT["charts"]["standardized_sensitivity"]["title"])
    exp("charts.standardized_sensitivity.expander_title")
    col2.plotly_chart(fig_std, use_container_width=True)

# ───────────────────────────────
 # Monte Carlo Summary and output distribution visualization
st.subheader(TXT["charts"]["monte_carlo"]["title"])
# Monte Carlo Summary Statistics expander above the stats
exp("charts.monte_carlo.expander_title")

options = ["E_total", "Success S"]
with col_mc2:
    mc_var = col_mc2.selectbox(
        TXT["charts"]["monte_carlo"]["variable"],
        options,
        index=options.index(get_state("mc_var", "E_total")),
        key="mc_var"
    )
data = Evals if mc_var == "E_total" else Svals
ci_low, ci_high = np.percentile(data, [5, 95])
mean = data.mean()
median = np.median(data)
decimals = ".2f" if mc_var == "E_total" else ".4f"

# Display Monte Carlo summary statistics in a horizontal layout with large font using st.metric()
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        f"{TXT['charts']['monte_carlo']['mean']} {TXT['charts']['monte_carlo']['card_unit']}",
        f"{mean:.2f}"
    )
with col2:
    st.metric(
        f"{TXT['charts']['monte_carlo']['median']} {TXT['charts']['monte_carlo']['card_unit']}",
        f"{median:.2f}"
    )
with col3:
    st.metric(
        f"{TXT['charts']['monte_carlo']['ci']} {TXT['charts']['monte_carlo']['card_unit']}",
        f"{ci_low:.2f} – {ci_high:.2f}"
    )


# Histogram label localization for MC output
LABELS = {
    "E_total": "E_total: 効率（総合）" if lang == "日本語" else "E_total: Efficiency (Total)",
    "count": "頻度 / Frequency" if lang == "日本語" else "Count / Frequency",
}

# --- MC Output Distributions: Bar, Box, Violin, side by side ---
fig_mc_bar = px.histogram(
    data,
    nbins=100,
    labels={"value": LABELS["E_total"]},
    color_discrete_sequence=["black"]
)
fig_mc_bar.update_traces(
    marker_color="#000000",
    marker_line_color="#000000",
    marker_line_width=0.5,
    showlegend=False
)
# Compute MC summary statistics for vertical lines
mc_values = data
mean_val = np.mean(mc_values)
median_val = np.median(mc_values)
p5 = np.percentile(mc_values, 5)
p95 = np.percentile(mc_values, 95)
# Remove annotation_text and use only line_dash and line_color as per instructions
fig_mc_bar.add_vline(x=mean_val, line_dash="solid", line_color="#000000")
fig_mc_bar.add_vline(x=median_val, line_dash="solid", line_color="#000000")
fig_mc_bar.add_vline(x=p5, line_dash="dot", line_color="#000000")
fig_mc_bar.add_vline(x=p95, line_dash="dot", line_color="#000000")
fig_mc_bar.update_layout(
    yaxis_title=LABELS["count"],
    bargap=0.01,
    xaxis_tickfont_size=12,
    yaxis_tickfont_size=12,
    margin=dict(t=70)
)

st.plotly_chart(fig_mc_bar, use_container_width=True)

# Add language-dependent legend as a caption below the histogram
legend_label = {
    "en": "Legend: Mean (solid), Median (solid), 5–95% CI (dotted)",
    "ja": "凡例：平均値（実線）、中央値（実線）、信頼区間5–95%（点線）"
}
st.caption(legend_label["en"] if lang == "English" else legend_label["ja"])
