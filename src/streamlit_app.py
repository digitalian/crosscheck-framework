def get_text_labels(lang):
    LANG = "EN" if lang == "English" else "JA"
    return TXT_ALL[LANG]

#
# crosscheck_sim_promac_bw_v9_relative.py
# Streamlit ≥1.35 | pip install streamlit plotly numpy pandas sympy

import streamlit as st
import numpy as np
import sympy as sp
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# ───────────────────────────────
# Centralized core calculation function for the model
def compute_metrics(
    a1v=None, a2v=None, a3v=None, bv=None,
    cross_ratio_v=None, prep_post_ratio_v=None, loss_unit_v=None,
    qualv=None, schedv=None, t1v=None, t2v=None, t3v=None
):
    """
    Compute key output metrics (success rate, cost, loss-adjusted cost, efficiency, and total efficiency)
    for a given set of parameters. If any argument is not provided, it defaults to the current sidebar value.
    This allows reuse for scenario analysis and sensitivity calculations.
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
        xaxis_label = TXT["rel_xaxis"]
        bar_order = order
    elif "std" in value_col.lower():
        color = "#000000"
        xaxis_label = TXT["std_xaxis"]
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

LANG = "EN" if lang == "English" else "JA"
TXT_EN = {
        "input": "INPUT PANEL",
        "output": "KEY OUTPUT METRICS",
        "a_total": "a_total",
        "succ": "Success Rate S",
        "C": "Labor Cost C",
        "Closs": "C_total (with loss)",
        "E_base": "Efficiency E (baseline)",
        "E_total": "E_total (cost per success)",
        "tornado_title": "Tornado Sensitivity (±20%)",
        "tornado_explain": "This chart visualizes the effect of ±20% changes in key parameters on E_total (cost per success).  \nSelected parameters (a₁, a₂, a₃, b₀, CR, PP, ℓ) are core drivers of success, effort, and loss.  \nThis helps identify which inputs most strongly affect cost-efficiency.",
        "spider_title": "Standardized Sensitivity (∂E/∂x × σₓ/σ_E)",
        "spider_explain": "This chart quantifies the influence of each parameter on E_total, normalized by its variability.  \nStandardized sensitivity highlights how strongly each uncertain factor contributes to the variance of cost efficiency.  \nUseful for uncertainty-based risk assessment. Includes loss-adjusted cost (C_total) and the approximated effect of total labor time (T), estimated as the midpoint of standardized sensitivities of cost and loss.",
        "qs_title": "Quality × Schedule 2×2 Matrix",
        "qs_explain": "Each bar shows E_total under different combinations of quality and schedule, with labels showing the corresponding success rate.  \nThis helps compare cost-performance tradeoffs across operational scenarios.",
        "mc_title": "Monte Carlo Summary Statistics",
        "mc_explain": "The following parameters are assigned probabilistic distributions to capture plausible uncertainty ranges:\n- **a₁, a₂**: Normally distributed (mean = selected value, σ = 0.03), reflecting variation in basic process success rates due to human or environmental variability.\n- **a₃**: Triangular distribution (±10%) to reflect process-specific asymmetry in the final step's reliability.\n- **b₀**: Uniform between 0.70–0.90, assuming checker quality varies widely across contexts.\n- **Cross-ratio (CR)** and **Prep/Post ratio (PP)**: Triangular (±20%) around selected values to reflect managerial estimation variance.\n- **Loss unit ℓ**: Triangular (±20%) for capturing business risk variability.\n\nThese distributions are selected based on empirical heuristics: normal for stable processes (a₁, a₂), triangular for bounded uncertain estimates (a₃, CR, PP, ℓ), and uniform for quality variability (b₀).",
        "rel_title": "Relative Sensitivity (∂E/∂x × x/E)",
        "rel_explain": "This chart shows the elasticity of E_total with respect to each parameter, representing the impact from a 1% input change.  \nRelative sensitivity helps identify which parameters most affect cost efficiency in response to design or policy changes.  \nUseful for prioritizing improvement efforts. Includes loss-adjusted cost (C_total) and approximated effect of total labor time (T), computed as the mean of sensitivities of C and C_total.",
        "mc_variable": "MC variable",
        "rel_xaxis": "Relative Sensitivity (∂E/∂x × x/E)",
        "std_xaxis": "Standardized Sensitivity (ΔE/σ_E)",
        "T": "Labor Time T",
        "mean": "Mean",
        "median": "Median",
        "ci": "5–95% CI",
        "mc_caption": {
            "ja": "モンテカルロ法による出力分布。中央値・平均値は実線、信頼区間（5%–95%）は点線で表示。",
            "en": "Monte Carlo simulation output distribution. Solid lines indicate median and mean; dotted lines show the 5–95% CI."
        },
        # Added keys:
        "rel_sens_hint": "Efficiency per unit parameter value (E / x)",
        "std_sens_hint": "Elasticity of E_total w.r.t each parameter (∂E/∂x × x / E)",
        # Quality × Schedule expander keys
        "qs_expander_title": "📘 About Quality × Schedule Chart",
        "qs_expander_content": "Each bar shows E_total under different combinations of quality and schedule, with labels showing the corresponding success rate.\n\nThis helps compare cost-performance tradeoffs across operational scenarios.",
        # Relative Sensitivity expander
        "rel_sens_expander_title": "📘 About Relative Sensitivity",
        "rel_sens_expander_content": (
            "This chart shows the elasticity of E_total with respect to each parameter, representing the impact from a 1% input change.  \n"
            "Relative sensitivity helps identify which parameters most affect cost efficiency in response to design or policy changes.  \n"
            "Useful for prioritizing improvement efforts. Includes loss-adjusted cost (C_total) and approximated effect of total labor time (T), "
            "computed as the mean of sensitivities of C and C_total."
        ),
        # Standardized Sensitivity expander
        "std_sens_expander_title": "📘 About Standardized Sensitivity",
        "std_sens_expander_content": (
            "This chart quantifies the influence of each parameter on E_total, normalized by its variability.  \n"
            "Standardized sensitivity highlights how strongly each uncertain factor contributes to the variance of cost efficiency.  \n"
            "Useful for uncertainty-based risk assessment. Includes loss-adjusted cost (C_total) and the approximated effect of total labor time (T), "
            "estimated as the midpoint of standardized sensitivities of cost and loss."
        ),
        # --- Added by user request ---
        "impact_chart_expander_title": "📘 Explanation: Impact of ±20% Parameter Changes",
        "impact_chart_expander_content": (
            "This chart visualizes the effect of ±20% changes in key parameters on E_total (cost per success).  \n"
            "Selected parameters (a₁, a₂, a₃, b₀, CR, PP, ℓ) are core drivers of success, effort, and loss.  \n"
            "This helps identify which inputs most strongly affect cost-efficiency."
        ),
        "mc_summary_expander_title": "📘 Explanation: Monte Carlo Parameter Distributions",
        "mc_summary_expander_content": (
            "The following parameters are assigned probabilistic distributions to capture plausible uncertainty ranges:  \n\n"
            "- **a₁, a₂**: Normally distributed (mean = selected value, σ = 0.03), reflecting variation in basic process success rates due to human or environmental variability.  \n"
            "- **a₃**: Triangular distribution (±10%) to reflect process-specific asymmetry in the final step's reliability.  \n"
            "- **b₀**: Uniform between 0.70–0.90, assuming checker quality varies widely across contexts.  \n"
            "- **Cross-ratio (CR), Prep/Post ratio (PP)**: Triangular (±20%) around selected values to reflect managerial estimation variance.  \n"
            "- **Loss unit ℓ**: Triangular (±20%) for capturing business risk variability.  \n\n"
            "These distributions are selected based on empirical heuristics: normal for stable processes (a₁, a₂), triangular for bounded uncertain estimates (a₃, CR, PP, ℓ), and uniform for quality variability (b₀)."
        )
    }
TXT_JA = {
        "input": "入力パネル",
        "output": "主要出力指標",
        "a_total": "a_total",
        "succ": "成功率 S",
        "C": "C（作業工数）",
        "Closs": "C_total（損失込）",
        "E_base": "効率 E（ベースライン）",
        "E_total": "E_total（成功1件あたりの総コスト）",
        "tornado_title": "トルネード感度分析（±20%）",
        "tornado_explain": "このチャートは、主要パラメータを±20%変化させた際のE_total（成功1件あたりの総コスト）への影響を可視化します。  \n対象パラメータ（a₁, a₂, a₃, b₀, CR, PP, ℓ）は、成功率・作業工数・損失額に影響を与える主要因として選定しています。  \nこれにより、コスト効率に最も影響を与える要因を特定できます。",
        "spider_title": "標準化感度（∂E/∂x × σₓ/σ_E）",
        "spider_explain": "各パラメータのばらつきを基準にE_totalへの影響度を標準化して定量化します。  \n標準化感度は、コスト効率に対する不確実性（ばらつき）の寄与を示し、リスク評価に有効です。  \n損失込みのコスト（C_total）と全体作業時間（T）の影響も含めています。T の標準化感度は C と L の中間として近似しています。",
        "qs_title": "品質×納期の2×2マトリクス",
        "qs_explain": "品質と納期の組み合わせごとのE_totalを棒グラフで示し、ラベルとして成功率を表示します。  \n運用シナリオごとのコストパフォーマンスの比較に役立ちます。",
        "mc_title": "モンテカルロ要約統計",
        "mc_explain": "以下のパラメータに確率的な揺らぎを与え、不確実性をモデル化しています：\n- **a₁, a₂**：平均を中心とした正規分布（σ=0.03）、人的または環境要因による変動を想定。\n- **a₃**：±10%の三角分布。最終工程に特有の非対称性を考慮。\n- **b₀**：0.70～0.90の一様分布。チェック品質の個人差を反映。\n- **クロスチェック比率（CR）・準備/事後比率（PP）**：±20%の三角分布。マネジメント判断のばらつきを想定。\n- **損失単位 ℓ**：±20%の三角分布。ビジネスリスクのばらつきを反映。\n\nこれらの分布は、経験的な判断に基づいて選定しています。",
        "rel_title": "相対感度（∂E/∂x × x/E）",
        "rel_explain": "各パラメータを1%変更した際のE_totalへの影響度（弾性）を示します。  \n相対感度は設計変更や方針変更による影響度の大きさを示し，改善の優先順位づけに有効です。  \n損失込みのコスト（C_total）と全体作業時間（T）の影響も含めています。T の相対感度は C と C_total の平均として近似しています。",
        "mc_variable": "MC対象変数",
        "rel_xaxis": "相対感度（∂E/∂x × x/E）",
        "std_xaxis": "標準化感度（ΔE/σ_E）",
        "T": "作業時間 T",
        "mean": "平均",
        "median": "中央値",
        "ci": "5〜95%信頼区間",
        "mc_caption": {
            "ja": "モンテカルロ法による出力分布。中央値・平均値は実線、信頼区間（5%–95%）は点線で表示。",
            "en": "Monte Carlo simulation output distribution. Solid lines indicate median and mean; dotted lines show the 5–95% CI."
        },
        # Added keys:
        "rel_sens_hint": "各パラメータ単位あたりの効率（E ÷ 値）",
        "std_sens_hint": "各パラメータに対する感度（∂E/∂x × x / E）",
        # Quality × Schedule expander keys
        "qs_expander_title": "📘 品質 × 納期グラフについて",
        "qs_expander_content": "各棒グラフは、品質と納期の組み合わせごとの E_total（総合効率）を示しており、棒のラベルにはその時の成功率が表示されています。\n\nこれにより、異なる運用シナリオにおけるコストとパフォーマンスのバランスを比較できます。",
        # Relative Sensitivity expander
        "rel_sens_expander_title": "📘 相対感度グラフについて",
        "rel_sens_expander_content": (
            "各パラメータを1%変更した際のE_totalへの影響度（弾性）を示します。  \n"
            "相対感度は設計変更や方針変更による影響度の大きさを示し，改善の優先順位づけに有効です。  \n"
            "損失込みのコスト（C_total）と全体作業時間（T）の影響も含めています。T の相対感度は C と C_total の平均として近似しています。"
        ),
        # Standardized Sensitivity expander
        "std_sens_expander_title": "📘 標準化感度グラフについて",
        "std_sens_expander_content": (
            "各パラメータのばらつきを基準にE_totalへの影響度を標準化して定量化します。  \n"
            "標準化感度は、コスト効率に対する不確実性（ばらつき）の寄与を示し、リスク評価に有効です。  \n"
            "損失込みのコスト（C_total）と全体作業時間（T）の影響も含めています。T の標準化感度は C と L の中間として近似しています。"
        ),
        # --- Added by user request ---
        "impact_chart_expander_title": "📘 説明：パラメータ±20％変化の影響",
        "impact_chart_expander_content": (
            "このグラフは，主要パラメータを±20％変動させたときのE_total（成功あたりコスト）への影響を可視化します。  \n"
            "対象パラメータ（a₁, a₂, a₃, b₀, CR, PP, ℓ）は，成功率・作業コスト・損失の主要因です。  \n"
            "どのパラメータがコスト効率に強く影響するかを把握できます。"
        ),
        "mc_summary_expander_title": "📘 説明：モンテカルロにおけるパラメータ分布",
        "mc_summary_expander_content": (
            "以下のパラメータに確率分布を設定し，不確実性の範囲をモデル化しています：  \n\n"
            "- **a₁, a₂**：平均値を中心とする正規分布（σ = 0.03）。作業成功率の人為的・環境的ばらつきを反映。  \n"
            "- **a₃**：±10%の三角分布。工程ごとの非対称性を想定。  \n"
            "- **b₀**：0.70～0.90の一様分布。確認者品質の個人差を想定。  \n"
            "- **CR・PP**：±20%の三角分布。見積誤差を想定。  \n"
            "- **損失単位 ℓ**：±20%の三角分布。事業リスクの不確実性を表現。  \n\n"
            "分布の選定は経験則に基づいています。正規分布は安定した工程，三角分布は管理上の見積不確実性，一様分布は品質ばらつきを想定します。"
        )
    }
TXT_ALL = {"EN": TXT_EN, "JA": TXT_JA}
TXT = get_text_labels(lang)

# ───────────────────────────────

# ───────────────────────────────
# Sidebar input encapsulation with session state
def get_sidebar_params():
    st.sidebar.title(TXT["input"])  
    # a1
    if "a1" not in st.session_state:
        st.session_state.a1 = 0.95
    a1 = st.sidebar.slider("a1 (step 1 success rate)", 0.5, 1.0, st.session_state.a1, 0.01, key="a1")
    # a2
    if "a2" not in st.session_state:
        st.session_state.a2 = 0.95
    a2 = st.sidebar.slider("a2 (step 2 success rate)", 0.5, 1.0, st.session_state.a2, 0.01, key="a2")
    # a3
    if "a3" not in st.session_state:
        st.session_state.a3 = 0.80
    a3 = st.sidebar.slider("a3 (step 3 success rate)", 0.5, 1.0, st.session_state.a3, 0.01, key="a3")
    # b0
    if "b0" not in st.session_state:
        st.session_state.b0 = 0.80
    b0 = st.sidebar.slider("b0 (checker success rate)", 0.0, 1.0, st.session_state.b0, 0.01, key="b0")
    # Group Quality and Schedule side-by-side
    col_qs1, col_qs2 = st.sidebar.columns(2)
    if "qual" not in st.session_state:
        st.session_state.qual = "Standard"
    with col_qs1:
        qual = st.selectbox("Qual-Grade", ["Standard", "Low"], index=0 if st.session_state.qual == "Standard" else 1, key="qual")
    if "sched" not in st.session_state:
        st.session_state.sched = "OnTime"
    with col_qs2:
        sched = st.selectbox("Schedule", ["OnTime", "Late"], index=0 if st.session_state.sched == "OnTime" else 1, key="sched")
    # loss_unit
    if "loss_unit" not in st.session_state:
        st.session_state.loss_unit = 0.0
    loss_unit = st.sidebar.slider("Loss unit ℓ", 0.0, 50.0, st.session_state.loss_unit, 0.1, key="loss_unit")
    # T1, T2, T3 grouped in a single row
    if "T1" not in st.session_state:
        st.session_state.T1 = 10
    if "T2" not in st.session_state:
        st.session_state.T2 = 10
    if "T3" not in st.session_state:
        st.session_state.T3 = 30
    col_t1, col_t2, col_t3 = st.sidebar.columns(3)
    with col_t1:
        T1 = st.number_input("T1 (h)", 0, 200, st.session_state.T1, key="T1")
    with col_t2:
        T2 = st.number_input("T2 (h)", 0, 200, st.session_state.T2, key="T2")
    with col_t3:
        T3 = st.number_input("T3 (h)", 0, 200, st.session_state.T3, key="T3")
    # cross_ratio and prep_post_ratio side-by-side
    if "cross_ratio" not in st.session_state:
        st.session_state.cross_ratio = 0.30
    if "prep_post_ratio" not in st.session_state:
        st.session_state.prep_post_ratio = 0.40
    col_cr, col_pp = st.sidebar.columns(2)
    with col_cr:
        cross_ratio = st.slider("Cross-ratio", 0.0, 0.5, st.session_state.cross_ratio, 0.01, key="cross_ratio")
    with col_pp:
        prep_post_ratio = st.slider("Prep+Post ratio", 0.0, 0.5, st.session_state.prep_post_ratio, 0.01, key="prep_post_ratio")
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

# ───────────────────────────────
# Run Monte Carlo simulations to estimate output variability
# Using normal, triangular, and uniform distributions depending on parameter characteristics
if "sample_n" not in st.session_state:
    st.session_state.sample_n = 100000
# Place Sample Size and MC Variable selection side-by-side
col_mc1, col_mc2 = st.sidebar.columns(2)
with col_mc1:
    sample_n = st.number_input(
        "Sample Size", 1000, 1000000, value=st.session_state.sample_n, step=10000, key="sample_n"
    )
N = sample_n
rng = np.random.default_rng(0)
a1s = rng.normal(params["a1"], 0.03, N).clip(0, 1)  # a1, a2: normal for stable process uncertainty
a2s = rng.normal(params["a2"], 0.03, N).clip(0, 1)
a3s = rng.triangular(params["a3"] * 0.9, params["a3"], params["a3"] * 1.1, N).clip(0, 1) if params["a3"] > 0 else np.zeros(N)  # a3: triangular
b0s = rng.uniform(0.70, 0.90, N).clip(0, 1)  # b0: uniform for checker variability
t1s = rng.normal(params["T1"], 0.5, N).clip(1)
t2s = rng.normal(params["T2"], 0.5, N).clip(1)
t3s = rng.normal(params["T3"], 0.5, N).clip(1)
cross_ratios = (
    rng.triangular(params["cross_ratio"] * 0.8, params["cross_ratio"], params["cross_ratio"] * 1.2, N)
    if params["cross_ratio"] > 0 else np.zeros(N)
)
prep_post_ratios = (
    rng.triangular(params["prep_post_ratio"] * 0.8, params["prep_post_ratio"], params["prep_post_ratio"] * 1.2, N)
    if params["prep_post_ratio"] > 0 else np.zeros(N)
)
loss_units = (
    rng.triangular(params["loss_unit"] * 0.8, params["loss_unit"], params["loss_unit"] * 1.2, N)
    if params["loss_unit"] > 0 else np.zeros(N)
)

# Preallocate arrays for MC results
Evals = np.empty(N)
Svals = np.empty(N)
Cvals = np.empty(N)
for i in range(N):
    # For each MC sample, calculate output metrics with sampled parameters
    at = a1s[i] * a2s[i] * a3s[i]
    be = b0s[i] * qual_B * sched_B
    si = 1 - (1 - at) * (1 - be)
    ci = (params["T1"] + params["T2"] + params["T3"]) * qual_T * sched_T * (1 + cross_ratios[i] + prep_post_ratios[i])
    Evals[i] = (ci + loss_units[i] * ci * (1 - si)) / si
    Svals[i] = si
    Cvals[i] = ci

# Calculate standard deviations for use in standardized sensitivity
σE = Evals.std()
σL = loss_units.std()
σC = Cvals.std()
σS = Svals.std()

# ───────────────────────────────

# Symbolic derivatives utility function
def symbolic_derivatives(C, S, L):
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
# ───────────────────────────────
# Output metrics and scenario analysis
left, right = st.columns([1, 2])
with left:
    # Output metrics for current scenario (main panel)
    st.subheader(TXT["output"])
    st.metric(TXT["a_total"], f"{a_total:.4f}")
    S_x, C_x, C_loss_x, E_x, E_total_x = compute_metrics(
        a1v=params["a1"], a2v=params["a2"], a3v=params["a3"], bv=params["b0"],
        cross_ratio_v=params["cross_ratio"], prep_post_ratio_v=params["prep_post_ratio"],
        loss_unit_v=params["loss_unit"], t1v=params["T1"], t2v=params["T2"], t3v=params["T3"]
    )
    st.metric(TXT["succ"],    f"{S_x:.2%}")
    st.metric(TXT["C"],       f"{C_x:.1f}")
    st.metric(TXT["Closs"],   f"{C_loss_x:.1f}")
    st.metric(TXT["E_base"],  f"{E_x:.1f}")
    st.metric(TXT["E_total"], f"{E_total_x:.1f}")


with right:
    # ───────────────────────────────
    # Quality × Schedule scenario matrix
    st.subheader(TXT['qs_title'])
    with st.expander(TXT["qs_expander_title"], expanded=False):
        st.markdown(TXT["qs_expander_content"])
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
    fig_q = px.bar(df_bars, x="Scenario", y="E_total", text="S",
                   color_discrete_sequence=["#000000"],
                   labels={"E_total": TXT["E_total"], "S": TXT["succ"], "Scenario": "Scenario"})
    fig_q.update_traces(
        textposition="auto",
        insidetextfont_color="white",
        outsidetextfont_color="gray"
    )
    fig_q.update_layout(bargap=0.1, font=dict(size=14))
    st.plotly_chart(fig_q, use_container_width=True)

    # ───────────────────────────────
    # Tornado Sensitivity Analysis (±20% parameter changes)
    st.subheader(TXT['tornado_title'])
    # Relative Impact Chart expander above the chart
    st.markdown("### " + TXT["impact_chart_title"] if "impact_chart_title" in TXT else TXT["tornado_title"])
    with st.expander(TXT["impact_chart_expander_title"], expanded=False):
        st.markdown(TXT["impact_chart_expander_content"])
    dark, light = "#000000", "#000000"
    params_for_sens = {
        "a1": params["a1"],
        "a2": params["a2"],
        "a3": params["a3"],
        "b": params["b0"],
        "cross_ratio": params["cross_ratio"],
        "prep_post_ratio": params["prep_post_ratio"],
        "loss_unit": params["loss_unit"],
        "T": params["T1"] + params["T2"] + params["T3"]
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
        if k == "T":
            # For total labor time, vary T1/T2/T3 proportionally
            base = params["T1"] + params["T2"] + params["T3"]
            dT = base * 0.2
            for label, base_val, lo_val, hi_val in [("T", base, base - dT, base + dT)]:
                lo_T = lo_val * np.array([params["T1"], params["T2"], params["T3"]]) / base if base > 0 else np.array([0,0,0])
                hi_T = hi_val * np.array([params["T1"], params["T2"], params["T3"]]) / base if base > 0 else np.array([0,0,0])
                _, _, _, _, E_var_lo = compute_metrics(
                    a1v=params["a1"], a2v=params["a2"], a3v=params["a3"], bv=params["b0"],
                    cross_ratio_v=params["cross_ratio"], prep_post_ratio_v=params["prep_post_ratio"],
                    loss_unit_v=params["loss_unit"], t1v=lo_T[0], t2v=lo_T[1], t3v=lo_T[2]
                )
                _, _, _, _, E_var_hi = compute_metrics(
                    a1v=params["a1"], a2v=params["a2"], a3v=params["a3"], bv=params["b0"],
                    cross_ratio_v=params["cross_ratio"], prep_post_ratio_v=params["prep_post_ratio"],
                    loss_unit_v=params["loss_unit"], t1v=hi_T[0], t2v=hi_T[1], t3v=hi_T[2]
                )
                rel_delta_lo = abs(E_var_lo - E_total_x) / E_total_x * 100
                rel_delta_hi = abs(E_var_hi - E_total_x) / E_total_x * 100
                rows.append((k, max(rel_delta_lo, rel_delta_hi)))
            continue
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
        _, _, _, _, E_var_lo = compute_metrics(**kwargs_lo)
        _, _, _, _, E_var_hi = compute_metrics(**kwargs_hi)
        rel_delta_lo = abs(E_var_lo - E_total_x) / E_total_x * 100
        rel_delta_hi = abs(E_var_hi - E_total_x) / E_total_x * 100
        rows.append((k, max(rel_delta_lo, rel_delta_hi)))
    df_t=pd.DataFrame(rows, columns=["Parameter","RelChange"])\
           .sort_values("RelChange", ascending=False)
    maxd=df_t["RelChange"].max()
    df_t["color"]=np.where(df_t["RelChange"]==maxd, dark, light)
    # Standardize tornado axis/label
    fig_t=px.bar(df_t, x="RelChange", y="Parameter", orientation="h",
                 color="color", color_discrete_map={dark:dark, light:light},
                 labels={"RelChange":"|ΔE/E| (%)","Parameter": TXT["rel_xaxis"]})
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
            "key": TXT["succ"],
            "rel_val": dE_dS * S_x / E_total_x,
            "std_val": dE_dS * σS / σE
        },
        {
            "key": TXT["C"],
            "rel_val": dE_dC * C_x / E_total_x,
            "std_val": dE_dC * σC / σE
        },
        {
            "key": TXT["Closs"],
            "rel_val": dE_dC * C_loss_x / E_total_x,
            "std_val": dE_dL * σL / σE
        },
        {
            "key": TXT["T"],
            # For T, rel_val and std_val are averaged as in previous logic
            "rel_val": None,
            "std_val": None
        }
    ]
    # Fill in T's rel_val and std_val as the mean of C and Closs sensitivities
    sens_config[3]["rel_val"] = (
        (sens_config[1]["rel_val"] + sens_config[2]["rel_val"]) / 2
    )
    sens_config[3]["std_val"] = (
        (abs(sens_config[1]["std_val"]) + abs(sens_config[2]["std_val"])) / 2
    )
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
    col1.markdown("### " + TXT["rel_title"])
    with col1.expander(TXT["rel_sens_expander_title"], expanded=False):
        st.markdown(TXT["rel_sens_expander_content"])
    col1.plotly_chart(fig_rel, use_container_width=True)
with col2:
    col2.markdown("### " + TXT["spider_title"])
    with col2.expander(TXT["std_sens_expander_title"], expanded=False):
        st.markdown(TXT["std_sens_expander_content"])
    col2.plotly_chart(fig_std, use_container_width=True)

# ───────────────────────────────
# Monte Carlo Summary and output distribution visualization
st.subheader(TXT['mc_title'])
# Monte Carlo Summary Statistics expander above the stats
st.markdown("### " + TXT["mc_summary_title"] if "mc_summary_title" in TXT else TXT["mc_title"])
with st.expander(TXT["mc_summary_expander_title"], expanded=False):
    st.markdown(TXT["mc_summary_expander_content"])

options = ["E_total", "Success S"]
if "mc_var" not in st.session_state:
    st.session_state.mc_var = "E_total"
with col_mc2:
    mc_var = st.selectbox(
        TXT["mc_variable"],
        options,
        index=options.index(st.session_state.mc_var),
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
    st.metric("Mean", f"{mean:.2f}")
with col2:
    st.metric("Median", f"{median:.2f}")
with col3:
    st.metric("5–95% CI", f"{ci_low:.2f} – {ci_high:.2f}")


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
fig_mc_bar.add_vline(x=median_val, line_dash="solid", line_color="#000000")
fig_mc_bar.add_vline(x=p5, line_dash="dot", line_color="#000000")
fig_mc_bar.add_vline(x=p95, line_dash="dot", line_color="#000000")
fig_mc_bar.update_layout(
    yaxis_title=LABELS["count"],
    bargap=0.01,
    xaxis_tickfont_size=12,
    yaxis_tickfont_size=12
)

st.plotly_chart(fig_mc_bar, use_container_width=True)

st.caption(TXT["mc_caption"]["ja"] if lang == "日本語" else TXT["mc_caption"]["en"])