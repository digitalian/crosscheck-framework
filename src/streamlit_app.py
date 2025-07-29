# ──────────────────────────────────────────────────────────────
#  Cross‑Check Framework  •  Streamlit Simulation App
#  v2025‑07‑29  –  Sobol global sensitivity & PCG64DXSM RNG
# ──────────────────────────────────────────────────────────────
#  (c) 2025 digitalian  –  MIT License
# ----------------------------------------------------------------
#  Changelog
#   • NEW  : Sobol first‑order/global sensitivity chart (SALib)
#   • NEW  : PCG64DXSM BitGenerator for faster, robust RNG
#   • FIX  : BUG‑1, BUG‑4  (see previous commits)
# ----------------------------------------------------------------

from __future__ import annotations

import streamlit as st
import numpy as np, math
import sympy as sp
import pandas as pd
import plotly.express as px
from typing import Tuple, Dict, List

# ── New: SALib for Sobol ────────────────────────────────────────
try:
    # SALib ≥1.5 推奨ルート
    from SALib.sample.sobol import sample as sobol_sample
except ImportError:          # SALib ≤1.4 fallback
    from SALib.sample import saltelli as sobol_sample
from SALib.analyze import sobol

# ── New: PCG64DXSM generator (faster & parallel‑safe) ──────────
from numpy.random import PCG64DXSM, Generator

# ╔══════════════════════════════════════════════════════════════╗
#  Section 1  •  Helper utilities
# ╚══════════════════════════════════════════════════════════════╝
def get_text_labels(lang: str) -> Dict:
    """Return language‑specific label dictionary."""
    LANG = "EN" if lang == "English" else "JA"
    return TXT_ALL[LANG]


def get_state(key, default):
    """Retrieve or initialise a value in `st.session_state`."""
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def exp(path: str):
    """
    Create an expander from nested TXT dict using a dot‑separated key path.
    Example: 'charts.tornado.expander_title'
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

# ╔══════════════════════════════════════════════════════════════╗
#  Section 2  •  Core deterministic model
# ╚══════════════════════════════════════════════════════════════╝
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
    """Return S, C, C_loss, E, E_total for a single scenario."""
    # Multipliers
    qual_T, qual_B = (1, 1) if qualv == "Standard" else (2 / 3, 0.8)
    sched_T, sched_B = (1, 1) if schedv == "OnTime" else (2 / 3, 0.8)

    # Success probability
    a_tot = a1v * a2v * a3v
    b_eff = bv * qual_B * sched_B
    S_x = 1 - (1 - a_tot) * (1 - b_eff)

    # Costs
    T = (t1v + t2v + t3v) * qual_T * sched_T
    C_x = T * (1 + cross_ratio_v + prep_post_ratio_v)
    C_loss_x = C_x + loss_unit_v * C_x * (1 - S_x)

    # Efficiencies
    E_x = C_x / S_x
    E_total_x = C_loss_x / S_x
    return S_x, C_x, C_loss_x, E_x, E_total_x


# ╔══════════════════════════════════════════════════════════════╗
#  Section 3  •  Sensitivity‑plot helper
# ╚══════════════════════════════════════════════════════════════╝
def make_sensitivity_bar(
    df: pd.DataFrame, value_col: str, tick_fmt: str = "{:.2f}", order=None
):
    """Horizontal bar (Plotly) for sensitivity tables."""
    df = df.copy()
    df[value_col] = df[value_col].abs()
    if order:
        df["Parameter"] = pd.Categorical(df["Parameter"], categories=order, ordered=True)
    fig = px.bar(
        df,
        x=value_col,
        y="Parameter",
        orientation="h",
        text=df[value_col].map(tick_fmt.format),
        color_discrete_sequence=["#000000"],
    )
    fig.update_traces(
        texttemplate="%{text}",
        insidetextfont_color="white",
        outsidetextfont_color="gray",
    )
    fig.update_layout(
        showlegend=False,
        yaxis=dict(categoryorder="array", categoryarray=order) if order else {},
        font=dict(size=14),
        bargap=0.1,
        margin=dict(t=30, b=40),
    )
    return fig


# ╔══════════════════════════════════════════════════════════════╗
#  Section 4  •  Streamlit‑wide config & localisation strings
# ╚══════════════════════════════════════════════════════════════╝
st.set_page_config(page_title="Cross‑Check Simulator", layout="wide")

if "lang" not in st.session_state:
    st.session_state.lang = "English"
lang = st.sidebar.radio("Language / 言語", ["English", "日本語"], index=0, key="lang", horizontal=True)

# --- Localisation dictionaries (EN & JA) ---
#   *Sobol 部分の新ラベルを追記*
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
        "sobol": {
            "title": "Global Sensitivity (Sobol S₁)",
            "expander_title": "📘 Sobol Global Sensitivity",
            "expander_content": (
                "Variance‑based global sensitivity analysis using Saltelli sampling (N×(k+2) runs). "
                "Bars show the first‑order Sobol index S₁ for each parameter; "
                "higher values = greater contribution to output variance."
            ),
            "xaxis": "Sobol S₁",
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
        "sobol": {
            "title": "グローバル感度 (Sobol S₁)",
            "expander_title": "📘 Sobol グローバル感度",
            "expander_content": (
                "Saltelli サンプリングによる Sobol 一次指数 (S₁)。"
                "E_total の分散に対する各パラメータの寄与度を示します。"
            ),
            "xaxis": "Sobol S₁",
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

# ╔══════════════════════════════════════════════════════════════╗
#  Section 5  •  Sidebar inputs
# ╚══════════════════════════════════════════════════════════════╝
def get_sidebar_params() -> Dict[str, float]:
    st.sidebar.title(TXT["panel"]["input"])
    # 1) Success rates
    a1 = st.sidebar.slider("a1 (step 1)", 0.5, 1.0, get_state("a1", 0.95), 0.01)
    a2 = st.sidebar.slider("a2 (step 2)", 0.5, 1.0, get_state("a2", 0.95), 0.01)
    a3 = st.sidebar.slider("a3 (step 3)", 0.5, 1.0, get_state("a3", 0.80), 0.01)
    # 2) Task times
    col_t1, col_t2, col_t3 = st.sidebar.columns(3)
    with col_t1:
        T1 = st.number_input("T1 [h]", 0, 200, get_state("T1", 10), key="T1")
    with col_t2:
        T2 = st.number_input("T2 [h]", 0, 200, get_state("T2", 10), key="T2")
    with col_t3:
        T3 = st.number_input("T3 [h]", 0, 200, get_state("T3", 30), key="T3")
    # 3) Quality & Schedule
    col_q, col_s = st.sidebar.columns(2)
    with col_q:
        qual = st.selectbox("Quality", ["Standard", "Low"], index=0 if get_state("qual", "Standard") == "Standard" else 1, key="qual")
    with col_s:
        sched = st.selectbox("Schedule", ["OnTime", "Late"], index=0 if get_state("sched", "OnTime") == "OnTime" else 1, key="sched")
    # 4) Checker
    st.sidebar.markdown("---")
    b0 = st.sidebar.slider("b0 (checker)", 0.0, 1.0, get_state("b0", 0.80), 0.01)
    # 5) Cost ratios
    cross_ratio = st.sidebar.slider("Cross‑ratio", 0.0, 0.5, get_state("cross_ratio", 0.30), 0.01)
    prep_post_ratio = st.sidebar.slider("Prep/Post ratio", 0.0, 0.5, get_state("prep_post_ratio", 0.40), 0.01)
    # 6) Loss unit
    st.sidebar.markdown("---")
    loss_unit = st.sidebar.slider("Loss unit ℓ", 0.0, 50.0, get_state("loss_unit", 0.0), 0.1)
    # 7) Monte Carlo
    st.sidebar.markdown("<hr style='border-top:3px solid black'>", unsafe_allow_html=True)
    st.sidebar.title("Monte‑Carlo")
    col_n, col_var = st.sidebar.columns(2)
    with col_n:
        sample_n = st.number_input("Samples", 1_000, 1_000_000, get_state("sample_n", 100_000), step=10_000)
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

params = get_sidebar_params()

# ╔══════════════════════════════════════════════════════════════╗
#  Section 6  •  Monte‑Carlo simulation  (PCG64DXSM)
# ╚══════════════════════════════════════════════════════════════╝
@st.cache_data(show_spinner=False, ttl=900)
def run_mc(p: Dict[str, float], N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised Monte‑Carlo returning Evals, Svals, Cvals, L_samples."""
    rng = Generator(PCG64DXSM(seed=0))

    # Success params
    a1s = rng.normal(p["a1"], 0.03, N).clip(0, 1)
    a2s = rng.normal(p["a2"], 0.03, N).clip(0, 1)
    a3s = rng.triangular(p["a3"] * 0.9, p["a3"], p["a3"] * 1.1, N).clip(0, 1)
    b0s = rng.uniform(0.70, 0.90, N)

    # Task times (±10 %)
    t1s = rng.normal(p["T1"], p["T1"] * 0.1, N).clip(min=1)
    t2s = rng.normal(p["T2"], p["T2"] * 0.1, N).clip(min=1)
    t3s = rng.normal(p["T3"], p["T3"] * 0.1, N).clip(min=1)

    # Cost ratios
    cross_ratios = rng.triangular(p["cross_ratio"] * 0.8, p["cross_ratio"], p["cross_ratio"] * 1.2, N) if p["cross_ratio"] > 0 else np.zeros(N)
    prep_post_ratios = rng.triangular(p["prep_post_ratio"] * 0.8, p["prep_post_ratio"], p["prep_post_ratio"] * 1.2, N) if p["prep_post_ratio"] > 0 else np.zeros(N)
    L_samples = rng.triangular(p["loss_unit"] * 0.8, p["loss_unit"], p["loss_unit"] * 1.2, N) if p["loss_unit"] > 0 else np.zeros(N)

    # Multipliers
    qual_T, qual_B = (1, 1) if p["qual"] == "Standard" else (2 / 3, 0.8)
    sched_T, sched_B = (1, 1) if p["sched"] == "OnTime" else (2 / 3, 0.8)

    a_tot = a1s * a2s * a3s
    b_eff = b0s * qual_B * sched_B
    Svals = 1 - (1 - a_tot) * (1 - b_eff)

    Tvals = (t1s + t2s + t3s) * qual_T * sched_T
    Cvals = Tvals * (1 + cross_ratios + prep_post_ratios)
    Evals = (Cvals + L_samples * Cvals * (1 - Svals)) / Svals
    return Evals, Svals, Cvals, L_samples


Evals, Svals, Cvals, L_samples = run_mc(params, int(params["sample_n"]))

# Std‑devs
σE, σC, σS, σL = Evals.std(), Cvals.std(), Svals.std(), (Cvals * L_samples).std()

# ╔══════════════════════════════════════════════════════════════╗
#  Section 7  •  Sobol global sensitivity  (PCG64DXSM)
# ╚══════════════════════════════════════════════════════════════╝
@st.cache_data(show_spinner=False, ttl=900)
def run_sobol(p: Dict[str, float], N: int = 10_000) -> pd.DataFrame:
    """Return DataFrame with first‑order Sobol indices."""
    # Problem definition
    problem = {
        "num_vars": 7,
        "names": ["a1", "a2", "a3", "b0", "CR", "PP", "L"],
        "bounds": [
            [0.5, 1.0],
            [0.5, 1.0],
            [0.5, 1.0],
            [0.0, 1.0],
            [0.0, 0.5],
            [0.0, 0.5],
            [0.0, 50.0],
        ],
    }
    # SALib ≤1.5: seed 引数は非対応。NumPy にシード固定で再現性を担保
    np.random.seed(0)
    # パッチ②: N を 2^n に丸める
    N_pow2 = 2 ** math.ceil(math.log2(N))
    if N_pow2 != N:
        st.info(f"Sobol sample数を {N} → {N_pow2} に調整（2^n 必須）")
    X = sobol_sample(problem, N_pow2, calc_second_order=False)
    # Vectorised model: compute E_total for each row
    qual_T, qual_B = (1, 1) if p["qual"] == "Standard" else (2 / 3, 0.8)
    sched_T, sched_B = (1, 1) if p["sched"] == "OnTime" else (2 / 3, 0.8)

    a_tot = X[:, 0] * X[:, 1] * X[:, 2]
    b_eff = X[:, 3] * qual_B * sched_B
    Sarr = 1 - (1 - a_tot) * (1 - b_eff)

    # Costs
    T_base = (p["T1"] + p["T2"] + p["T3"]) * qual_T * sched_T  # keep times constant
    C_base = T_base * (1 + X[:, 4] + X[:, 5])
    E_total_arr = (C_base + X[:, 6] * C_base * (1 - Sarr)) / Sarr

    Si = sobol.analyze(problem, E_total_arr, calc_second_order=False, print_to_console=False)
    df = pd.DataFrame({"Parameter": problem["names"], "S1": Si["S1"], "ST": Si["ST"]})
    return df.sort_values("S1", ascending=False)


df_sobol = run_sobol(params)

# ╔══════════════════════════════════════════════════════════════╗
#  Section 8  •  Deterministic baseline computation
# ╚══════════════════════════════════════════════════════════════╝
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

# ╔══════════════════════════════════════════════════════════════╗
#  Section 9  •  Relative & standardized local elasticities
# ╚══════════════════════════════════════════════════════════════╝
def symbolic_derivatives(C_: float, S_: float, L_: float) -> Dict[str, float]:
    C_sym, S_sym, L_sym = sp.symbols("C S L")
    E_expr = (C_sym + C_sym * L_sym * (1 - S_sym)) / S_sym
    return {
        "dE_dC": float(sp.diff(E_expr, C_sym).subs({C_sym: C_, S_sym: S_, L_sym: L_})),
        "dE_dS": float(sp.diff(E_expr, S_sym).subs({C_sym: C_, S_sym: S_, L_sym: L_})),
        "dE_dL": float(sp.diff(E_expr, L_sym).subs({C_sym: C_, S_sym: S_, L_sym: L_})),
    }


derivs = symbolic_derivatives(C, S, params["loss_unit"])
rel_C = derivs["dE_dC"] * C / E_total
rel_S = derivs["dE_dS"] * S / E_total
rel_L = derivs["dE_dL"] * params["loss_unit"] / E_total
std_C = derivs["dE_dC"] * σC / σE
std_S = derivs["dE_dS"] * σS / σE
std_L = derivs["dE_dL"] * σL / σE

# ╔══════════════════════════════════════════════════════════════╗
#  Section 10  •  UI layout
# ╚══════════════════════════════════════════════════════════════╝
left, right = st.columns([1, 2])
with left:
    st.subheader(TXT["panel"]["output"])
    st.metric(TXT["metrics"]["a_total"], f"{a_total:.4f}")
    st.metric(TXT["metrics"]["succ"], f"{S:.2%}")
    st.metric(TXT["metrics"]["C"], f"{C:.1f}")
    st.metric(TXT["metrics"]["Closs"], f"{C_loss:.1f}")
    st.metric(TXT["metrics"]["E_base"], f"{E:.1f}")
    st.metric(TXT["metrics"]["E_total"], f"{E_total:.1f}")

with right:
    # Quality × Schedule matrix ────────────────────────────────
    st.subheader(TXT["charts"]["quality_schedule"]["title"])
    exp("charts.quality_schedule.expander_title")

    scenarios = [
        ("Std/On",  "Standard", "OnTime"),
        ("Std/Late","Standard", "Late"),
        ("Low/On",  "Low",      "OnTime"),
        ("Low/Late","Low",      "Late"),
    ]
    bars = []
    for label, qg, scd in scenarios:
        S_qs, _, _, _, E_tot_qs = compute_metrics(
            a1v=params["a1"], a2v=params["a2"], a3v=params["a3"],
            bv=params["b0"],
            cross_ratio_v=params["cross_ratio"],
            prep_post_ratio_v=params["prep_post_ratio"],
            loss_unit_v=params["loss_unit"],
            qualv=qg, schedv=scd,
            t1v=params["T1"], t2v=params["T2"], t3v=params["T3"],
        )
        bars.append(dict(Scenario=label, E_total=E_tot_qs, S=f"{S_qs:.1%}"))

    df_qs = pd.DataFrame(bars)
    fig_qs = px.bar(
        df_qs, x="Scenario", y="E_total", text="S",
        color_discrete_sequence=["#000000"],
        labels={"E_total": TXT["metrics"]["E_total"], "Scenario": ""}
    )
    fig_qs.update_traces(textposition="auto",
                         insidetextfont_color="white",
                         outsidetextfont_color="gray")
    fig_qs.update_layout(font=dict(size=14), bargap=0.1, margin=dict(t=30, b=40))
    st.plotly_chart(fig_qs, use_container_width=True)

    # Tornado local sensitivity  ────────────────────────────────
    st.subheader(TXT["charts"]["tornado"]["title"])
    exp("charts.tornado.expander_title")

    # ① 影響度を再計算（±20 %）
    sens_targets = {
        "a1": params["a1"],
        "a2": params["a2"],
        "a3": params["a3"],
        "b0": params["b0"],
        "CR": params["cross_ratio"],
        "PP": params["prep_post_ratio"],
        "L":  params["loss_unit"],
    }
    # 引数名マッピング
    name_map = {
        "a1": "a1v", "a2": "a2v", "a3": "a3v",
        "b0": "bv",  "CR": "cross_ratio_v",
        "PP": "prep_post_ratio_v", "L": "loss_unit_v",
    }
    tornado_rows: List[Tuple[str, float]] = []
    for key, base in sens_targets.items():
        lo = max(base * 0.8, 0)
        hi = base * 1.2 if key not in ("a1", "a2", "a3", "b0") else min(base * 1.2, 1)

        def e_total_at(**override):
            kw = dict(
                a1v=params["a1"], a2v=params["a2"], a3v=params["a3"],
                bv=params["b0"], cross_ratio_v=params["cross_ratio"],
                prep_post_ratio_v=params["prep_post_ratio"], loss_unit_v=params["loss_unit"],
                qualv=params["qual"], schedv=params["sched"],
                t1v=params["T1"], t2v=params["T2"], t3v=params["T3"],
            )
            kw.update(override)
            return compute_metrics(**kw)[4]  # E_total

        delta = max(
            abs(e_total_at(**{name_map[key]: lo}) - E_total),
            abs(e_total_at(**{name_map[key]: hi}) - E_total),
        ) / E_total  # 相対変化 (0–1)

        tornado_rows.append((key, delta))

    df_tornado = (
        pd.DataFrame(tornado_rows, columns=["Parameter", "RelChange"])
        .sort_values("RelChange", ascending=False)
    )

    fig_tornado = make_sensitivity_bar(
        df_tornado.rename(columns={"RelChange": "Tornado"}),
        value_col="Tornado",
        tick_fmt="{:.2%}",
        order=df_tornado["Parameter"].tolist(),
    )
    st.plotly_chart(fig_tornado, use_container_width=True)

    # ---------------- NEW: Sobol global sensitivity ------------
    st.subheader(TXT["charts"]["sobol"]["title"])
    exp("charts.sobol.expander_title")

    fig_sobol = make_sensitivity_bar(
        df_sobol[["Parameter", "S1"]].rename(columns={"S1": "Sobol"}),
        value_col="Sobol",
        tick_fmt="{:.2f}",
        order=df_sobol["Parameter"].tolist(),
    )
    st.plotly_chart(fig_sobol, use_container_width=True)

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