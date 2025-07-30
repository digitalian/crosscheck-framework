# ──────────────────────────────────────────────────────────────
#  Cross‑Check Framework  •  Streamlit Simulation App
#  v2025‑07‑30  –  UI‑centric MC & Sobol  (all params ±α % around UI)
# ──────────────────────────────────────────────────────────────
#  (c) 2025 digitalian  –  MIT License
# ----------------------------------------------------------------
#  Changelog (since 2025‑07‑29)
#   • CHANGE: Monte Carlo – every parameter now sampled around the UI value
#   • CHANGE: Sobol – bounds dynamically follow current UI baseline (±α %)
#   • ADD   : Bold headings 🔶 for "high‑uncertainty" params (PP, ℓ, T₁–T₃)
#   • ADD   : Tool‑tips / expander text updated to reflect new sampling logic
# ----------------------------------------------------------------

from __future__ import annotations

import streamlit as st
import numpy as np, math
import sympy as sp
import pandas as pd
import plotly.express as px
from typing import Tuple, Dict, List

st.set_page_config(
    page_title="Cross-Check Simulator",
    layout="wide"
)

# ── New: SALib for Sobol ────────────────────────────────────────
try:
    from SALib.sample.sobol import sample as sobol_sample   # SALib ≥1.5
except ImportError:
    from SALib.sample import saltelli as sobol_sample       # SALib ≤1.4
from SALib.analyze import sobol

# ── New: PCG64DXSM generator (fast & parallel‑safe) ────────────
from numpy.random import PCG64DXSM, Generator

# --- Localisation dictionaries (English & Japanese) -------------
# *Monte Carlo / Sobol 説明文を UI 基準 ±α% 仕様に書き換え*
# ░░░  Language packs with super-detailed, publication-ready wording  ░░░
#  *Key structure is 100 % backward-compatible with the original code.*

TXT_EN = {
    "panel": {
        "input": "INPUT PANEL",
        "output": "KEY OUTPUT METRICS",
    },
    "metrics": {
        "a_total": "a_total",
        "succ": "Success Rate S",
        "C": "Labor Cost C",
        "Closs": "C_total (incl. loss)",
        "loss_unit": "Loss unit ℓ",
        "E_base": "Efficiency E (baseline)",
        "E_total": "E_total",
    },
    "charts": {
        # ———————————————————————————————————————————————
        "quality_schedule": {
            "title": "Quality × Schedule 2 × 2 Matrix",
            "expander_title": "📘 About the Quality × Schedule Chart",
            "expander_content": (
                "This 2 × 2 bar chart conveys the combined impact of the **quality policy** "
                "(e.g., ‘Standard’ vs ‘Low’) and the **schedule policy** "
                "(‘On-time’ vs ‘Late’) on the total cost-per-success **E_total**.  \n\n"
                "- **Bar height = E_total**  (the lower, the better).  \n"
                "- **Bar label = success probability S**  (shown as a percentage).  \n\n"
                "Reading tip  →  Compare the four cells horizontally and vertically to "
                "discern whether quality or schedule exerts a stronger economic leverage "
                "under the current parameter baseline."
            ),
        },
        # ———————————————————————————————————————————————
        "tornado": {
            "title": "Local Tornado Sensitivity (±20 %)",
            "expander_title": "📘 Interpreting the Tornado Diagram",
            "expander_content": (
                "The tornado diagram quantifies the **local (one-at-a-time) sensitivity** of "
                "E_total to each input parameter by perturbing that parameter ±20 % around "
                "its current UI value *while holding all others fixed*.  \n\n"
                "- **Bar length |ΔE/E|**  = relative change in E_total.  \n"
                "- **Ordering**  = bars are sorted from the most to the least influential, "
                "making the plot visually resemble a tornado.  \n\n"
                "Use this view to prioritise which parameter warrants the most immediate "
                "attention when performing local optimisation or design-of-experiments."
            ),
            "xaxis": "|ΔE/E|",
        },
        # ———————————————————————————————————————————————
        "sobol": {
            "title": "Global Sensitivity (Sobol S₁)",
            "expander_title": "📘 Sobol Global Sensitivity Analysis",
            "expander_content": (
                "We perform a variance-based global sensitivity analysis employing the "
                "Saltelli extension of Sobol’ sampling.  \n\n"
                "- **Sampling bounds**  = each parameter is allowed to vary within ±α % of "
                "its UI baseline (α depends on the parameter class; see code comments).  \n"
                "- **Displayed metric**  = first-order Sobol index **S₁**, which represents "
                "the fraction of total output variance attributable to that parameter alone, "
                "excluding interaction effects.  \n\n"
                "A larger S₁ indicates a stronger contribution to the uncertainty of "
                "E_total across the multidimensional parameter space."
            ),
            "xaxis": "Sobol S₁",
        },
        # ———————————————————————————————————————————————
        "relative_sensitivity": {
            "title": "Relative Elasticity",
            "xaxis": "Relative Sensitivity  (∂E/∂x · x / E)",
            "expander_title": "📘 Relative vs Standardised Sensitivity",
            "expander_content": (
                "**Relative (elasticity)** expresses how many percent E_total changes in "
                "response to a 1 % proportional change in a given parameter (i.e., a "
                "dimension-free slope).  \n\n"
                "In contrast, **Standardised sensitivity** scales the partial derivative "
                "by the parameter’s own standard deviation, illuminating which sources of "
                "uncertainty dominate the overall variability.  \n\n"
                "In practice, high elasticity indicates a *lever* for managerial control, "
                "whereas high standardised sensitivity signals a *risk* that should be "
                "mitigated (e.g., via additional data collection or process stabilisation)."
            ),
        },
        # ———————————————————————————————————————————————
        "standardized_sensitivity": {
            "title": "Standardised Sensitivity",
            "xaxis": "Standardised Sensitivity  (ΔE / σ_E)",
            "expander_title": "📘 Standardised Sensitivity (σ-normalised)",
            "expander_content": (
                "Computed as (∂E/∂x) · σₓ / σ_E, this metric places all parameters on a "
                "common variance-normalised footing. A value of 1 implies that a "
                "one-standard-deviation shock in the parameter shifts E_total by one "
                "standard deviation, ceteris paribus."
            ),
        },
        # ———————————————————————————————————————————————
        "monte_carlo": {
            "title": "Monte-Carlo Summary",
            "expander_title": "📘 Monte-Carlo Input Distributions",
            "expander_content": (
                "Each uncertain parameter is stochastically sampled around the *current UI "
                "value* to emulate real-world process variability.  \n\n"
                "• **a₁, a₂**   Normal (μ = UI, σ = 3 % μ)  \n"
                "• **a₃**       Triangular (lower = 0.9 μ, mode = μ, upper = 1.1 μ)  \n"
                "• **b₀**       Uniform [max(0, μ−0.10), min(1, μ+0.10)]  \n"
                "• **CR, PP**   Triangular [0.8 μ, μ, 1.2 μ]  \n"
                "• **ℓ**        Triangular [0.8 μ, μ, 1.2 μ]  \n"
                "• **T₁–T₃**    Normal (μ = UI, σ = 10 % μ)  \n\n"
                "_If the UI value of CR, PP, or ℓ is **zero**, the parameter is kept at "
                "zero (i.e., no stochastic variation is introduced)._  \n\n"
                "The resulting histogram overlays the mean (solid line), median (solid line), "
                "and 5–95 % credible interval (dotted lines) to provide an at-a-glance view "
                "of central tendency and dispersion."
            ),
            "variable": "MC variable",
            "mean": "Mean",
            "median": "Median",
            "ci": "5–95 % CI",
            "caption": {
                "en": "Histogram with mean (solid), median (solid), 5–95 % CI (dotted)",
                "ja": "ヒストグラム：平均/中央値＝実線、信頼区間＝点線"
            },
            "card_unit": "[E_total]",
        },
    },
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
        # ———————————————————————————————————————————————
        "quality_schedule": {
            "title": "品質 × 納期 2 × 2 マトリクス",
            "expander_title": "📘 チャートの読み方",
            "expander_content": (
                "横軸に品質（標準／低）、縦軸に納期（オンタイム／遅延）の "
                "2 × 2 組み合わせを配置し、各バーの高さで **E_total** "
                "（成功 1 件あたり総コスト）を示します。バー上のラベルは "
                "対応する成功率 **S** を百分率で表示します。  \n\n"
                "👉 4 通りのシナリオを一目で比較し、品質施策と納期施策の "
                "どちらが経済的に優位かを判断してください。"
            ),
        },
        # ———————————————————————————————————————————————
        "tornado": {
            "title": "ローカル トルネード感度 (±20 %)",
            "expander_title": "📘 トルネード図とは",
            "expander_content": (
                "各入力パラメータを **UI 値から ±20 %** だけ単独で変動させ、"
                "そのときのコスト効率 **E_total** の相対変化 |ΔE/E| をバーの長さ "
                "として描画します。  \n\n"
                "バーが長い＝そのパラメータが **局所的** に最も強い影響を持つ "
                "ことを意味し、改善・調整の優先度を示唆します。"
            ),
            "xaxis": "|ΔE/E|",
        },
        # ———————————————————————————————————————————————
        "sobol": {
            "title": "グローバル感度 (Sobol S₁)",
            "expander_title": "📘 Sobol 感度解析の概要",
            "expander_content": (
                "Saltelli 拡張を用いた Sobol 法で、各パラメータを UI 基準値 ±α % "
                "の範囲で同時にサンプリングし、**E_total** の分散に対する一次寄与 "
                "（Sobol 指数 **S₁**）を算出します。  \n\n"
                "S₁ が大きいほど、そのパラメータ単独で結果の不確実性を左右している "
                "度合いが高いと解釈できます。"
            ),
            "xaxis": "Sobol S₁",
        },
        # ———————————————————————————————————————————————
        "relative_sensitivity": {
            "title": "相対弾性値",
            "xaxis": "相対感度 (∂E/∂x·x/E)",
            "expander_title": "📘 指標の意味と活用",
            "expander_content": (
                "相対弾性値（Elasticity）はパラメータを 1 % 変化させた際に "
                "**E_total** が何パーセント変動するかを示す次元レス量です。  \n\n"
                "値が大きいほど “てこの原理” が効きやすく、コスト効率改善の "
                "レバーとして有効であることを示唆します。"
            ),
        },
        # ———————————————————————————————————————————————
        "standardized_sensitivity": {
            "title": "標準化感度",
            "xaxis": "標準化感度 (ΔE/σ_E)",
            "expander_title": "📘 標準化感度とは",
            "expander_content": (
                "パラメータの標準偏差 σₓ で正規化した感度 "
                "(∂E/∂x)·σₓ/σ_E を示します。  \n\n"
                "大きい値は『そのパラメータの不確実性が **E_total** の変動に "
                "大きく寄与している』ことを示し、リスク管理やデータ収集の "
                "優先度付けに役立ちます。"
            ),
        },
        # ———————————————————————————————————————————————
        "monte_carlo": {
            "title": "モンテカルロ要約",
            "expander_title": "📘 入力分布の設定 (UI 基準)",
            "expander_content": (
                "各パラメータは **現在の UI 値** を中心に以下の分布でサンプリング "
                "されます：  \n\n"
                "• **a₁, a₂**  正規分布 (μ = UI, σ = 3 % μ)  \n"
                "• **a₃**      三角分布 (下限 = 0.9 μ, モード = μ, 上限 = 1.1 μ)  \n"
                "• **b₀**      一様分布 [max(0, μ−0.10), min(1, μ+0.10)]  \n"
                "• **CR, PP**  三角分布 [0.8 μ, μ, 1.2 μ]  \n"
                "• **ℓ**       三角分布 [0.8 μ, μ, 1.2 μ]  \n"
                "• **T₁–T₃**   正規分布 (μ = UI, σ = 10 % μ)  \n\n"
                "※ **CR・PP・ℓ の UI 値が 0** の場合、そのパラメータは 0 に固定され "
                "変動を与えません。  \n\n"
                "ヒストグラムには平均（実線）、中央値（実線）、信頼区間 5–95 % "
                "（点線）が重ね描きされ、中心傾向とばらつきが一目で把握できます。"
            ),
            "variable": "MC対象変数",
            "mean": "平均",
            "median": "中央値",
            "ci": "5–95 % CI",
            "caption": {
                "en": "Histogram with mean (solid), median (solid), 5–95 % CI (dotted)",
                "ja": "ヒストグラム：平均/中央値＝実線、信頼区間＝点線"
            },
            "card_unit": "[E_total]",
        },
    },
}
# ░░░░░░░░░░░░░ ねこ語 UI パック ░░░░░░░░░░░░░
# ＊英語(JA)と同じキー構造なので drop-in 置換できるにゃ＊

TXT_CAT = {
    "panel": {
        "input": "にゅうりょく ぱねる にゃ",
        "output": "たいせつ けっか にゃ",
    },

    # ────────────────────────
    "metrics": {
        "a_total": "a_total にゃ",
        "succ": "せいこうりつ S にゃ",
        "C": "おしごとコスト C にゃ",
        "Closs": "C_total (そんしつこみ) にゃ",
        "loss_unit": "そんしつたんか ℓ にゃ",
        "E_base": "こうりつ E (べーす) にゃ",
        "E_total": "E_total にゃ",
    },

    # ────────────────────────
    "charts": {
        # 1) 品質×納期
        "quality_schedule": {
            "title": "ひんしつ × のうき 2×2 にゃ",
            "expander_title": "📘 これなあに？ にゃ",
            "expander_content": (
                "４つのバーで **E_total** のたかさをくらべるにゃ。"
                "バーのうえの数字は **S** (せいこう％) にゃ。\n\n"
                "ねこポイント：よこ列・たて列で『どっちがトク？』を見つけるにゃ〜🐾"
            ),
        },

        # 2) トルネード
        "tornado": {
            "title": "とるねーど がんど (±20%) にゃ",
            "expander_title": "📘 ぐるぐる棒のひみつ にゃ",
            "expander_content": (
                "パラメータを１こずつ ±20% うごかして "
                "**|ΔE/E|** (E_total のへんか) を棒のながさで見せるにゃ。\n\n"
                "なが〜い棒 → 『ここ なおすと いちばん きく！』 にゃ🐱"
            ),
            "xaxis": "|ΔE/E| にゃ",
        },

        # 3) ソーボル
        "sobol": {
            "title": "そーぼる S₁ にゃ",
            "expander_title": "📘 そーぼる？ おいしい？ にゃ",
            "expander_content": (
                "ぜんぶのパラメータを いっせいに ユサユサして "
                "ぶれのわりあい **S₁** をはかるにゃ。\n\n"
                "S₁ が 1 にちかい → その子だけで 大あばれ にゃ！"
            ),
            "xaxis": "S₁ にゃ",
        },

        # 4) 相対弾性度
        "relative_sensitivity": {
            "title": "そうたい びよ〜ん にゃ",
            "xaxis": "Elasticity (=∂E/∂x·x/E) にゃ",
            "expander_title": "📘 びよ〜ん とは？ にゃ",
            "expander_content": (
                "1% うごかすと **E_total** が 何% うごくかを見るにゃ。\n"
                "大きい値 → 『せっけい がんばる と いいにゃ！』"
            ),
        },

        # 5) 標準化感度
        "standardized_sensitivity": {
            "title": "ひょうじゅんか かんど にゃ",
            "xaxis": "StdSens (=∂E/∂x·σₓ/σ_E) にゃ",
            "expander_title": "📘 リスクに注意にゃ",
            "expander_content": (
                "パラメータの ふらつき (σ) をかけて\n"
                "**E_total** が どれだけ ゆれるかをチェックにゃ。\n"
                "大きい値 → 『運用ちゅう リスク注意！』"
            ),
        },

        # 6) モンテカルロ
        "monte_carlo": {
            "title": "もんて かるろ にゃ〜",
            "expander_title": "📘 サンプリングのおやつ にゃ",
            "expander_content": (
                "ぜんぶ UI の今の値を まんなかに ふらふらサンプルにゃ。\n\n"
                "・a₁,a₂ → 正規(±3%) にゃ\n"
                "・a₃ → 三角(0.9〜1.1) にゃ\n"
                "・b₀ → 一様(±0.10) にゃ\n"
                "・CR,PP,ℓ → 三角(0.8〜1.2) にゃ (ぜろなら固定にゃ)\n"
                "・T₁–T₃ → 正規(±10%) にゃ\n\n"
                "ヒストグラムに平均・中央値(実線)と 5–95% (点線) をペタッとにゃ。"
            ),
            "variable": "みる子 にゃ",
            "mean": "へいきん にゃ",
            "median": "ちゅうおう にゃ",
            "ci": "5–95% にゃ",
            "caption": {
                "en": "Mean & Median = solid, CI = dotted にゃ",
                "ja": "平均/中央値=実線, 信頼区間=点線 にゃ"
            },
            "card_unit": "[E_total] にゃ",
        },
    },
}

# ------ register all packs (order: EN is default) ------
TXT_ALL = {"EN": TXT_EN, "JA": TXT_JA, "CAT": TXT_CAT}   # ← ここで初めてまとめる

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
    Show markdown from nested TXT dict using dot‑path key.
    Example: exp('charts.tornado.expander_title')
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
#  Section 2  •  Deterministic model
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
#  Section 4  •  Config & localisation
# ╚══════════════════════════════════════════════════════════════╝
st.set_page_config(page_title="Cross‑Check Simulator", layout="wide")
if "lang" not in st.session_state:
    st.session_state.lang = "English"
lang_code = st.sidebar.radio(
    "Language / 言語",                  # 表示ラベル
    ["EN", "JA", "CAT"],               # 内部キー
    index=0,                           # デフォは English
    format_func=lambda k: {"EN":"English",
                           "JA":"日本語",
                           "CAT":"😸 にゃー"}[k],
    horizontal=True,
)
TXT = TXT_ALL[lang_code]               # ここだけで全 UI 切替

# ╔══════════════════════════════════════════════════════════════╗
#  Section 5  •  Sidebar inputs  (rooted in UI)
# ╚══════════════════════════════════════════════════════════════╝
def get_sidebar_params() -> Dict[str, float]:
    st.sidebar.title(TXT["panel"]["input"])

    # ---- 基本パラメータ（実績あり） --------------------------
    st.sidebar.subheader("✅ Reliable (Gerrit / COCOMO)")

    a1 = st.sidebar.slider("a1 (step 1)", 0.5, 1.0,
                           get_state("a1", 0.95), 0.01)
    a2 = st.sidebar.slider("a2 (step 2)", 0.5, 1.0,
                           get_state("a2", 0.95), 0.01)
    a3 = st.sidebar.slider("a3 (step 3)", 0.5, 1.0,
                           get_state("a3", 0.80), 0.01)

    # ---- Quality & Schedule (離散) ----------------------------
    col_q, col_s = st.sidebar.columns(2)
    with col_q:
        qual = st.selectbox("Quality",
                            ["Standard", "Low"],
                            index=0 if get_state("qual", "Standard") == "Standard" else 1,
                            key="qual")
    with col_s:
        sched = st.selectbox("Schedule",
                             ["OnTime", "Late"],
                             index=0 if get_state("sched", "OnTime") == "OnTime" else 1,
                             key="sched")

    b0 = st.sidebar.slider("b₀ (checker skill)", 0.0, 1.0,
                           get_state("b0", 0.80), 0.01)
    cross_ratio = st.sidebar.slider("CR (cross‑ratio)",
                                    0.0, 0.5,
                                    get_state("cross_ratio", 0.30), 0.01)

    # ---- 不確実性が高いパラメータ -----------------------------
    st.sidebar.subheader("🔶 High‑uncertainty (Adjust & Watch)")

    # Bold heading for PP
    st.sidebar.markdown("**PP (Prep/Post ratio)**")
    prep_post_ratio = st.sidebar.slider("", 0.0, 0.5,
                                        get_state("prep_post_ratio", 0.40),
                                        0.01, label_visibility="collapsed")

    st.sidebar.markdown("**ℓ (Loss unit)**")
    loss_unit = st.sidebar.slider("", 0.0, 50.0,
                                  get_state("loss_unit", 0.0),
                                  0.5, label_visibility="collapsed")

    st.sidebar.markdown("**T₁–T₃ (Task hours)**")
    col_t1, col_t2, col_t3 = st.sidebar.columns(3)
    with col_t1:
        T1 = st.number_input("T₁ [h]", 0, 200,
                             get_state("T1", 10), key="T1")
    with col_t2:
        T2 = st.number_input("T₂ [h]", 0, 200,
                             get_state("T2", 10), key="T2")
    with col_t3:
        T3 = st.number_input("T₃ [h]", 0, 200,
                             get_state("T3", 30), key="T3")

    # ---- Monte Carlo settings ---------------------------------
    st.sidebar.markdown("---")
    st.sidebar.title("Monte‑Carlo")
    col_n, col_var = st.sidebar.columns(2)
    with col_n:
        sample_n = st.number_input("Samples",
                                   1_000, 1_000_000,
                                   get_state("sample_n", 100_000),
                                   step=10_000)
    with col_var:
        mc_var = st.selectbox("MC variable",
                              ["E_total", "Success S"], 0,
                              key="mc_var")

    return dict(
        a1=a1, a2=a2, a3=a3, b0=b0,
        cross_ratio=cross_ratio, prep_post_ratio=prep_post_ratio,
        loss_unit=loss_unit,
        T1=T1, T2=T2, T3=T3,
        qual=qual, sched=sched,
        sample_n=sample_n, mc_var=mc_var,
    )

params = get_sidebar_params()

# ╔══════════════════════════════════════════════════════════════╗
#  Section 6  •  Monte‑Carlo simulation  (all ±α % around UI)
# ╚══════════════════════════════════════════════════════════════╝
@st.cache_data(show_spinner=False, ttl=900)
def run_mc(p: Dict[str, float], N: int) -> Tuple[np.ndarray, ...]:
    """Vectorised Monte‑Carlo; returns Evals, Svals, Cvals, L_samples."""
    rng = Generator(PCG64DXSM(seed=0))

    # Success rates
    a1s = rng.normal(p["a1"], max(0.01, p["a1"] * 0.03), N).clip(0, 1)
    a2s = rng.normal(p["a2"], max(0.01, p["a2"] * 0.03), N).clip(0, 1)
    a3s = rng.triangular(p["a3"] * 0.9, p["a3"], p["a3"] * 1.1, N).clip(0, 1)
    b0s = rng.uniform(max(0, p["b0"] - 0.10), min(1, p["b0"] + 0.10), N)

    # Task times (±10 %)
    t1s = rng.normal(p["T1"], p["T1"] * 0.10, N).clip(min=1)
    t2s = rng.normal(p["T2"], p["T2"] * 0.10, N).clip(min=1)
    t3s = rng.normal(p["T3"], p["T3"] * 0.10, N).clip(min=1)

    # Cost ratios & loss
    CRs = (
        rng.triangular(p["cross_ratio"] * 0.8, p["cross_ratio"],
                    p["cross_ratio"] * 1.2, N)
        if p["cross_ratio"] > 0
        else np.zeros(N)
    )
    PPs = (
        rng.triangular(p["prep_post_ratio"] * 0.8, p["prep_post_ratio"],
                    p["prep_post_ratio"] * 1.2, N)
        if p["prep_post_ratio"] > 0
        else np.zeros(N)
    )
    Ls = (
        rng.triangular(p["loss_unit"] * 0.8, p["loss_unit"],
                    p["loss_unit"] * 1.2, N)
        if p["loss_unit"] > 0
        else np.zeros(N)
    )

    # Multipliers
    qual_T, qual_B = (1, 1) if p["qual"] == "Standard" else (2 / 3, 0.8)
    sched_T, sched_B = (1, 1) if p["sched"] == "OnTime" else (2 / 3, 0.8)

    a_tot = a1s * a2s * a3s
    b_eff = b0s * qual_B * sched_B
    Svals = 1 - (1 - a_tot) * (1 - b_eff)

    Tvals = (t1s + t2s + t3s) * qual_T * sched_T
    Cvals = Tvals * (1 + CRs + PPs)
    Evals = (Cvals + Ls * Cvals * (1 - Svals)) / Svals
    return Evals, Svals, Cvals, Ls

Evals, Svals, Cvals, L_samples = run_mc(params, int(params["sample_n"]))
σE, σC, σS, σL = Evals.std(), Cvals.std(), Svals.std(), (Cvals * L_samples).std()

# ╔══════════════════════════════════════════════════════════════╗
#  Section 7  •  Sobol global sensitivity (UI‑relative bounds)
# ╚══════════════════════════════════════════════════════════════╝
@st.cache_data(show_spinner=False, ttl=900)
def run_sobol(p: Dict[str, float], N: int = 10_000) -> pd.DataFrame:
    """
    Sobol analysis with bounds defined as (UI value ± α %)
    α = 3 % for a₁,a₂, 10 % a₃, 10 % T, 20 % CR/PP/ℓ, 0.10 for b₀.
    """
    # Bounds helper
    def clip(lo, hi, low=0.0, high=1e9, eps=1e-6):
        lo_c = max(low, lo)
        hi_c = min(high, hi)
        if hi_c <= lo_c:          # 幅ゼロを回避
            hi_c = lo_c + eps
        return [lo_c, hi_c]


    problem = {
        "num_vars": 10,  # a1,a2,a3,b0,CR,PP,L,T1,T2,T3
        "names": ["a1","a2","a3","b0","CR","PP","L",
                  "T1","T2","T3"],
        "bounds": [
            clip(p["a1"]*0.97, p["a1"]*1.03, 0, 1),
            clip(p["a2"]*0.97, p["a2"]*1.03, 0, 1),
            clip(p["a3"]*0.90, p["a3"]*1.10, 0, 1),
            clip(p["b0"]-0.10, p["b0"]+0.10, 0, 1),
            clip(p["cross_ratio"]*0.8, p["cross_ratio"]*1.2, 0, 0.5),
            clip(p["prep_post_ratio"]*0.8, p["prep_post_ratio"]*1.2, 0, 0.5),
            clip(p["loss_unit"]*0.8, p["loss_unit"]*1.2, 0, 50),
            clip(p["T1"]*0.9, p["T1"]*1.1, 1, 1e3),
            clip(p["T2"]*0.9, p["T2"]*1.1, 1, 1e3),
            clip(p["T3"]*0.9, p["T3"]*1.1, 1, 1e3),
        ]
    }
    # -- If you need qual_F / sched_F, uncomment lines below ----
    # problem["names"] += ["qual_F","sched_F"]
    # problem["bounds"] += [[0.8,1.0],[0.8,1.0]]
    # problem["num_vars"] = len(problem["names"])
    # -----------------------------------------------------------

    # Saltelli requires N = 2^k
    np.random.seed(0)
    N_pow2 = 2 ** math.ceil(math.log2(N))
    if N_pow2 != N:
        st.info(f"Sobol samples adjusted to {N_pow2} (nearest power‑of‑2).")
    X = sobol_sample(problem, N_pow2, calc_second_order=False)

    # Unpack X
    a_tot = X[:,0] * X[:,1] * X[:,2]
    b_eff = X[:,3] * (1 if params["qual"]=="Standard" else 0.8) \
                     * (1 if params["sched"]=="OnTime" else 0.8)
    Sarr = 1 - (1 - a_tot) * (1 - b_eff)

    Tarr = (X[:,7] + X[:,8] + X[:,9]) * \
           (1 if params["qual"]=="Standard" else 2/3) * \
           (1 if params["sched"]=="OnTime" else 2/3)
    Cbase = Tarr * (1 + X[:,4] + X[:,5])
    E_tot = (Cbase + X[:,6] * Cbase * (1 - Sarr)) / Sarr

    Si = sobol.analyze(problem, E_tot,
                       calc_second_order=False, print_to_console=False)
    df = pd.DataFrame({"Parameter": problem["names"],
                       "S1": Si["S1"], "ST": Si["ST"]})
    return df.sort_values("S1", ascending=False)

df_sobol = run_sobol(params)

# ╔══════════════════════════════════════════════════════════════╗
#  Section 8  •  Deterministic baseline
# ╚══════════════════════════════════════════════════════════════╝
qual_T, qual_B = (1,1) if params["qual"]=="Standard" else (2/3,0.8)
sched_T, sched_B = (1,1) if params["sched"]=="OnTime" else (2/3,0.8)

a_total = params["a1"]*params["a2"]*params["a3"]
b_eff   = params["b0"]*qual_B*sched_B
S = 1 - (1 - a_total)*(1 - b_eff)
T = (params["T1"]+params["T2"]+params["T3"])*qual_T*sched_T
C = T * (1 + params["cross_ratio"] + params["prep_post_ratio"])
C_loss = C + params["loss_unit"]*C*(1 - S)
E = C / S
E_total = C_loss / S

# ╔══════════════════════════════════════════════════════════════╗
#  Section 9  •  Local elasticities (symbolic)
# ╚══════════════════════════════════════════════════════════════╝
def symbolic_derivatives(C_:float,S_:float,L_:float)->Dict[str,float]:
    C_sym,S_sym,L_sym=sp.symbols("C S L")
    E_expr=(C_sym+C_sym*L_sym*(1-S_sym))/S_sym
    return {
        "dE_dC":float(sp.diff(E_expr,C_sym).subs({C_sym:C_,S_sym:S_,L_sym:L_})),
        "dE_dS":float(sp.diff(E_expr,S_sym).subs({C_sym:C_,S_sym:S_,L_sym:L_})),
        "dE_dL":float(sp.diff(E_expr,L_sym).subs({C_sym:C_,S_sym:S_,L_sym:L_})),
    }
derivs=symbolic_derivatives(C,S,params["loss_unit"])
rel_C,rel_S,rel_L = (derivs["dE_dC"]*C/E_total,
                     derivs["dE_dS"]*S/E_total,
                     derivs["dE_dL"]*params["loss_unit"]/E_total)
std_C,std_S,std_L = (derivs["dE_dC"]*σC/σE,
                     derivs["dE_dS"]*σS/σE,
                     derivs["dE_dL"]*σL/σE)

# ╔══════════════════════════════════════════════════════════════╗
#  Section 10 •  Layout & plots
# ╚══════════════════════════════════════════════════════════════╝
left,right = st.columns([1,2])
with left:
    st.subheader(TXT["panel"]["output"])
    st.metric(TXT["metrics"]["a_total"], f"{a_total:.4f}")
    st.metric(TXT["metrics"]["succ"],   f"{S:.2%}")
    st.metric(TXT["metrics"]["C"],      f"{C:.1f}")
    st.metric(TXT["metrics"]["Closs"],  f"{C_loss:.1f}")
    st.metric(TXT["metrics"]["E_base"], f"{E:.1f}")
    st.metric(TXT["metrics"]["E_total"],f"{E_total:.1f}")

with right:
    # --- Quality × Schedule bar -------------------------------
    st.subheader(TXT["charts"]["quality_schedule"]["title"])
    exp("charts.quality_schedule.expander_title")

    scenarios=[("Std/On","Standard","OnTime"),
               ("Std/Late","Standard","Late"),
               ("Low/On","Low","OnTime"),
               ("Low/Late","Low","Late")]
    bars=[]
    for lbl,qg,scd in scenarios:
        S_qs,_,_,_,E_qs=compute_metrics(
            a1v=params["a1"],a2v=params["a2"],a3v=params["a3"],
            bv=params["b0"],cross_ratio_v=params["cross_ratio"],
            prep_post_ratio_v=params["prep_post_ratio"],
            loss_unit_v=params["loss_unit"],
            qualv=qg,schedv=scd,
            t1v=params["T1"],t2v=params["T2"],t3v=params["T3"])
        bars.append(dict(Scenario=lbl,E_total=E_qs,S=f"{S_qs:.1%}"))
    fig_qs=px.bar(pd.DataFrame(bars),x="Scenario",y="E_total",text="S",
                  color_discrete_sequence=["#000000"],
                  labels={"E_total":TXT["metrics"]["E_total"],"Scenario":""})
    fig_qs.update_traces(textposition="auto",
                         insidetextfont_color="white",
                         outsidetextfont_color="gray")
    fig_qs.update_layout(font=dict(size=14),bargap=0.1,
                         margin=dict(t=30,b=40))
    st.plotly_chart(fig_qs,use_container_width=True)

    # --- Tornado ----------------------------------------------
    st.subheader(TXT["charts"]["tornado"]["title"])
    exp("charts.tornado.expander_title")
    sens_targets={"a1":params["a1"],"a2":params["a2"],"a3":params["a3"],
                  "b0":params["b0"],"CR":params["cross_ratio"],
                  "PP":params["prep_post_ratio"],"L":params["loss_unit"]}
    name_map={"a1":"a1v","a2":"a2v","a3":"a3v",
              "b0":"bv","CR":"cross_ratio_v",
              "PP":"prep_post_ratio_v","L":"loss_unit_v"}
    tornado=[]
    for k,base in sens_targets.items():
        lo=max(base*0.8,0)
        hi=min(base*1.2,1) if k in ("a1","a2","a3","b0") else base*1.2
        def e_tot(**ov):
            kw=dict(a1v=params["a1"],a2v=params["a2"],a3v=params["a3"],
                    bv=params["b0"],cross_ratio_v=params["cross_ratio"],
                    prep_post_ratio_v=params["prep_post_ratio"],
                    loss_unit_v=params["loss_unit"],
                    qualv=params["qual"],schedv=params["sched"],
                    t1v=params["T1"],t2v=params["T2"],t3v=params["T3"])
            kw.update(ov); return compute_metrics(**kw)[4]
        delta=max(abs(e_tot(**{name_map[k]:lo})-E_total),
                  abs(e_tot(**{name_map[k]:hi})-E_total))/E_total
        tornado.append((k,delta))
    df_tornado=pd.DataFrame(tornado,columns=["Parameter","RelChange"])\
                 .sort_values("RelChange",ascending=False)
    fig_tornado=make_sensitivity_bar(
        df_tornado.rename(columns={"RelChange":"Tornado"}),
        value_col="Tornado",tick_fmt="{:.2%}",
        order=df_tornado["Parameter"].tolist())
    st.plotly_chart(fig_tornado,use_container_width=True)

    # --- Sobol -------------------------------------------------
    st.subheader(TXT["charts"]["sobol"]["title"])
    exp("charts.sobol.expander_title")
    fig_sobol=make_sensitivity_bar(
        df_sobol[["Parameter","S1"]].rename(columns={"S1":"Sobol"}),
        value_col="Sobol",tick_fmt="{:.2f}",
        order=df_sobol["Parameter"].tolist())
    st.plotly_chart(fig_sobol,use_container_width=True)

    # --- Relative / Standardised ------------------------------
    sens_df=pd.DataFrame(dict(
        Parameter=[TXT["metrics"]["loss_unit"],
                   TXT["metrics"]["C"],
                   TXT["metrics"]["succ"]],
        Relative=[abs(rel_L),abs(rel_C),abs(rel_S)],
        Standardised=[abs(std_L),abs(std_C),abs(std_S)]))
    order=sens_df["Parameter"].tolist()
    fig_rel=make_sensitivity_bar(
        sens_df[["Parameter","Relative"]].rename(columns={"Relative":"rel"}),
        value_col="rel",order=order,tick_fmt="{:.2f}")
    fig_std=make_sensitivity_bar(
        sens_df[["Parameter","Standardised"]].rename(columns={"Standardised":"std"}),
        value_col="std",order=order,tick_fmt="{:.3f}")

# -- Display side‑by‑side
col_rel,col_std=st.columns(2)
with col_rel:
    col_rel.subheader(TXT["charts"]["relative_sensitivity"]["title"])
    exp("charts.relative_sensitivity.expander_title")
    col_rel.plotly_chart(fig_rel,use_container_width=True)
with col_std:
    col_std.subheader(TXT["charts"]["standardized_sensitivity"]["title"])
    exp("charts.standardized_sensitivity.expander_title")
    col_std.plotly_chart(fig_std,use_container_width=True)

# ═════ Monte Carlo histogram ═══════════════════════════════════
st.subheader(TXT["charts"]["monte_carlo"]["title"])
exp("charts.monte_carlo.expander_title")
data = Evals if params["mc_var"]=="E_total" else Svals
ci_low,ci_high=np.percentile(data,[5,95])
mean,median=data.mean(),np.median(data)
dec=".2f" if params["mc_var"]=="E_total" else ".4f"
c1,c2,c3=st.columns(3)
c1.metric(TXT["charts"]["monte_carlo"]["mean"]+TXT["charts"]["monte_carlo"]["card_unit"],
          f"{mean:{dec}}")
c2.metric(TXT["charts"]["monte_carlo"]["median"]+TXT["charts"]["monte_carlo"]["card_unit"],
          f"{median:{dec}}")
c3.metric(TXT["charts"]["monte_carlo"]["ci"]+TXT["charts"]["monte_carlo"]["card_unit"],
          f"{ci_low:{dec}} – {ci_high:{dec}}")

# ✅ 新しい（lang → lang_code）
label_E   = "E_total" if lang_code=="EN" else ("E_total: 総合効率" if lang_code=="JA" else "E_total にゃ")
label_cnt = "Count"   if lang_code=="EN" else ("頻度"                 if lang_code=="JA" else "かず にゃ")
fig_hist=px.histogram(data,nbins=100,
                      labels={"value":label_E},
                      color_discrete_sequence=["#000000"])
fig_hist.update_traces(marker_line_width=0.5)
for x,style in [(mean,"solid"),(median,"solid"),
                (ci_low,"dot"),(ci_high,"dot")]:
    fig_hist.add_vline(x=x,line_dash=style,line_color="#000000")
fig_hist.update_layout(yaxis_title=label_cnt,bargap=0.01,
                       margin=dict(t=70),showlegend=False)
st.plotly_chart(fig_hist,use_container_width=True)
legend    = "Mean / Median: solid  5–95 % CI: dotted" if lang_code=="EN" \
          else ("平均/中央値：実線  信頼区間5–95 %：点線"            if lang_code=="JA"
          else  "平均/中央値=線にゃ  CI=点線にゃ")
st.caption(legend)
