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
# Centralized core calculation
def compute_metrics(
    a1v=None, a2v=None, a3v=None, bv=None,
    cross_ratio_v=None, prep_post_ratio_v=None, loss_unit_v=None,
    qualv=None, schedv=None, t1v=None, t2v=None, t3v=None
):
    # fill in defaults from current sidebar inputs when not provided
    if a1v is None: a1v = a1
    if a2v is None: a2v = a2
    if a3v is None: a3v = a3
    if bv is None: bv = b0
    if cross_ratio_v is None: cross_ratio_v = CR
    if prep_post_ratio_v is None: prep_post_ratio_v = PP
    if loss_unit_v is None: loss_unit_v = Lunit
    if qualv is None: qualv = "Standard"
    if schedv is None: schedv = "OnTime"
    if t1v is None: t1v = T1
    if t2v is None: t2v = T2
    if t3v is None: t3v = T3
    qual_T, qual_B = (1,1)
    sched_T, sched_B = (1,1)
    if qualv == "Low":
        qual_T, qual_B = (2/3, 0.8)
    if schedv == "Late":
        sched_T, sched_B = (2/3, 0.8)
    a_tot = a1v * a2v * a3v
    b_eff = bv * qual_B * sched_B
    S_x    = 1 - (1 - a_tot) * (1 - b_eff)
    T  = (t1v + t2v + t3v) * qual_T * sched_T
    C_x    = T * (1 + cross_ratio_v + prep_post_ratio_v)
    C_loss_x = C_x + loss_unit_v * C_x * (1 - S_x)
    E_x    = C_x / S_x
    E_total_x = C_loss_x / S_x
    return S_x, C_x, C_loss_x, E_x, E_total_x

st.set_page_config(page_title="Cross-Check Simulator (B/W)", layout="wide")

# ───────────────────────────────
# Language selection (for MC histogram etc.)
lang = st.sidebar.radio("Language / 言語", ["日本語", "English"])

LANG = "EN" if lang == "English" else "JA"
TXT_ALL = {
    "EN": {
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
        }
    },
    "JA": {
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
        }
    }
}
TXT = TXT_ALL[LANG]

# ───────────────────────────────
# Sidebar inputs
st.sidebar.title(TXT["input"])
a1 = st.sidebar.slider("a1 (step 1 success rate)", 0.5, 1.0, 0.95, 0.01)
a2 = st.sidebar.slider("a2 (step 2 success rate)", 0.5, 1.0, 0.95, 0.01)
a3 = st.sidebar.slider("a3 (step 3 success rate)", 0.5, 1.0, 0.80, 0.01)
b0 = st.sidebar.slider("b0 (checker success rate)", 0.0, 1.0, 0.80, 0.01)
qual  = st.sidebar.selectbox("Qual-Grade", ["Standard", "Low"])
sched = st.sidebar.selectbox("Schedule",   ["OnTime", "Late"])
loss_unit = st.sidebar.slider("Loss unit ℓ", 0.0, 50.0, 0.0, 0.1)
T1, T2, T3 = [st.sidebar.number_input(f"T{i} (h)", 0, 200, v)
              for i, v in zip((1,2,3), (10,10,30))]
cross_ratio = st.sidebar.slider("Cross-ratio", 0.0, 0.5, 0.30, 0.01)
prep_post_ratio  = st.sidebar.slider("Prep+Post ratio", 0.0, 0.5, 0.40, 0.01)

# ───────────────────────────────
# Deterministic core
qual_T, qual_B  = (1,1) if qual=="Standard" else (2/3,0.8)
sched_T, sched_B= (1,1) if sched=="OnTime"   else (2/3,0.8)
a_total = a1 * a2 * a3
b_eff   = b0 * qual_B * sched_B
S       = 1 - (1 - a_total) * (1 - b_eff)
T = (T1 + T2 + T3) * qual_T * sched_T
C       = T * (1 + cross_ratio + prep_post_ratio)
C_loss  = C + loss_unit * C * (1 - S)
E       = C / S
E_total = C_loss / S

# ───────────────────────────────

# ───────────────────────────────
# Pre-sample Monte Carlo for σ’s
N = 100_000
rng = np.random.default_rng(0)
a1s = rng.normal(a1, 0.03, N).clip(0, 1)
a2s = rng.normal(a2, 0.03, N).clip(0, 1)
a3s = rng.triangular(a3 * 0.9, a3, a3 * 1.1, N).clip(0, 1) if a3 > 0 else np.zeros(N)
b0s = rng.uniform(0.70, 0.90, N).clip(0, 1)
t1s = rng.normal(T1, 0.5, N).clip(1)
t2s = rng.normal(T2, 0.5, N).clip(1)
t3s = rng.normal(T3, 0.5, N).clip(1)
cross_ratios = (
    rng.triangular(cross_ratio * 0.8, cross_ratio, cross_ratio * 1.2, N)
    if cross_ratio > 0 else np.zeros(N)
)
prep_post_ratios = (
    rng.triangular(prep_post_ratio * 0.8, prep_post_ratio, prep_post_ratio * 1.2, N)
    if prep_post_ratio > 0 else np.zeros(N)
)
loss_units = (
    rng.triangular(loss_unit * 0.8, loss_unit, loss_unit * 1.2, N)
    if loss_unit > 0 else np.zeros(N)
)

Evals = np.empty(N)
Svals = np.empty(N)
Cvals = np.empty(N)
for i in range(N):
    at = a1s[i] * a2s[i] * a3s[i]
    be = b0s[i] * qual_B * sched_B
    si = 1 - (1 - at) * (1 - be)
    ci = (T1 + T2 + T3) * qual_T * sched_T * (1 + cross_ratios[i] + prep_post_ratios[i])
    Evals[i] = (ci + loss_units[i] * ci * (1 - si)) / si
    Svals[i] = si
    Cvals[i] = ci

σE = Evals.std()
σL = loss_units.std()
σC = Cvals.std()
σS = Svals.std()

# ───────────────────────────────
# Symbolic derivatives
C_sym, S_sym, L_sym = sp.symbols("C S L")
E_sym = (C_sym + L_sym*(1-S_sym)) / S_sym
dE_dC = float(sp.diff(E_sym, C_sym).subs({C_sym:C, S_sym:S, L_sym:loss_unit}))
dE_dS = float(sp.diff(E_sym, S_sym).subs({C_sym:C, S_sym:S, L_sym:loss_unit}))
dE_dL = float(sp.diff(E_sym, L_sym).subs({C_sym:C, S_sym:S, L_sym:loss_unit}))

# Standardized sensitivities
std_C = dE_dC * σC / σE
std_S = dE_dS * σS / σE
std_L = dE_dL * σL / σE

# ───────────────────────────────
left, right = st.columns([1, 2])
with left:
    st.subheader(TXT["output"])
    st.metric(TXT["a_total"], f"{a_total:.4f}")
    S_x, C_x, C_loss_x, E_x, E_total_x = compute_metrics(
        a1v=a1, a2v=a2, a3v=a3, bv=b0,
        cross_ratio_v=cross_ratio, prep_post_ratio_v=prep_post_ratio,
        loss_unit_v=loss_unit, t1v=T1, t2v=T2, t3v=T3
    )
    st.metric(TXT["succ"],    f"{S_x:.2%}")
    st.metric(TXT["C"],       f"{C_x:.1f}")
    st.metric(TXT["Closs"],   f"{C_loss_x:.1f}")
    st.metric(TXT["E_base"],  f"{E_x:.1f}")
    st.metric(TXT["E_total"], f"{E_total_x:.1f}")

with right:
    ## Quality × Schedule
    st.subheader(TXT['qs_title'])
    st.markdown(TXT["qs_explain"])
    scenarios = [("Std/On","Standard","OnTime"),
                 ("Std/Late","Standard","Late"),
                 ("Low/On","Low","OnTime"),
                 ("Low/Late","Low","Late")]
    bars=[]
    for name, qg, scd in scenarios:
        S_x, C_x, C_loss_x, E_x, E_total_x = compute_metrics(
            a1v=a1, a2v=a2, a3v=a3, bv=b0,
            cross_ratio_v=cross_ratio, prep_post_ratio_v=prep_post_ratio,
            loss_unit_v=loss_unit, qualv=qg, schedv=scd, t1v=T1, t2v=T2, t3v=T3
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

    ## Tornado Sensitivity
    dark, light = "#000000", "#000000"
    st.subheader(TXT['tornado_title'])
    st.markdown(TXT["tornado_explain"])
    params = {
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "b": b0,
        "cross_ratio": cross_ratio,
        "prep_post_ratio": prep_post_ratio,
        "loss_unit": loss_unit,
        "T": T1 + T2 + T3
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
    for k, v in params.items():
        if k == "T":
            base = T1 + T2 + T3
            dT = base * 0.2
            for label, base_val, lo_val, hi_val in [("T", base, base - dT, base + dT)]:
                lo_T = lo_val * np.array([T1, T2, T3]) / base if base > 0 else np.array([0,0,0])
                hi_T = hi_val * np.array([T1, T2, T3]) / base if base > 0 else np.array([0,0,0])
                _, _, _, _, E_var_lo = compute_metrics(
                    a1v=a1, a2v=a2, a3v=a3, bv=b0,
                    cross_ratio_v=cross_ratio, prep_post_ratio_v=prep_post_ratio,
                    loss_unit_v=loss_unit, t1v=lo_T[0], t2v=lo_T[1], t3v=lo_T[2]
                )
                _, _, _, _, E_var_hi = compute_metrics(
                    a1v=a1, a2v=a2, a3v=a3, bv=b0,
                    cross_ratio_v=cross_ratio, prep_post_ratio_v=prep_post_ratio,
                    loss_unit_v=loss_unit, t1v=hi_T[0], t2v=hi_T[1], t3v=hi_T[2]
                )
                rel_delta_lo = abs(E_var_lo - E_total_x) / E_total_x * 100
                rel_delta_hi = abs(E_var_hi - E_total_x) / E_total_x * 100
                rows.append((k, max(rel_delta_lo, rel_delta_hi)))
            continue
        lo = max(v * 0.8, 0)
        hi = (min(v * 1.2, 1) if k in ("a1", "a2", "a3", "b") else v * 1.2)
        # Build kwargs for compute_metrics, filling all required arguments
        kwargs_lo = {
            "a1v": a1,
            "a2v": a2,
            "a3v": a3,
            "bv": b0,
            "cross_ratio_v": cross_ratio,
            "prep_post_ratio_v": prep_post_ratio,
            "loss_unit_v": loss_unit,
            "t1v": T1,
            "t2v": T2,
            "t3v": T3
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

    # Symbolic elasticity-based relative sensitivities for C, S
    rel_C = dE_dC * C_x / E_total_x
    rel_S = dE_dS * S_x / E_total_x
    rel_C_loss = dE_dC * C_loss_x / E_total_x
    rel_T = (rel_C + rel_C_loss) / 2  # 総工数TはCとC_totalの影響の中間と仮定
    df_rel = pd.DataFrame({
        "Parameter": [
            TXT["succ"],
            TXT["C"],
            TXT["Closs"],
            TXT["T"]
        ],
        "Relative Sensitivity": [
            abs(rel_S),
            abs(rel_C),
            abs(rel_C_loss),
            abs(rel_T)
        ]
    })
    df_rel["Parameter"] = pd.Categorical(
        df_rel["Parameter"],
        categories=[
            TXT["succ"],
            TXT["C"],
            TXT["Closs"],
            TXT["T"]
        ],
        ordered=True
    )
    fig_rel = px.bar(df_rel, x="Relative Sensitivity", y="Parameter", orientation="h",
                     text_auto=".2f", color_discrete_sequence=["#000000"],
                     labels={"Relative Sensitivity": TXT["rel_xaxis"], "Parameter": ""})
    fig_rel.update_traces(
        texttemplate='%{text}',
        text=df_rel["Relative Sensitivity"].map("{:.2f}".format),
        insidetextfont_color="white",
        outsidetextfont_color="gray"
    )
    fig_rel.update_layout(
        showlegend=False,
        yaxis=dict(
            categoryorder="array",
            categoryarray=[
                TXT["succ"],
                TXT["C"],
                TXT["Closs"],
                TXT["T"]
            ]
        ),
        xaxis_title=TXT["rel_xaxis"],
        font=dict(size=14),
        bargap=0.1
    )
    # Enforce black-and-white coloring for Promac compliance
    fig_rel.update_traces(marker_color="#000000")

# ───────────────────────────────

#
#
#
# Standardized Sensitivity (with Contribution %)
# Reuse previously computed dE_dC, dE_dS, dE_dL and values from compute_metrics
std_C = dE_dC * σC / σE
std_S = dE_dS * σS / σE
std_T = (std_C + abs(std_L)) / 2  # 総工数Tはコストと損失の中間として近似
df_std = pd.DataFrame({
    "Parameter": [
        TXT["succ"],
        TXT["C"],
        TXT["Closs"],
        TXT["T"]
    ],
    "Std Sens": [
        abs(std_S),
        abs(std_C),
        abs(std_L),
        abs(std_T)
    ]
})
df_std["Parameter"] = pd.Categorical(
    df_std["Parameter"],
    categories=[
        TXT["succ"],
        TXT["C"],
        TXT["Closs"],
        TXT["T"]
    ],
    ordered=True
)
fig_std = px.bar(
    df_std,
    x="Std Sens",
    y="Parameter",
    orientation="h",
    text=df_std["Std Sens"].map("{:.3f}".format),
    color_discrete_sequence=["#000000"],
    labels={"Std Sens": TXT["std_xaxis"], "Parameter": ""}
)
fig_std.update_traces(
    texttemplate='%{text}',
    text=df_std["Std Sens"].map("{:.3f}".format),
    textposition="auto",
    insidetextfont_color="white",
    outsidetextfont_color="gray"
)
fig_std.update_layout(
    showlegend=False,
    yaxis=dict(
        categoryorder="array",
        categoryarray=[
            TXT["succ"],
            TXT["C"],
            TXT["Closs"],
            TXT["T"]
        ]
    ),
    xaxis_title=TXT["std_xaxis"],
    font=dict(size=14),
    margin=dict(r=80),
    bargap=0.1
)
# Enforce black-and-white coloring for Promac compliance
fig_std.update_traces(marker_color="#000000")

# Display Relative and Standardized Sensitivity side by side
sens_col1, sens_col2 = st.columns(2)
with sens_col1:
    st.subheader(TXT['spider_title'])
    st.markdown(TXT["spider_explain"])
    st.plotly_chart(fig_std, use_container_width=True)
with sens_col2:
    st.subheader(TXT['rel_title'])
    st.markdown(TXT["rel_explain"])
    st.plotly_chart(fig_rel, use_container_width=True)

#
# ───────────────────────────────
# Monte Carlo Summary
st.subheader(TXT['mc_title'])
st.markdown(TXT["mc_explain"])

mc_var = st.sidebar.selectbox(TXT["mc_variable"], ["E_total", "Success S"])
data = Evals if mc_var == "E_total" else Svals
p5, p95 = np.percentile(data, [5, 95])
mean = data.mean()
med = np.median(data)
decimals = ".2f" if mc_var == "E_total" else ".4f"

st.metric("Mean", format(mean, decimals))
st.metric("Median", format(med, decimals))
st.metric("5–95% CI", f"{format(p5, decimals)} – {format(p95, decimals)}")

# Histogram label localization
LABELS = {
    "E_total": "E_total: 効率（総合）" if lang == "日本語" else "E_total: Efficiency (Total)",
    "count": "頻度 / Frequency" if lang == "日本語" else "Count / Frequency",
}

fig_h = px.histogram(
    data,
    nbins=100,
    labels={"value": LABELS["E_total"]},
    color_discrete_sequence=["black"]
)
fig_h.update_traces(
    marker_color="#000000",
    marker_line_color="#000000",
    marker_line_width=0.5
)
fig_h.update_traces(showlegend=False)
fig_h.add_vline(x=mean, line_dash="solid", line_color="#000000")
fig_h.add_vline(x=med, line_dash="solid", line_color="#000000")
fig_h.add_vline(x=p5, line_dash="dot", line_color="#000000")
fig_h.add_vline(x=p95, line_dash="dot", line_color="#000000")
fig_h.update_layout(
    yaxis_title=LABELS["count"],
    bargap=0.01,
    xaxis_tickfont_size=12,
    yaxis_tickfont_size=12
)
st.plotly_chart(fig_h, use_container_width=True)
st.caption(TXT["mc_caption"]["ja"] if lang == "日本語" else TXT["mc_caption"]["en"])