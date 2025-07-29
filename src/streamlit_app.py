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
def compute_metrics(a1v=None, a2v=None, a3v=None, b0v=None, CRv=None, Pv=None, Lv=None):
    # fill in defaults from current sidebar inputs when not provided
    if a1v is None: a1v = a1
    if a2v is None: a2v = a2
    if a3v is None: a3v = a3
    if b0v is None: b0v = b0
    if CRv is None: CRv = cross_ratio
    if Pv is None: Pv = prep_ratio
    if Lv is None: Lv = Lunit
    # reuse global qual and sched selectors
    qual_T, qual_B = (1,1) if qual=="Standard" else (2/3,0.8)
    sched_T, sched_B= (1,1) if sched=="OnTime"   else (2/3,0.8)
    a_tot = a1v * a2v * a3v
    b_eff = b0v * qual_B * sched_B
    Sx    = 1 - (1 - a_tot) * (1 - b_eff)
    Ttot  = (T1 + T2 + T3) * qual_T * sched_T
    Cx    = Ttot * (1 + CRv + Pv)
    C_lossx = Cx + Lv * Cx * (1 - Sx)
    Ex    = Cx / Sx
    E_totalx = C_lossx / Sx
    return Sx, Cx, C_lossx, Ex, E_totalx

st.set_page_config(page_title="Cross-Check Simulator (B/W)", layout="wide")

# ───────────────────────────────
# Text resources (UNCHANGED)
LANG = st.sidebar.radio("Language / 言語", ("EN", "JA"))
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
        "spider_explain": "This chart quantifies the influence of each parameter on E_total, normalized by its variability.  \nStandardized sensitivity highlights how strongly each uncertain factor contributes to the variance of cost efficiency.  \nUseful for uncertainty-based risk assessment. Includes loss-adjusted cost (C_total) and the approximated effect of total labor time (T_total).",
        "qs_title": "Quality × Schedule 2×2 Matrix",
        "qs_explain": "Each bar shows E_total under different combinations of quality and schedule, with labels showing the corresponding success rate.  \nThis helps compare cost-performance tradeoffs across operational scenarios.",
        "mc_title": "Monte Carlo Summary Statistics",
        "mc_explain": "The following parameters are assigned probabilistic distributions to capture plausible uncertainty ranges:\n- **a₁, a₂**: Normally distributed (mean = selected value, σ = 0.03), reflecting variation in basic process success rates due to human or environmental variability.\n- **a₃**: Triangular distribution (±10%) to reflect process-specific asymmetry in the final step's reliability.\n- **b₀**: Uniform between 0.70–0.90, assuming checker quality varies widely across contexts.\n- **Cross-ratio (CR)** and **Prep/Post ratio (PP)**: Triangular (±20%) around selected values to reflect managerial estimation variance.\n- **Loss unit ℓ**: Triangular (±20%) for capturing business risk variability.\n\nThese distributions are selected based on empirical heuristics: normal for stable processes (a₁, a₂), triangular for bounded uncertain estimates (a₃, CR, PP, ℓ), and uniform for quality variability (b₀).",
        "rel_title": "Relative Sensitivity (∂E/∂x × x/E)",
        "rel_explain": "This chart shows the elasticity of E_total with respect to each parameter, representing the impact from a 1% input change.  \nRelative sensitivity helps identify which parameters most affect cost efficiency in response to design or policy changes.  \nUseful for prioritizing improvement efforts. Includes loss-adjusted cost (C_total) and approximated effect of total labor time (T_total).",
        "mc_variable": "MC variable"
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
        "spider_explain": "各パラメータのばらつきを基準にE_totalへの影響度を標準化して定量化します。  \n標準化感度は、コスト効率に対する不確実性（ばらつき）の寄与を示し、リスク評価に有効です。  \n損失込みのコスト（C_total）と全体作業時間（T_total）の影響も含めています。",
        "qs_title": "品質×納期の2×2マトリクス",
        "qs_explain": "品質と納期の組み合わせごとのE_totalを棒グラフで示し、ラベルとして成功率を表示します。  \n運用シナリオごとのコストパフォーマンスの比較に役立ちます。",
        "mc_title": "モンテカルロ要約統計",
        "mc_explain": "以下のパラメータに確率的な揺らぎを与え、不確実性をモデル化しています：\n- **a₁, a₂**：平均を中心とした正規分布（σ=0.03）、人的または環境要因による変動を想定。\n- **a₃**：±10%の三角分布。最終工程に特有の非対称性を考慮。\n- **b₀**：0.70～0.90の一様分布。チェック品質の個人差を反映。\n- **クロスチェック比率（CR）・準備/事後比率（PP）**：±20%の三角分布。マネジメント判断のばらつきを想定。\n- **損失単位 ℓ**：±20%の三角分布。ビジネスリスクのばらつきを反映。\n\nこれらの分布は、経験的な判断に基づいて選定しています。",
        "rel_title": "相対感度（∂E/∂x × x/E）",
        "rel_explain": "各パラメータを1%変更した際のE_totalへの影響度（弾性）を示します。  \n相対感度は設計変更や方針変更による影響度の大きさを示し，改善の優先順位づけに有効です。  \n損失込みのコスト（C_total）と全体作業時間（T_total）の影響も含めています。",
        "mc_variable": "MC対象変数"
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
Lunit = st.sidebar.slider("Loss unit ℓ", 0.0, 50.0, 0.0, 0.1)
T1, T2, T3 = [st.sidebar.number_input(f"T{i} (h)", 0, 200, v)
              for i, v in zip((1,2,3), (10,10,30))]
cross_ratio = st.sidebar.slider("Cross-ratio", 0.0, 0.5, 0.30, 0.01)
prep_ratio  = st.sidebar.slider("Prep+Post ratio", 0.0, 0.5, 0.40, 0.01)

# ───────────────────────────────
# Deterministic core
qual_T, qual_B  = (1,1) if qual=="Standard" else (2/3,0.8)
sched_T, sched_B= (1,1) if sched=="OnTime"   else (2/3,0.8)
a_total = a1 * a2 * a3
b_eff   = b0 * qual_B * sched_B
S       = 1 - (1 - a_total) * (1 - b_eff)
T_total = (T1 + T2 + T3) * qual_T * sched_T
C       = T_total * (1 + cross_ratio + prep_ratio)
C_loss  = C + Lunit * C * (1 - S)
E       = C / S
E_total = C_loss / S

# ───────────────────────────────

# ───────────────────────────────
# Pre-sample Monte Carlo for σ’s
N, rng = 100_000, np.random.default_rng(0)
a1s = rng.normal(a1, 0.03, N).clip(0,1)
a2s = rng.normal(a2, 0.03, N).clip(0,1)
a3s = rng.triangular(a3*0.9, a3, a3*1.1, N).clip(0,1) if a3>0 else np.zeros(N)
CRs = rng.triangular(cross_ratio*0.8, cross_ratio, cross_ratio*1.2, N) if cross_ratio>0 else np.zeros(N)
PPs = rng.triangular(prep_ratio*0.8, prep_ratio, prep_ratio*1.2, N) if prep_ratio>0 else np.zeros(N)
b0s = rng.uniform(0.70, 0.90, N).clip(0,1)
Ls  = rng.triangular(Lunit*0.8, Lunit, Lunit*1.2, N) if Lunit>0 else np.zeros(N)

Evals = np.empty(N); Svals = np.empty(N); Cvals = np.empty(N)
for i in range(N):
    at = a1s[i]*a2s[i]*a3s[i]
    be = b0s[i]*qual_B*sched_B
    si = 1 - (1 - at)*(1 - be)
    ci = (T1+T2+T3)*qual_T*sched_T*(1+CRs[i]+PPs[i])
    Evals[i] = (ci + Ls[i]*ci*(1-si)) / si
    Svals[i] = si
    Cvals[i] = ci

σE, σL, σC, σS = Evals.std(), Ls.std(), Cvals.std(), Svals.std()

# ───────────────────────────────
# Symbolic derivatives
C_sym, S_sym, L_sym = sp.symbols("C S L")
E_sym = (C_sym + L_sym*(1-S_sym)) / S_sym
dE_dC = float(sp.diff(E_sym, C_sym).subs({C_sym:C, S_sym:S, L_sym:Lunit}))
dE_dS = float(sp.diff(E_sym, S_sym).subs({C_sym:C, S_sym:S, L_sym:Lunit}))
dE_dL = float(sp.diff(E_sym, L_sym).subs({C_sym:C, S_sym:S, L_sym:Lunit}))

# Standardized sensitivities
std_C = dE_dC * σC / σE
std_S = dE_dS * σS / σE
std_L = dE_dL * σL / σE

# ───────────────────────────────
left, right = st.columns([1, 2])
with left:
    st.subheader(TXT["output"])
    st.metric(TXT["a_total"], f"{a_total:.4f}")
    Sx, Cx, C_lossx, Ex, E_totalx = compute_metrics(a1, a2, a3, b0, cross_ratio, prep_ratio, Lunit)
    st.metric(TXT["succ"],    f"{Sx:.2%}")
    st.metric(TXT["C"],       f"{Cx:.1f}")
    st.metric(TXT["Closs"],   f"{C_lossx:.1f}")
    st.metric(TXT["E_base"],  f"{Ex:.1f}")
    st.metric(TXT["E_total"], f"{E_totalx:.1f}")

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
        qT,qB = (1,1) if qg=="Standard" else (2/3,0.8)
        sT,sB = (1,1) if scd=="OnTime"   else (2/3,0.8)
        # Scenario-specific metrics
        # total success probability
        a_tot_s = a1 * a2 * a3
        b_eff_s = b0 * qB * sB
        Sx = 1 - (1 - a_tot_s) * (1 - b_eff_s)
        # total labor time
        Ttot = (T1 + T2 + T3) * qT * sT
        # labor cost
        Cx = Ttot * (1 + cross_ratio + prep_ratio)
        # cost including loss
        C_lossx = Cx + Lunit * Cx * (1 - Sx)
        # cost per successful outcome
        E_totalx = C_lossx / Sx
        bars.append(dict(Scenario=name,
                         E_total=E_totalx,
                         S=f"{Sx:.1%}"))

    fig_q = px.bar(pd.DataFrame(bars), x="Scenario", y="E_total", text="S",
                   color_discrete_sequence=["#888888"])
    fig_q.update_traces(textposition="auto", insidetextfont_color="white", outsidetextfont_color="gray")
    st.plotly_chart(fig_q, use_container_width=True)

    ## Tornado Sensitivity
    dark, light = "#333333", "#BBBBBB"
    st.subheader(TXT['tornado_title'])
    st.markdown(TXT["tornado_explain"])
    params = {"a1":a1,"a2":a2,"a3":a3,"b0":b0,
              "cross_ratio":cross_ratio,
              "prep_ratio":prep_ratio,"Lunit":Lunit,
              "T_total": T1 + T2 + T3}
    name_map={"a1":"a1v","a2":"a2v","a3":"a3v",
              "b0":"b0v","cross_ratio":"CRv",
              "prep_ratio":"Pv","Lunit":"Lv",
              "T_total": "Ttot"}
    rows=[]
    for k,v in params.items():
        if k == "T_total":
            base_T = T1 + T2 + T3
            lo = max(base_T * 0.8, 0)
            hi = base_T * 1.2
            # For T_total, need to scale T1, T2, T3 proportionally
            T_scaler_lo = lo / base_T if base_T > 0 else 0
            T_scaler_hi = hi / base_T if base_T > 0 else 0
            # Scale T1, T2, T3 and recalc E_total
            _, _, _, _, E_var_lo = compute_metrics(
                a1v=a1, a2v=a2, a3v=a3, b0v=b0,
                CRv=cross_ratio, Pv=prep_ratio, Lv=Lunit
            )
            # Temporarily override T1, T2, T3 for lo
            T1_lo = T1 * T_scaler_lo
            T2_lo = T2 * T_scaler_lo
            T3_lo = T3 * T_scaler_lo
            Ttot_lo = (T1_lo + T2_lo + T3_lo) * qual_T * sched_T
            Cx_lo = Ttot_lo * (1 + cross_ratio + prep_ratio)
            Sx_lo = 1 - (1 - a1 * a2 * a3) * (1 - b0 * qual_B * sched_B)
            C_lossx_lo = Cx_lo + Lunit * Cx_lo * (1 - Sx_lo)
            E_var_lo = C_lossx_lo / Sx_lo

            T1_hi = T1 * T_scaler_hi
            T2_hi = T2 * T_scaler_hi
            T3_hi = T3 * T_scaler_hi
            Ttot_hi = (T1_hi + T2_hi + T3_hi) * qual_T * sched_T
            Cx_hi = Ttot_hi * (1 + cross_ratio + prep_ratio)
            Sx_hi = 1 - (1 - a1 * a2 * a3) * (1 - b0 * qual_B * sched_B)
            C_lossx_hi = Cx_hi + Lunit * Cx_hi * (1 - Sx_hi)
            E_var_hi = C_lossx_hi / Sx_hi

            rel_delta_lo = abs(E_var_lo - E_totalx) / E_totalx * 100  # percent change
            rel_delta_hi = abs(E_var_hi - E_totalx) / E_totalx * 100
            rows.append((k, max(rel_delta_lo, rel_delta_hi)))
            continue
        lo=max(v*0.8,0)
        hi= (min(v*1.2,1) if k in ("a1","a2","a3","b0") else v*1.2)
        _, _, _, _, E_var_lo = compute_metrics(**{name_map[k]:lo})
        _, _, _, _, E_var_hi = compute_metrics(**{name_map[k]:hi})
        rel_delta_lo = abs(E_var_lo - E_totalx) / E_totalx * 100  # percent change
        rel_delta_hi = abs(E_var_hi - E_totalx) / E_totalx * 100
        rows.append((k, max(rel_delta_lo, rel_delta_hi)))
    df_t=pd.DataFrame(rows, columns=["param","RelChange"])\
           .sort_values("RelChange", ascending=False)
    maxd=df_t["RelChange"].max()
    df_t["color"]=np.where(df_t["RelChange"]==maxd, dark, light)
    fig_t=px.bar(df_t, x="RelChange", y="param", orientation="h",
                 color="color", color_discrete_map={dark:dark,light:light},
                 labels={"RelChange":"|ΔE/E| (%)","param":""})
    fig_t.update_traces(text=df_t["RelChange"].map("{:.1f}%".format), textposition="auto", insidetextfont_color="white", outsidetextfont_color="gray")
    fig_t.update_layout(showlegend=False,
                        yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig_t, use_container_width=True)

    # Symbolic elasticity-based relative sensitivities for C, S
    rel_C = dE_dC * Cx / E_totalx
    rel_S = dE_dS * Sx / E_totalx
    rel_C_loss = dE_dC * C_lossx / E_totalx
    rel_T = (rel_C + rel_C_loss) / 2  # 総工数TはCとC_totalの影響の中間と仮定
    df_rel = pd.DataFrame({
        "Parameter": [
            "Success Rate S",
            "Labor Cost C",
            "Cost (w/ Loss)",
            "Total Labor Time T"
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
            "Success Rate S",
            "Labor Cost C",
            "Cost (w/ Loss)",
            "Total Labor Time T"
        ],
        ordered=True
    )
    fig_rel = px.bar(df_rel, x="Relative Sensitivity", y="Parameter", orientation="h",
                     text_auto=".2f", color_discrete_sequence=["#777777"])
    fig_rel.update_traces(
        texttemplate='%{text}',
        text=df_rel["Relative Sensitivity"].map("{:.2f}".format),
        insidetextfont_color="white",
        outsidetextfont_color="gray"
    )
    fig_rel.update_layout(
        showlegend=False,
        yaxis=dict(categoryorder="array", categoryarray=["Success Rate S", "Labor Cost C"]),
        xaxis_title="Relative Sensitivity (∂E/∂x × x/E)",
        font=dict(size=14)
    )

# ───────────────────────────────

#
#
# Standardized Sensitivity (with Contribution %)
# Reuse previously computed dE_dC, dE_dS, dE_dL and values from compute_metrics
std_C = dE_dC * σC / σE
std_S = dE_dS * σS / σE
std_T = (std_C + abs(std_L)) / 2  # 総工数Tはコストと損失の中間として近似
df_std = pd.DataFrame({
    "Parameter": [
        "Success Rate S",
        "Labor Cost C",
        "Cost (w/ Loss)",
        "Total Labor Time T"
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
        "Success Rate S",
        "Labor Cost C",
        "Cost (w/ Loss)",
        "Total Labor Time T"
    ],
    ordered=True
)
fig_std = px.bar(
    df_std,
    x="Std Sens",
    y="Parameter",
    orientation="h",
    text=None,  # We'll set custom bar labels below
    color_discrete_sequence=["#777777"],
    labels={"Std Sens": "Standardized Sensitivity (ΔE/σ_E)", "Parameter": ""}
)
fig_std.update_traces(
    texttemplate=None,
    insidetextfont_color="white",
    outsidetextfont_color="gray"
)
fig_std.update_layout(
    showlegend=False,
    yaxis=dict(categoryorder="array", categoryarray=["Success Rate S", "Labor Cost C"]),
    xaxis_title="Standardized Sensitivity (ΔE/σ_E)",
    font=dict(size=14),
    margin=dict(r=120)
)
# Format bar labels with 3 decimal places for standardized sensitivity chart
for trace in fig_std.data:
    if hasattr(trace, "x") and hasattr(trace, "text"):
        trace.text = [f"{v:.3f}" for v in trace.x]
        trace.textposition = "auto"

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
data = Evals if mc_var=="E_total" else Svals
p5,p95 = np.percentile(data,[5,95]); mean=data.mean(); med=np.median(data)
decimals = ".2f" if mc_var == "E_total" else ".4f"
st.metric("Mean",   format(mean, decimals))
st.metric("Median", format(med, decimals))
st.metric("5–95% CI", f"{format(p5, decimals)} – {format(p95, decimals)}")
fig_h = px.histogram(data, nbins=35, labels={"value":mc_var})
fig_h.update_traces(marker_color="#888888")
fig_h.add_vline(x=mean, line_dash="solid", line_color="#222222",
                annotation_text=f"Mean: {format(mean, decimals)}", annotation_position="top right")
fig_h.add_vline(x=med,  line_dash="solid", line_color="#444444",
                annotation_text=f"Median: {format(med, decimals)}", annotation_position="top left")
fig_h.add_vline(x=p5,   line_dash="dot",   line_color="#555555",
                annotation_text=f"5th pct: {format(p5, decimals)}", annotation_position="bottom left")
fig_h.add_vline(x=p95,  line_dash="dot",   line_color="#555555",
                annotation_text=f"95th pct: {format(p95, decimals)}", annotation_position="bottom right")
st.plotly_chart(fig_h, use_container_width=True)