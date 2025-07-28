# crosscheck_sim_promac_bw_v8.py
# Streamlit ≥1.35 | pip install streamlit plotly numpy pandas

import streamlit as st
import numpy as np
import sympy as sp
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(page_title="Cross-Check Simulator (B/W)", layout="wide")

# ───────────────────────────────
# Text resources
LANG = st.sidebar.radio("Language / 言語", ("EN", "JA"))
TXT = {
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
        "tornado_hint": "Visualizes how much each parameter affects E when varied ±20%. Darker bars indicate parameters with the highest sensitivity and strongest influence on E.",
        "spider_title": "Sensitivity Breakdown: Partial Derivative vs. Elasticity",
        "spider_hint": "This chart compares two types of sensitivity: (1) Partial derivatives ∂E/∂x, which show the absolute effect of each parameter on E, and (2) Elasticity, which shows the relative (%-based) impact when a parameter changes by 1%. Partial sensitivity reflects raw influence; elasticity reflects scale-adjusted influence.",
        "qs_title": "Quality × Schedule 2×2 Matrix",
        "qs_hint": "Bar height represents E_total (cost per success), and the label shows the resulting success probability. Helps identify the optimal balance between quality (S) and cost (C).",
        "mc_title": "Monte Carlo Summary Statistics (100,000 trials)",
        "mc_hint": "Solid lines = median and mean; dashed = 95% confidence interval (5–95%). Based on 100,000 trials with a1–a3 ±3%, b’s cost 25–35%, and b’s success rate 70–90%. Close median–mean and narrow CI suggest model robustness."
    },
    "JA": {
        "input": "入力パネル",
        "output": "主要出力指標",
        "a_total": "a_total",
        "succ": "成功率 S",
        "C": "C（作業工数）",
        "Closs": "C_total（損失を含む総コスト）",
        "E_base": "効率 E（ベースライン）",
        "E_total": "E_total（成功1件あたりの総コスト）",
        "tornado_title": "トルネード感度分析（±20％）",
        "tornado_hint": "各パラメータを±20％変化させたときの効率Eへの影響を示す。バーが濃いほど感度が高く、効率に強い影響を与える要因であることを示す。",
        "spider_title": "感度分解：偏微分と相対感度の比較",
        "spider_hint": "このグラフは、効率Eに対する感度を2種類の尺度で比較する：① 偏微分（∂E/∂x）は各パラメータがEに与える絶対的な影響を示し、② Elasticity（相対感度）は各パラメータを1％変化させたときにEが何％変化するかを示す。前者は「効果量」、後者は「影響の割合」に基づく指標である。",
        "qs_title": "品質 × 納期の2×2マトリクス",
        "qs_hint": "各バーの高さは E_total（成功1件あたりのコスト）を、ラベルはそのときの成功率 S を示す。品質（成功率）と納期（工数）のバランスを比較・選択するための視覚的な支援となる。",
        "mc_title": "モンテカルロ要約統計（10万回試行）",
        "mc_hint": "選択した変数の分布を表示する：実線は中央値（中位値）と平均値、点線は95%信頼区間（5–95%範囲）を示す。a1・a2・a3を±3%、bのコストを25–35%、bの成功率を70–90%で揺らして10万回試行した結果をもとに、モデルの安定性と妥当性を可視化する。中央値と平均が近く、CIが理論値周辺に収束していれば、ロバストなモデルと言える。"
    }
}[LANG]

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

# 成功率 S を計算する関数
def compute_S(a1v=a1, a2v=a2, a3v=a3, b0v=b0, CRv=cross_ratio, Pv=prep_ratio, Lv=Lunit):
    a_tot = a1v * a2v * a3v
    b_eff = b0v * qual_B * sched_B
    return 1 - (1 - a_tot) * (1 - b_eff)

def compute_E(a1v=a1, a2v=a2, a3v=a3, b0v=b0, CRv=cross_ratio, Pv=prep_ratio, Lv=Lunit):
    """Compute E_total given parameters."""
    a_tot = a1v * a2v * a3v
    b_eff = b0v * qual_B * sched_B
    Sx    = 1 - (1 - a_tot) * (1 - b_eff)
    Ttot  = (T1 + T2 + T3) * qual_T * sched_T
    Cx    = Ttot * (1 + CRv + Pv)
    return (Cx + Lv * Cx * (1 - Sx)) / Sx

# ───────────────────────────────
left, right = st.columns([1, 2])
with left:
    st.subheader(TXT["output"])
    st.metric(TXT["a_total"], f"{a_total:.4f}")
    st.metric(TXT["succ"],    f"{S:.4f}")
    st.metric(TXT["C"],       f"{C:.2f}")
    st.metric(TXT["Closs"],   f"{C_loss:.2f}")
    st.metric(TXT["E_base"],       f"{E:.2f}")
    st.metric(TXT["E_total"], f"{E_total:.2f}")

# ───────────────────────────────
with right:
    ## Quality × Schedule
    st.markdown(f"**{TXT['qs_title']}**")
    st.caption(TXT["qs_hint"])
    scenarios = [
      ("Std/On", "Standard","OnTime"),
      ("Std/Late","Standard","Late"),
      ("Low/On","Low","OnTime"),
      ("Low/Late","Low","Late")
    ]
    bars = []
    for name, qg, scd in scenarios:
        qT, qB = (1,1) if qg=="Standard" else (2/3,0.8)
        sT, sB = (1,1) if scd=="OnTime"   else (2/3,0.8)
        Sx = 1 - (1 - a_total)*(1 - b0*qB*sB)
        Cx = (T1+T2+T3)*qT*sT*(1+cross_ratio+prep_ratio)
        bars.append(dict(Scenario=name, E_total=(Cx+Lunit*Cx*(1-Sx))/Sx, S=f"{Sx:.4f}"))
    fig_q = px.bar(
        pd.DataFrame(bars),
        x="Scenario", y="E_total", text="S",
        color_discrete_sequence=["#888888"]
    )

    # E = 90 の基準線（点線）を追加
    fig_q.add_hline(
        y=90,
        line_dash="dash",
        line_color="black",
        annotation_text="Baseline E = 90",
        annotation_position="top left",
        annotation_font_size=12,
        opacity=0.6
    )

    st.plotly_chart(fig_q, use_container_width=True)
        
    ## Tornado plot
    dark, light = "#333333", "#BBBBBB"
    st.markdown(f"**{TXT['tornado_title']}**")
    st.caption(TXT["tornado_hint"])

    params = {
        "a1": a1, "a2": a2, "a3": a3,
        "b0": b0,
        "cross_ratio": cross_ratio,
        "prep_ratio": prep_ratio,
        "Lunit": Lunit
    }
    name_map = {
        "a1":"a1v","a2":"a2v","a3":"a3v",
        "b0":"b0v","cross_ratio":"CRv",
        "prep_ratio":"Pv","Lunit":"Lv"
    }

    rows = []
    for k, v in params.items():
        lo = max(v * 0.8, 0)
        hi = min(v * 1.2, 1) if k in ("a1","a2","a3","b0") else v * 1.2
        d_lo = abs(compute_E(**{name_map[k]: lo}) - E_total)
        d_hi = abs(compute_E(**{name_map[k]: hi}) - E_total)
        rows.append((k, max(d_lo, d_hi)))

    df_t = pd.DataFrame(rows, columns=["param","delta"])\
             .sort_values("delta", ascending=False)
    maxd = df_t["delta"].max()
    df_t["color"] = np.where(np.isclose(df_t["delta"], maxd, atol=1e-9), dark, light)

    fig_t = px.bar(
        df_t,
        x="delta", y="param", orientation="h",
        color="color",
        color_discrete_map={dark:dark, light:light},
        labels={"delta":"|ΔE|","param":""}
    )
    fig_t.update_layout(showlegend=False,
                        yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig_t, use_container_width=True)

    ## Slope plot
    st.markdown(f"**{TXT['spider_title']}**")
    st.caption(TXT["spider_hint"])

    # 記号の定義（数値と区別するため _sym を付ける）
    C_sym, S_sym, L_sym = sp.symbols("C S Lunit")

    # 効率 E の式
    E_sym = (C_sym + L_sym * (1 - S_sym)) / S_sym

    # 偏微分（∂E/∂x）
    dE_dL = sp.diff(E_sym, L_sym)
    dE_dC = sp.diff(E_sym, C_sym)
    dE_dS = sp.diff(E_sym, S_sym)

    # 数値代入用辞書
    subs_dict = {C_sym: C, S_sym: S, L_sym: Lunit}

    # 効率値 E の数値化
    E_val = float(E_sym.subs(subs_dict))

    # 偏微分の数値評価
    slope_L = float(dE_dL.subs(subs_dict))
    slope_C = float(dE_dC.subs(subs_dict))
    slope_S = float(dE_dS.subs(subs_dict))

    # Elasticity（相対感度）評価
    elast_L = slope_L * (Lunit / E_val)
    elast_C = slope_C * (C / E_val)
    elast_S = slope_S * (S / E_val)

    # データフレーム化（2系列）
    df = pd.DataFrame({
        "Parameter": ["Loss unit ℓ", "Labor C", "Success S"],
        "Partial (∂E/∂x)": [slope_L, slope_C, slope_S],
        "Elasticity (∂E/∂x × x/E)": [elast_L, elast_C, elast_S],
    })

    df_melt = df.melt(id_vars="Parameter", var_name="Type", value_name="Value")

    # パラメータ名と値（横軸用）
    params = ["Loss unit ℓ", "Labor C", "Success S"]
    partials = [slope_L, slope_C, slope_S]
    elasticities = [elast_L, elast_C, elast_S]

    # サブプロット作成（上下2段・スペーサー拡大）
    fig = make_subplots(
        rows=2, cols=1,
        shared_yaxes=True,
        vertical_spacing=0.30,
        subplot_titles=["Partial (∂E/∂x)", "Elasticity (∂E/∂x × x/E)"]
    )

    # 上段：Partial（∂E/∂x）
    fig.add_trace(
        go.Bar(
            x=partials, y=params,
            name="Partial", orientation="h",
            marker_color="#888888",
            text=[f"{v:.0f}" for v in partials],
            textposition="outside",
        ),
        row=1, col=1
    )

    # 下段：Elasticity（相対感度）も灰色で統一
    fig.add_trace(
        go.Bar(
            x=elasticities, y=params,
            name="Elasticity", orientation="h",
            marker_color="#888888",
            text=[f"{v:.2f}" for v in elasticities],
            textposition="outside",
        ),
        row=2, col=1
    )

    # レイアウト調整
    fig.update_layout(
        height=600,
        showlegend=False,
        margin=dict(l=80, r=40, t=60, b=40),
    )

    # 軸ラベル設定
    fig.update_xaxes(title_text="∂E/∂x", row=1, col=1)
    fig.update_xaxes(title_text="∂E/∂x × x / E", row=2, col=1)
    fig.update_yaxes(title_text=None)

    # 表示
    st.plotly_chart(fig, use_container_width=True)

# ───────────────────────────────
# Monte-Carlo Global Sensitivity
if st.sidebar.checkbox("Run Monte-Carlo (100 000)"):
    mc_var = st.sidebar.selectbox("MC variable", ["E_total","Success S"])

    N, rng = 100_000, np.random.default_rng(0)
    a1s = rng.normal(a1, 0.03, N).clip(0,1)
    a2s = rng.normal(a2, 0.03, N).clip(0,1)
    if a3 > 0:
        a3s = rng.triangular(a3*0.9, a3, a3*1.1, N).clip(0,1)
    else:
        a3s = np.zeros(N)

    if cross_ratio > 0:
        CRs = rng.triangular(cross_ratio*0.8, cross_ratio, cross_ratio*1.2, N)
    else:
        CRs = np.zeros(N)

    if prep_ratio > 0:
        PPs = rng.triangular(prep_ratio*0.8, prep_ratio, prep_ratio*1.2, N)
    else:
        PPs = np.zeros(N)

    b0s = rng.uniform(0.70, 0.90, N).clip(0,1)
    if Lunit > 0:
        Ls = rng.triangular(Lunit*0.8, Lunit, Lunit*1.2, size=N)
    else:
        Ls = np.zeros(N)

    Evals, Svals = [], []
    for i in range(N):
        a_tot = a1s[i]*a2s[i]*a3s[i]
        b_e   = b0s[i]*qual_B*sched_B
        S_i   = 1 - (1 - a_tot)*(1 - b_e)
        C_i   = (T1+T2+T3)*qual_T*sched_T*(1+CRs[i]+PPs[i])
        Evals.append((C_i+Ls[i]*C_i*(1-S_i))/S_i)
        Svals.append(S_i)

    data  = np.array(Evals if mc_var=="E_total" else Svals)
    label = "E_total" if mc_var=="E_total" else "Success S"

    mean   = data.mean()
    median = np.median(data)
    p5, p95= np.percentile(data, [5,95])
    st.write("---")
    st.subheader(TXT["mc_title"])
    st.caption(TXT["mc_hint"])
    st.markdown(f"**{mc_var}**")
    st.metric("Mean",   f"{mean:.4f}")
    st.metric("Median", f"{median:.4f}")
    st.metric("5–95% CI", f"{p5:.4f} – {p95:.4f}")

    fig_h = px.histogram(data, nbins=35, labels={"value":label})
    fig_h.update_traces(marker_color="#888888")
    fig_h.add_vline(x=mean,   line_dash="solid", line_color="#222222",
                    annotation_text=f"Mean: {mean:.4f}",   annotation_position="top right")
    fig_h.add_vline(x=median, line_dash="solid", line_color="#444444",
                    annotation_text=f"Median: {median:.4f}", annotation_position="top left")
    fig_h.add_vline(x=p5,     line_dash="dot",   line_color="#555555",
                    annotation_text=f"5th pct: {p5:.4f}", annotation_position="bottom left")
    fig_h.add_vline(x=p95,    line_dash="dot",   line_color="#555555",
                    annotation_text=f"95th pct: {p95:.4f}", annotation_position="bottom right")
    st.plotly_chart(fig_h, use_container_width=True)