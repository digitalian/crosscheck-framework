# crosscheck_sim_promac_bw_v8.py
# Streamlit ≥1.35 | pip install streamlit plotly numpy pandas

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Cross-Check Simulator (B/W)", layout="wide")

# ───────────────────────────────
# Text resources
LANG = st.sidebar.radio("Language / 言語", ("EN", "JA"))
TXT = {
    "EN": {
        "input":               "INPUTS",
        "output":              "Headline outputs",
        "a_total":             "a_total",
        "succ":                "S",
        "C":                   "C",
        "Closs":              "C_total",
        "E_base":              "E (base)",
        "E_total":             "E_total",
        "tornado_title":       "Tornado (±20 %)",
        "tornado_hint":        "Dark = top driver(s).",
        "spider_title":        "Slope Sensitivity",
        "spider_hint":         "Shorter bar = more robust.",
        "qs_title":            "E_total & Success (2×2)",
        "qs_hint":             "Bars = E_total; labels = success rate.",
        "mc_title":            "Monte-Carlo (100 000)",
        "mc_hint":             "Variable selectable; dashed = 95 % CI, solid = mean."
    },
    "JA": {
        "input":               "入力パネル",
        "output":              "主要出力指標",
        "a_total":             "a_total",
        "succ":                "成功率 S",
        "C":                   "C（工数）",
        "Closs":              "C_total（損失込）",
        "E_base":              "E（基本）",
        "E_total":             "E_total（コスト/成功）",
        "tornado_title":       "トルネード図（±20 %）",
        "tornado_hint":        "濃灰＝最大感度バー。最大感度のパラメータがEに影響しやすい。",
        "spider_title":        "感度スロープ図",
        "spider_hint":         "バーが短いほどロバスト（揺らぎ態勢）があり。長いほどEに対する影響が大きい。",
        "qs_title":            "品質×納期 2×2",
        "qs_hint":             "棒＝E_total、ラベル＝成功率。",
        "mc_title":            "モンテカルロ（100 000回）",
        "mc_hint":             "変数選択可；点線＝95 %区間、実線＝平均。a1･a2･a3を±3%、bのコストを25-35%、bの成功率を70-90%揺らしてランダム10万回試行したグラフ。平均･中央値画理論値でずれていないこと、5–95% CIが理論値付近に集まっていれば、モデルとしての妥当性が高いと言える。"
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
Lunit = st.sidebar.slider("Loss unit ℓ", 0.0, 20.0, 0.0, 0.1)
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
    st.metric(TXT["a_total"], f"{a_total:.3f}")
    st.metric(TXT["succ"],    f"{S:.2%}")
    st.metric(TXT["C"],       f"{C:.1f}")
    st.metric(TXT["Closs"],   f"{C_loss:.1f}")
    st.metric(TXT["E_base"],       f"{E:.1f}")
    st.metric(TXT["E_total"], f"{E_total:.1f}")

# ───────────────────────────────
with right:
    
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
    slope_L = (1 - S) / S         # ∂E/∂ℓ  sensitivity
    slope_C = 1      / S         # ∂E/∂C  sensitivity
    slope_S = (C * (1 + Lunit)) / (S**2)  # ∂E/∂S  sensitivity

    df_s = pd.DataFrame([
        ("Loss unit ℓ (∂E/∂ℓ)", slope_L),
        ("Labor C (∂E/∂C)"     , slope_C),
        ("Success S (∂E/∂S)"   , slope_S),
    ], columns=["Parameter","Slope"])
    fig_s = px.bar(df_s, x="Parameter", y="Slope", text_auto=".2f",
                   color_discrete_sequence=["#777777"])
    fig_s.update_traces(textposition="outside")
    st.plotly_chart(fig_s, use_container_width=True)

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
        bars.append(dict(Scenario=name, E_total=(Cx+Lunit*Cx*(1-Sx))/Sx, S=f"{Sx:.1%}"))
    fig_q = px.bar(pd.DataFrame(bars), x="Scenario", y="E_total", text="S",
                   color_discrete_sequence=["#888888"])
    st.plotly_chart(fig_q, use_container_width=True)

# ───────────────────────────────
# Monte-Carlo Global Sensitivity
if st.sidebar.checkbox("Run Monte-Carlo (100 000)"):
    mc_var = st.sidebar.selectbox("MC variable", ["E_total","Success S"])
    st.markdown(f"**{TXT['mc_title']} – {mc_var}**")
    st.caption(TXT["mc_hint"])

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
    st.subheader("Monte-Carlo Summary Statistics (100 000)")
    st.metric("Mean",   f"{mean:.2f}")
    st.metric("Median", f"{median:.2f}")
    st.metric("5–95% CI", f"{p5:.2f} – {p95:.2f}")

    fig_h = px.histogram(data, nbins=35, labels={"value":label})
    fig_h.update_traces(marker_color="#888888")
    fig_h.add_vline(x=mean,   line_dash="solid", line_color="#222222",
                    annotation_text=f"Mean: {mean:.2f}",   annotation_position="top right")
    fig_h.add_vline(x=median, line_dash="solid", line_color="#444444",
                    annotation_text=f"Median: {median:.2f}", annotation_position="top left")
    fig_h.add_vline(x=p5,     line_dash="dot",   line_color="#555555",
                    annotation_text=f"5th pct: {p5:.2f}", annotation_position="bottom left")
    fig_h.add_vline(x=p95,    line_dash="dot",   line_color="#555555",
                    annotation_text=f"95th pct: {p95:.2f}", annotation_position="bottom right")
    st.plotly_chart(fig_h, use_container_width=True)