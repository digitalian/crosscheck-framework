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
        "C_loss":              "C_total",
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
        "C_loss":              "C_total（損失込）",
        "E_base":              "E（基本）",
        "E_total":             "E_total（コスト/成功）",
        "tornado_title":       "トルネード図（±20 %）",
        "tornado_hint":        "濃灰＝最大感度バー。",
        "spider_title":        "感度スロープ図",
        "spider_hint":         "バーが短いほどロバスト。",
        "qs_title":            "品質×納期 2×2",
        "qs_hint":             "棒＝E_total、ラベル＝成功率。",
        "mc_title":            "モンテカルロ（100 000回）",
        "mc_hint":             "変数選択可；点線＝95 %区間、実線＝平均。"
    }
}[LANG]

# ───────────────────────────────
# Sidebar inputs
st.sidebar.title(TXT["input"])
a1           = st.sidebar.slider("a1 (step 1)", 0.5, 1.0, 0.95, 0.01)
a2           = st.sidebar.slider("a2 (step 2)", 0.5, 1.0, 0.95, 0.01)
a3           = st.sidebar.slider("a3 (step 3)", 0.5, 1.0, 0.80, 0.01)
b0           = st.sidebar.slider("b0 (checker)", 0.5, 1.0, 0.80, 0.01)
qual         = st.sidebar.selectbox("Qual-Grade", ["Standard", "Low"])
sched        = st.sidebar.selectbox("Schedule",   ["OnTime",   "Late"])
Lunit        = st.sidebar.slider("Loss unit ℓ", 0.0, 20.0, 0.0, 0.1)
T1, T2, T3   = [st.sidebar.number_input(f"T{i} (h)", 1, 200, v) 
                for i, v in zip((1,2,3), (10,10,30))]
cross_ratio  = st.sidebar.slider("Cross-check overhead", 0.0, 0.5, 0.30, 0.01)
prep_ratio   = st.sidebar.slider("Prep + Post",          0.0, 0.5, 0.40, 0.01)

# ───────────────────────────────
def compute_core_metrics(a1, a2, a3, b0, qual, sched,
                         T1, T2, T3, cross_ratio, prep_ratio, Lunit):
    # grade factors
    qual_T, qual_B   = (1, 1) if qual=="Standard" else (2/3, 0.8)
    sched_T, sched_B = (1, 1) if sched=="OnTime"   else (2/3, 0.8)
    # success
    a_total = a1 * a2 * a3
    b_eff   = b0 * qual_B * sched_B
    S       = 1 - (1 - a_total) * (1 - b_eff)
    # time cost
    T_total = (T1 + T2 + T3) * qual_T * sched_T
    C       = T_total * (1 + cross_ratio + prep_ratio)
    # loss cost
    C_loss  = C + Lunit * C * (1 - S)
    # metrics
    E_base  = C / S
    E_total = C_loss / S
    return a_total, S, C, C_loss, E_base, E_total

# compute once
a_total, S, C, C_loss, E_base, E_total = compute_core_metrics(
    a1, a2, a3, b0, qual, sched,
    T1, T2, T3, cross_ratio, prep_ratio, Lunit
)

# ───────────────────────────────
# Display metrics
left, right = st.columns([1, 2])
with left:
    st.subheader(TXT["output"])
    st.metric(TXT["a_total"], f"{a_total:.3f}")
    st.metric(TXT["succ"],    f"{S:.2%}")
    st.metric(TXT["C"],       f"{C:.1f}")
    st.metric(TXT["C_loss"],  f"{C_loss:.1f}")
    st.metric(TXT["E_base"],  f"{E_base:.1f}")
    st.metric(TXT["E_total"], f"{E_total:.1f}")

# ───────────────────────────────
# Tornado & Slope & Qual/Sched in tabs
tabs = right.tabs(["Tornado", "Sensitivity", "2×2 Scenarios"])
with tabs[0]:
    # Tornado diagram
    dark, light = "#333333", "#BBBBBB"
    st.markdown(f"**{TXT['tornado_title']}**")
    st.caption(TXT["tornado_hint"])
    def compute_E_var(k, v):
        lo = max(v*0.8, 0.0)
        hi = min(v*1.2, 1.0) if k=="b0" else v*1.2
        return max(
            abs(compute_core_metrics(
                **{f"{k}": lo} if k in ("a1","a2","a3","b0","cross_ratio","prep_ratio","Lunit") else {}
            )[-1] - E_total),
            abs(compute_core_metrics(
                **{f"{k}": hi} 
            )[-1] - E_total)
        )
    params = dict(a1=a1, a2=a2, a3=a3, b0=b0,
                  cross_ratio=cross_ratio, prep_ratio=prep_ratio, Lunit=Lunit)
    rows = [(k, compute_E_var(k, v)) for k,v in params.items()]
    df_t = pd.DataFrame(rows, columns=["param","delta"]).sort_values("delta", ascending=False)
    df_t["is_max"] = df_t["delta"] == df_t["delta"].max()
    fig_t = px.bar(df_t, x="delta", y="param", orientation="h",
                   color="is_max", color_discrete_map={True: dark, False: light},
                   labels={"delta":"|ΔE|","param":""}, title="")
    fig_t.update_layout(showlegend=False, yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig_t, use_container_width=True)

with tabs[1]:
    # Slope sensitivity
    st.markdown(f"**{TXT['spider_title']}**")
    st.caption(TXT["spider_hint"])
    slope_L = (1 - S) / S
    slope_C = 1 / S
    slope_S = abs(- (C * (1 + Lunit)) / (S**2))
    df_s = pd.DataFrame([
        ("Loss unit ℓ", slope_L),
        ("Labor C"    , slope_C),
        ("Success S"  , slope_S)
    ], columns=["Parameter","Sensitivity"])
    fig_s = px.bar(df_s, x="Parameter", y="Sensitivity", text_auto=".2f",
                   color_discrete_sequence=["#777777"])
    fig_s.update_traces(textposition="outside")
    st.plotly_chart(fig_s, use_container_width=True)

with tabs[2]:
    # 2×2 Qual vs Schedule
    st.markdown(f"**{TXT['qs_title']}**")
    st.caption(TXT["qs_hint"])
    scenarios = []
    for label, qg, sc in [
        ("Std/On", "Standard", "OnTime"),
        ("Std/Lt", "Standard", "Late"),
        ("Low/On" , "Low"     , "OnTime"),
        ("Low/Lt" , "Low"     , "Late")
    ]:
        a_t, Sx, Cx, Clx, Eb, Et = compute_core_metrics(
            a1, a2, a3, b0, qg, sc,
            T1, T2, T3, cross_ratio, prep_ratio, Lunit
        )
        scenarios.append({"Scenario":label, "E_total":Et, "Success":f"{Sx:.1%}"})
    fig_q = px.bar(pd.DataFrame(scenarios),
                   x="Scenario", y="E_total", text="Success",
                   color_discrete_sequence=["#888888"])
    fig_q.update_traces(textposition="outside")
    st.plotly_chart(fig_q, use_container_width=True)

# ───────────────────────────────
# Monte-Carlo section
if st.sidebar.checkbox("Run Monte-Carlo (100 000)"):
    mc_var = st.sidebar.selectbox("MC variable", ["E_total","Success S"])
    st.markdown(f"**{TXT['mc_title']} – {mc_var}**")
    st.caption(TXT["mc_hint"])

    N, rng = 100_000, np.random.default_rng(0)
    a1s = rng.normal(a1, 0.03, N).clip(0,1)
    a2s = rng.normal(a2, 0.03, N).clip(0,1)
    a3s = rng.normal(a3, 0.03, N).clip(0,1)
    CRs = rng.triangular(cross_ratio*0.8, cross_ratio, cross_ratio*1.2, N)
    PPs = rng.triangular(prep_ratio*0.8, prep_ratio, prep_ratio*1.2, N)
    b0s = rng.uniform(0.70, 0.90, N)
    Ls  = rng.lognormal(mean=np.log(Lunit+1), sigma=0.3, size=N)

    results = []
    for a1v, a2v, a3v, CRv, PPv, b0v, Lv in zip(a1s, a2s, a3s, CRs, PPs, b0s, Ls):
        a_tot = a1v * a2v * a3v
        b_e   = b0v * (1 if qual=="Standard" else 0.8) * (1 if sched=="OnTime" else 0.8)
        S_i   = 1 - (1 - a_tot) * (1 - b_e)
        C_i   = (T1 + T2 + T3) * (1 if qual=="Standard" else 2/3) * (1 if sched=="OnTime" else 2/3) * (1 + CRv + prep_ratio)
        E_i   = (C_i + Lv * C_i * (1 - S_i)) / S_i
        results.append((E_i, S_i))

    data = np.array([r[0] if mc_var=="E_total" else r[1] for r in results])
    label = "E_total" if mc_var=="E_total" else "Success S"

    # summary
    mean, median = data.mean(), np.median(data)
    p5, p95      = np.percentile(data, [5,95])
    st.write("---")
    st.subheader("Monte-Carlo Summary Statistics")
    st.metric("Mean",   f"{mean:.2f}")
    st.metric("Median", f"{median:.2f}")
    st.metric("5–95 % CI", f"{p5:.2f} – {p95:.2f}")

    # histogram
    fig_h = px.histogram(data, nbins=35, labels={"value":label})
    fig_h.update_traces(marker_color="#888888")
    # vertical lines
    for x, style, txt in [
        (mean,   "solid",  f"Mean: {mean:.2f}"),
        (median, "solid",  f"Median: {median:.2f}"),
        (p5,     "dot",    f"5 %: {p5:.2f}"),
        (p95,    "dot",    f"95 %: {p95:.2f}")
    ]:
        fig_h.add_vline(x=x, line_dash=style,
                        line_color="#222222" if style=="solid" else "#555555",
                        annotation_text=txt, annotation_position="top right")
    st.plotly_chart(fig_h, use_container_width=True)
