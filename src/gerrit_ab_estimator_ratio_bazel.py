# src/gerrit_ab_estimator_ratio_bazel.py
# ──────────────────────────────────────────────────────────────
# ❶ Pulls Gerrit change metadata (since 2023-01-01) for the
#    projects/hosts below
# ❷ Prints per-project  a_hat  and  b_dir
# ❸ Saves three artefacts into ./data/ (dated):
#      · gerrit_result_YYYYMMDD.txt   – console table
#      · gerrit_sample_all_YYYYMMDD.json
#      · gerrit_sample_corrected_YYYYMMDD.json
# ──────────────────────────────────────────────────────────────
import re, json, os
from datetime import datetime
from urllib.parse import quote
from pygerrit2 import GerritRestAPI

# -------------- CONFIG --------------------------------------------------
HOSTS = {
    "https://gerrit-review.googlesource.com": [
        "plugins/owners", "plugins/replication", "plugins/checks",
    ],
    "https://bazel-review.googlesource.com": [
        "bazel",
    ],
    "https://android-review.googlesource.com": [
        "platform/external/okhttp",
    ],
}
SINCE = "2023-01-01"
OPT   = "o=CURRENT_REVISION&o=MESSAGES"

OUTDIR = "data"
os.makedirs(OUTDIR, exist_ok=True)
today  = datetime.today().strftime("%Y%m%d")
txt_path  = os.path.join(OUTDIR, f"gerrit_result_{today}.txt")
all_path  = os.path.join(OUTDIR, f"gerrit_sample_all_{today}.json")
corr_path = os.path.join(OUTDIR, f"gerrit_sample_corrected_{today}.json")

# -------------- DEFECT-DETECTION HEURISTIC ------------------------------
PAT = re.compile(r"\b(nit|fix|typo|minor)\b", re.I)
def chk(text:str)->bool:
    return ("Code-Review" in text and ("-1" in text or "-2" in text)) or \
           ("Verified"    in text and "-1" in text) or bool(PAT.search(text))

def worker_ok(c)  -> bool: return not any(chk(m["message"]) for m in c["messages"])
def defect_hit(c) -> bool: return any(chk(m["message"]) for m in c["messages"])

# -------------- API FETCH -----------------------------------------------
def fetch_changes(rest: GerritRestAPI, project:str):
    pq = quote(f"project:{project}", safe=":")
    out, more, s = [], True, 0
    while more:
        q = f"/changes/?q={pq}+status:merged+after:{SINCE}&n=100&{OPT}&S={s}"
        chunk = rest.get(q)
        if not chunk: break
        out.extend(chunk)
        more = chunk[-1].get("_more_changes", False)
        s   += 100
    return out

# -------------- AGGREGATION ---------------------------------------------
rows, all_changes = [], []
for host, projects in HOSTS.items():
    rest = GerritRestAPI(url=host)
    for prj in projects:
        ch = fetch_changes(rest, prj)
        N  = len(ch)
        if N == 0: continue
        a_hat = sum(worker_ok(c)  for c in ch) / N
        b_dir = sum(defect_hit(c) for c in ch) / N
        rows.append((prj, N, a_hat, b_dir))
        all_changes.extend(ch)

# legacy = exclude bazel + okhttp
legacy_rows = [r for r in rows if r[0] not in {"bazel","platform/external/okhttp"}]
N_leg = sum(r[1] for r in legacy_rows)
a_leg = sum(r[2]*r[1] for r in legacy_rows)/N_leg
b_leg = sum(r[3]*r[1] for r in legacy_rows)/N_leg

# -------------- CONSOLE & TXT OUTPUT ------------------------------------
header = f"{'Project':25s} {'N':>6s}   {'a_hat':>6s}   {'b_dir':>6s}"
lines  = [header]
for p,n,a,b in sorted(rows, key=lambda x: x[0]):
    lines.append(f"{p:25s} {n:6d}   {a:6.3f}   {b:6.3f}")

Ntot  = len(all_changes)
a_tot = sum(worker_ok(c)  for c in all_changes)/Ntot
b_tot = sum(defect_hit(c) for c in all_changes)/Ntot
lines.append(f"\nWeighted avg (all)        {Ntot:6d}   {a_tot:.3f}   {b_tot:.3f}")
lines.append(f"Weighted avg (legacy)     {N_leg:6d}   {a_leg:.3f}   {b_leg:.3f}")

out_txt = "\n".join(lines)
print(out_txt)

with open(txt_path, "w") as f:
    f.write(out_txt)
print(f">> Saved text table to {txt_path}")

# -------------- JSON SAMPLES --------------------------------------------
# (i) first 50 merged changes as-is
with open(all_path, "w") as f:
    json.dump(all_changes[:50], f, indent=2)

# (ii) first 50 “defect-detected” (= cross-checked) changes
corrected = [c for c in all_changes if defect_hit(c)]
with open(corr_path, "w") as f:
    json.dump(corrected[:50], f, indent=2)

print(f">> Saved 50-row ALL sample       to {all_path}")
print(f">> Saved 50-row CORRECTED sample to {corr_path}")

