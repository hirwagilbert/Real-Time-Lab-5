"""
Mini Project 5.2 - Monte Carlo Reliability Simulator: Simplex vs TMR (and 5MR/7MR)
                                                    (Real-Time Systems, Unit 5)

Fault model (Section 5.1) - transient and permanent faults are SEPARATE processes:
  * each hour a module that is up draws one uniform number u:
        u < lambda * PERM_SHARE          -> permanent fault, down REPAIR_H hours
        u < lambda                       -> transient fault, down TRANSIENT_H hour(s)
  * a K-of-N majority voter: the system is down while fewer than K = N//2 + 1
    modules are up (TMR: 2 of 3, 5MR: 3 of 5, 7MR: 4 of 7).

Speed: drawing a uniform number every hour until the first failure is
statistically identical to drawing the failure time from a geometric
distribution, so the sweep "fast-forwards" with geometric jumps. Module.step()
implements the literal hour-by-hour version and self_test() shows both agree.

Statistics: a 20-trial pilot estimates the failure count per trial; the number
of fresh trials is then chosen so that about (1.96/0.05)^2 = 1537 failures are
expected, giving a 95 % confidence interval on MTBF of about +-5 % (capped at
MAX_TRIALS). For zero observed failures the one-sided 95 % lower bound (rule of
three) is reported.

Outputs (../outputs/): mp5_2_sweep_results.csv, mp5_2_recommendations.csv and,
if matplotlib is installed, mp5_2_*.png plots.
Python 3.8+ standard library only (matplotlib optional).
Run:  python monte_carlo_nmr.py           (about 30-90 s)
      python monte_carlo_nmr.py --quick   (smaller sweep, a few seconds)
"""

import csv
import math
import os
import random
import sys
from statistics import NormalDist

SEED = 2026
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
QUICK = "--quick" in sys.argv

# ---------------- documented parameters ----------------
RATES = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]   # per module per hour
if QUICK:
    RATES = [0.0001, 0.001, 0.01]
PERM_SHARE = 0.30          # 30 % of faults are permanent, 70 % transient
TRANSIENT_H = 1            # transient fault clears after 1 h (e.g. SEU, scrub/reset)
REPAIR_H = 24              # permanent fault repaired after 24 h
TRIAL_HOURS = 100_000      # simulated hours per trial
MISSION_H = 100            # mission time for reliability R(t)
BATCH = 20                 # trials added per batch
MAX_TRIALS = 400 if QUICK else 3000
TARGET_REL_HALF_WIDTH = 0.05
CONFIGS = [("Simplex", 1), ("TMR", 3), ("5MR", 5), ("7MR", 7)]
Z = NormalDist().inv_cdf(0.975)


# ---------------------------------------------------------------------------
# Module (Suggested approach step 1)
# ---------------------------------------------------------------------------
class Module:
    def __init__(self, failure_rate, rng):
        self.failure_rate = failure_rate
        self.rng = rng
        self.healthy = True
        self.fault_type = None          # 'transient' | 'permanent' | None
        self.down_left = 0

    def step(self):
        """Advance one hour. Returns True if the module is DOWN during this hour."""
        if not self.healthy:
            self.down_left -= 1
            if self.down_left <= 0:
                self.healthy, self.fault_type = True, None
            else:
                return True
        u = self.rng.random()
        if u < self.failure_rate:
            self.healthy = False
            self.fault_type = "permanent" if u < self.failure_rate * PERM_SHARE else "transient"
            self.down_left = REPAIR_H if self.fault_type == "permanent" else TRANSIENT_H
            return True
        return False

    def outages(self, hours):
        """Fast-forward equivalent of calling step() `hours` times: returns the
        list of (start, end) down intervals using geometric jumps."""
        out, t, lam, rng = [], 0, self.failure_rate, self.rng
        log1m = math.log1p(-lam)
        while True:
            t += int(math.log(1.0 - rng.random()) / log1m)   # up hours before failure
            if t >= hours:
                return out
            perm = rng.random() < PERM_SHARE
            d = REPAIR_H if perm else TRANSIENT_H
            out.append((t, min(t + d, hours), "P" if perm else "T"))
            t += d


# ---------------------------------------------------------------------------
# K-of-N voter (steps 2 and 6)
# ---------------------------------------------------------------------------
def voter_ok(module_down_flags, k):
    """Generalised voter: the output is valid if at least k modules agree (are up)."""
    return sum(1 for d in module_down_flags if not d) >= k


def system_outages(module_outages, n, hours):
    """Merged intervals during which fewer than K = n//2+1 modules are up."""
    k = n // 2 + 1
    m = n - k + 1                          # modules down needed to fail the system
    events = []
    for iv in module_outages:
        for s, e, _ in iv:
            events.append((s, 1))
            events.append((e, -1))
    events.sort(key=lambda x: (x[0], x[1]))
    out, active, start = [], 0, None
    for t, d in events:
        active += d
        if active >= m and start is None:
            start = t
        elif active < m and start is not None:
            if out and start <= out[-1][1]:
                out[-1] = (out[-1][0], t)
            else:
                out.append((start, t))
            start = None
    return out


def run_trial(rate, n, rng, hours=TRIAL_HOURS):
    mods = [Module(rate, random.Random(rng.randrange(2**63))).outages(hours) for _ in range(n)]
    sys_out = mods[0] if n == 1 else system_outages(mods, n, hours)
    sys_out = [(s, e) for s, e, *_ in sys_out]
    down = sum(e - s for s, e in sys_out)
    # mission reliability: consecutive MISSION_H windows that start up and see no failure
    starts = sorted(s for s, _ in sys_out)
    windows = hours // MISSION_H
    bad = set(s // MISSION_H for s in starts)
    # windows that start inside an outage are also unsuccessful
    for s, e in sys_out:
        w0 = -(-s // MISSION_H)                  # first window start >= s
        for w in range(w0, min(windows, -(-e // MISSION_H))):
            if w * MISSION_H < e:
                bad.add(w)
    ok = windows - len([w for w in bad if w < windows])
    return len(sys_out), down, ok, windows


# ---------------------------------------------------------------------------
# Analytical model (Sections 5.1, 5.2)
# ---------------------------------------------------------------------------
def analytic(rate, n):
    mttr_mod = (1 - PERM_SHARE) * TRANSIENT_H + PERM_SHARE * REPAIR_H
    if n == 1:
        mtbf = 1 / rate
        a = mtbf / (mtbf + mttr_mod)
        return {"mtbf": mtbf, "mttr": mttr_mod, "availability": a,
                "reliability": a * math.exp(-MISSION_H / mtbf)}
    k = n // 2 + 1
    m = n - k + 1
    u = rate * mttr_mod / (1 + rate * mttr_mod)           # module unavailability
    # discrete-time overlap correction: a fault that starts in the same hour as
    # another only overlaps it in one ordering -> effective duration d - 0.5
    d_eff = (1 - PERM_SHARE) * (TRANSIENT_H - 0.5) + PERM_SHARE * (REPAIR_H - 0.5)
    u_eff = rate * d_eff / (1 + rate * mttr_mod)
    p_down = sum(math.comb(n, i) * u**i * (1 - u)**(n - i) for i in range(m, n + 1))
    # failure frequency = (m-1 modules already down) x (one more fails)
    freq = math.comb(n, m - 1) * u_eff * u**(m - 2) * (1 - u)**(n - m + 1) * (n - m + 1) * rate
    mtbf = (1 - p_down) / freq
    mttr = p_down / freq
    a = mtbf / (mtbf + mttr)                                # = 1 - p_down
    return {"mtbf": mtbf, "mttr": mttr, "availability": a,
            "reliability": a * math.exp(-MISSION_H / mtbf)}


def classic_mission_reliability(rate, n, t=MISSION_H):
    """Non-repairable textbook form: R_NMR = sum_{i>=K} C(N,i) R^i (1-R)^(N-i)."""
    r = math.exp(-rate * t)
    k = n // 2 + 1
    return sum(math.comb(n, i) * r**i * (1 - r)**(n - i) for i in range(k, n + 1))


# ---------------------------------------------------------------------------
# Confidence intervals (step 5)
# ---------------------------------------------------------------------------
def chi2_quantile(p, dof):
    """Wilson-Hilferty approximation of the chi-square quantile (no SciPy)."""
    z = NormalDist().inv_cdf(p)
    return dof * (1 - 2 / (9 * dof) + z * math.sqrt(2 / (9 * dof))) ** 3


def mtbf_ci(uptime, failures):
    """95 % CI for MTBF from a Poisson failure count (time-terminated test)."""
    if failures == 0:
        return uptime / 3.0, math.inf                # one-sided 95 % (rule of three)
    lo = 2 * uptime / chi2_quantile(0.975, 2 * failures + 2)
    hi = 2 * uptime / chi2_quantile(0.025, 2 * failures)
    return lo, hi


# ---------------------------------------------------------------------------
# Sweep driver (steps 3-6)
# ---------------------------------------------------------------------------
def sweep():
    rng = random.Random(SEED)
    rows = []
    for rate in RATES:
        for name, n in CONFIGS:
            # Pilot run -> required trial count (avoids optional-stopping bias):
            # a Poisson count of n failures gives a relative 95 % half-width of
            # about 1.96/sqrt(n), so n* = (1.96 / target)^2 failures are needed.
            pilot = sum(run_trial(rate, n, rng)[0] for _ in range(BATCH))
            need = (Z / TARGET_REL_HALF_WIDTH) ** 2
            per_trial = pilot / BATCH
            planned = MAX_TRIALS if pilot == 0 else min(MAX_TRIALS,
                                                        max(BATCH, math.ceil(need / per_trial)))
            fails = down = ok = windows = trials = 0
            for _ in range(planned):                     # fresh trials, pilot discarded
                f, d, o, w = run_trial(rate, n, rng)
                fails += f; down += d; ok += o; windows += w; trials += 1
            width = Z / math.sqrt(fails) if fails else math.inf
            total = trials * TRIAL_HOURS
            up = total - down
            lo, hi = mtbf_ci(up, fails)
            emp_mtbf = up / fails if fails else math.inf
            ana = analytic(rate, n)
            row = {
                "failure_rate": rate, "config": name, "modules": n, "trials": trials,
                "sim_hours": total, "system_failures": fails,
                "mtbf_emp": emp_mtbf, "mtbf_ci_low": lo, "mtbf_ci_high": hi,
                "mtbf_ana": ana["mtbf"],
                "mtbf_dev_pct": (emp_mtbf - ana["mtbf"]) / ana["mtbf"] * 100 if fails >= 30 else None,
                "mttr_emp": down / fails if fails else 0.0, "mttr_ana": ana["mttr"],
                "avail_emp": up / total, "avail_ana": ana["availability"],
                "avail_dev_pct": ((up / total) - ana["availability"]) / ana["availability"] * 100,
                "rel_emp": ok / windows, "rel_ana": ana["reliability"],
                "rel_classic_norepair": classic_mission_reliability(rate, n),
                "ci_rel_half_width_pct": 100 * width if fails else None,
            }
            rows.append(row)
            print(f"{rate:<8g}{name:<8}{trials:>6}{fails:>9}"
                  f"{fmt(emp_mtbf):>14}{fmt(ana['mtbf']):>14}"
                  f"{pct(row['mtbf_dev_pct']):>8}{ci_txt(lo, hi):>26}"
                  f"{up / total:>11.6f}{ana['availability']:>11.6f}"
                  f"{row['rel_emp']:>9.4f}{ana['reliability']:>9.4f}")
    return rows


def fmt(x):
    if x is None or not math.isfinite(x):
        return "no fail"
    return f"{x:,.0f}" if x < 1e7 else f"{x:.2e}"


def pct(x):
    return " n<30" if x is None else f"{x:+.1f}"


def ci_txt(lo, hi):
    if not math.isfinite(hi):
        return f">= {fmt(lo)} (95%)"
    return f"[{fmt(lo)}, {fmt(hi)}]"


# ---------------------------------------------------------------------------
# Recommendation at three operating points (report requirement)
# ---------------------------------------------------------------------------
REQ_AVAIL = 0.999           # three nines
REQ_REL = 0.99              # 99 % chance of a 100-hour mission without an outage


def recommend(rows):
    print(f"\nRequirement used for the recommendation: availability >= {REQ_AVAIL} and "
          f"R({MISSION_H} h) >= {REQ_REL}; choose the cheapest configuration that meets both.")
    recs = []
    for rate in (0.0001, 0.001, 0.01):
        pts = [r for r in rows if r["failure_rate"] == rate]
        if not pts:
            continue
        ok = [r for r in pts if r["avail_ana"] >= REQ_AVAIL and r["rel_ana"] >= REQ_REL]
        pick = min(ok, key=lambda r: r["modules"]) if ok else None
        print(f"  lambda = {rate:g}/h:")
        for r in pts:
            unav = 1 - r["avail_ana"]
            print(f"     {r['config']:<8} cost {r['modules']}x  A = {r['avail_ana']:.6f} "
                  f"(unavailability {unav:.2e})  R({MISSION_H}h) = {r['rel_ana']:.4f}"
                  f"{'  <-- meets requirement' if r in ok else ''}")
        print(f"     RECOMMENDATION: {pick['config'] if pick else 'none meets it - improve the module (lower lambda or MTTR) instead of adding redundancy'}")
        recs.append({"failure_rate": rate, "recommended": pick["config"] if pick else "none",
                     **{f"A_{r['config']}": round(r["avail_ana"], 6) for r in pts},
                     **{f"R100_{r['config']}": round(r["rel_ana"], 4) for r in pts}})
    return recs


def self_test():
    """Step-by-step Module.step() and the fast-forward sampler give the same statistics."""
    lam, hours = 0.01, 2_000_000
    m = Module(lam, random.Random(1))
    down_step = sum(1 for _ in range(hours) if m.step())
    iv = Module(lam, random.Random(2)).outages(hours)
    down_fast = sum(e - s for s, e, _ in iv)
    print(f"Self-test (lambda=0.01, 2,000,000 h): unavailability step-by-step "
          f"{down_step / hours:.4f}, fast-forward {down_fast / hours:.4f}, "
          f"analytical {analytic(lam, 1)['mttr'] * lam / (1 + lam * analytic(lam, 1)['mttr']):.4f}")
    # voter check
    assert voter_ok([False, False, True], 2) and not voter_ok([True, False, True], 2)
    assert voter_ok([True, True, False, False, False], 3)


def plots(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plots (CSV contains all data).")
        return
    colors = {"Simplex": "#2a78d6", "TMR": "#eb6834", "5MR": "#1baf7a", "7MR": "#eda100"}
    marks = {"Simplex": "o", "TMR": "s", "5MR": "^", "7MR": "D"}
    for metric, ylabel, fname, logy in (
            ("mtbf", "System MTBF (hours, log scale)", "mp5_2_mtbf.png", True),
            ("unavail", "Unavailability 1 - A (log scale)", "mp5_2_unavailability.png", True),
            ("rel", f"Mission reliability R({MISSION_H} h)", "mp5_2_reliability.png", False)):
        fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=200)
        for name, _ in CONFIGS:
            pts = [r for r in rows if r["config"] == name]
            x = [r["failure_rate"] for r in pts]
            if metric == "mtbf":
                ya = [r["mtbf_ana"] for r in pts]
                ye = [(r["failure_rate"], r["mtbf_emp"]) for r in pts if math.isfinite(r["mtbf_emp"])]
            elif metric == "unavail":
                ya = [1 - r["avail_ana"] for r in pts]
                ye = [(r["failure_rate"], 1 - r["avail_emp"]) for r in pts if r["avail_emp"] < 1]
            else:
                ya = [r["rel_ana"] for r in pts]
                ye = [(r["failure_rate"], r["rel_emp"]) for r in pts]
            ax.plot(x, ya, color=colors[name], lw=2, label=f"{name} analytical")
            if ye:
                ax.plot([p[0] for p in ye], [p[1] for p in ye], ls="none", marker=marks[name],
                        ms=6, color=colors[name], markeredgecolor="white", label=f"{name} simulated")
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("Per-module failure rate lambda (per hour, log scale)", color="#5f5e5a")
        ax.set_ylabel(ylabel, color="#5f5e5a")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(color="#e6e5e0", lw=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, fname))
        plt.close(fig)
    print("Saved plots: outputs/mp5_2_mtbf.png, mp5_2_unavailability.png, mp5_2_reliability.png")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 112)
    print("MINI PROJECT 5.2 - MONTE CARLO RELIABILITY: SIMPLEX vs TMR vs 5MR vs 7MR")
    print("=" * 112)
    print(f"Faults: {int(PERM_SHARE * 100)}% permanent (repair {REPAIR_H} h), "
          f"{100 - int(PERM_SHARE * 100)}% transient ({TRANSIENT_H} h). Trial = {TRIAL_HOURS:,} h, "
          f"pilot of {BATCH} trials sizes the run for a +-{int(TARGET_REL_HALF_WIDTH * 100)}% CI "
          f"(cap {MAX_TRIALS} trials). Seed {SEED}.")
    self_test()
    print(f"\n{'lambda':<8}{'config':<8}{'trials':>6}{'failures':>9}{'MTBF emp':>14}"
          f"{'MTBF ana':>14}{'dev %':>8}{'95% CI on MTBF':>26}{'A emp':>11}{'A ana':>11}"
          f"{'R emp':>9}{'R ana':>9}")
    rows = sweep()
    with open(os.path.join(OUT_DIR, "mp5_2_sweep_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 8) if isinstance(v, float) and math.isfinite(v) else v)
                        for k, v in r.items()})
    recs = recommend(rows)
    if recs:
        with open(os.path.join(OUT_DIR, "mp5_2_recommendations.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
            w.writeheader()
            w.writerows(recs)
    plots(rows)
    print("Saved: outputs/mp5_2_sweep_results.csv, outputs/mp5_2_recommendations.csv")


if __name__ == "__main__":
    main()
