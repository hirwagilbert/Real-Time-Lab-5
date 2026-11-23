"""
Mini Project 5.1 - Simulating TMR Voting, Reliability Metrics and
Watchdog-Based Recovery                       (Real-Time Systems, Unit 5)

Covers the full procedure (steps 1-10) and the four Lab Tasks:
  * simplex vs TMR under TRANSIENT and PERMANENT faults (kept distinct)
  * empirical vs analytical MTBF and availability (Section 5.2)
  * software watchdog: missed heartbeat -> forced restart, MTTR samples
  * watchdog timeout sweep (MTTR lever with MTBF fixed)
  * Lab Task 1  diverse (non-identical) module failure rates
  * Lab Task 2  bounded vs unbounded voter overhead
  * Lab Task 3  watchdog false positives vs true positives
  * Lab Task 4  availability vs MTTR, point of diminishing returns

Time unit: 1 hour. Python 3.8+ standard library only.
Run:  python reliability_sim.py          (about 20-40 s on a laptop)
      python reliability_sim.py --quick  (fewer trials, a few seconds)
"""

import csv
import math
import os
import random
import statistics
import sys

random.seed(42)                                   # Procedure step 1
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
QUICK = "--quick" in sys.argv

HOURS = 50_000 if QUICK else 200_000              # simulated hours per trial
TRIALS = 10 if QUICK else 30                      # independent trials
FAILURE_RATE = 0.001                              # per module per hour
REPAIR_HOURS = 24                                 # permanent fault: repair delay
TRANSIENT_HOURS = 1                               # transient fault: clears next hour


# ---------------------------------------------------------------------------
# Fault model (Procedure step 2)
# ---------------------------------------------------------------------------
def simulate_module(hours, failure_rate, fault_type, rng, repair_hours=REPAIR_HOURS):
    """Step hour by hour. While the module is up, draw a uniform number; below
    failure_rate the module fails in that hour.
      transient : down only for the failure hour, recovers on the next hour
      permanent : stays down until an explicit repair delay has elapsed
    Returns (down_intervals [(start, end_exclusive)], failure_count)."""
    if fault_type == "transient":
        down_for = TRANSIENT_HOURS
    elif fault_type == "permanent":
        down_for = repair_hours
    else:
        raise ValueError("fault_type must be 'transient' or 'permanent'")
    intervals, failures = [], 0
    t = 0
    rnd = rng.random
    while t < hours:
        if rnd() < failure_rate:                 # failure drawn for hour t
            failures += 1
            end = min(t + down_for, hours)
            intervals.append((t, end))
            t = end                              # no draws while down
        else:
            t += 1
    return intervals, failures


def downtime(intervals):
    return sum(e - s for s, e in intervals)


def k_of_n_down(module_intervals, k, hours):
    """System is down whenever at least k modules are down at the same time
    (for TMR, k = 2: the voter cannot form a majority). Event sweep.
    Returns (system_down_intervals)."""
    events = []
    for iv in module_intervals:
        for s, e in iv:
            events.append((s, 1))
            events.append((e, -1))
    events.sort(key=lambda x: (x[0], x[1]))      # ends before starts at same hour
    active, out, start = 0, [], None
    for t, d in events:
        active += d
        if active >= k and start is None:
            start = t
        elif active < k and start is not None:
            if t > start:
                out.append((start, t))
            start = None
    # merge touching intervals into one outage
    merged = []
    for s, e in out:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(e, merged[-1][1]))
        else:
            merged.append((s, e))
    return merged


def metrics(intervals, hours):
    """Empirical figures for one trial."""
    down = downtime(intervals)
    fails = len(intervals)
    up = hours - down
    return {"failures": fails, "downtime": down,
            "mtbf_up": up / fails if fails else math.inf,       # mean up time per failure
            "mtbf_total": hours / fails if fails else math.inf, # manual: hours / failures
            "mttr": down / fails if fails else 0.0,
            "availability": up / hours}


# ---------------------------------------------------------------------------
# Simplex and TMR simulators (Procedure steps 3-4)
# ---------------------------------------------------------------------------
def run_simplex(hours, rate, fault_type, rng, repair_hours=REPAIR_HOURS):
    iv, _ = simulate_module(hours, rate, fault_type, rng, repair_hours)
    return metrics(iv, hours)


def run_nmr(hours, rates, fault_type, seeds, k=None, repair_hours=REPAIR_HOURS):
    """N modular redundancy; each module has its own independently seeded RNG.
    TMR = 3 modules, system down if >= 2 modules down."""
    n = len(rates)
    k = k or (n // 2 + 1)                 # majority voter
    mods = [simulate_module(hours, r, fault_type, random.Random(s), repair_hours)[0]
            for r, s in zip(rates, seeds)]
    return metrics(k_of_n_down(mods, k, hours), hours)


# ---------------------------------------------------------------------------
# Analytical model (Section 5.2; Procedure step 6)
# ---------------------------------------------------------------------------
def analytic_simplex(rate, mttr):
    mtbf = 1 / rate
    return {"mtbf": mtbf, "mttr": mttr, "availability": mtbf / (mtbf + mttr)}


def p_two_of_three_down(u1, u2, u3):
    """Probability that at least 2 of 3 independent modules are down."""
    return (u1 * u2 * (1 - u3) + u1 * u3 * (1 - u2) + u2 * u3 * (1 - u1) + u1 * u2 * u3)


def analytic_tmr(rates, fault_type, repair_hours=REPAIR_HOURS):
    """Each module is down a fraction u = MTTR/(MTBF+MTTR) of the time. The voter
    fails while >= 2 modules are down: P2 = 3u^2(1-u) + u^3 (identical modules).
    Transient faults: the system is down only in an hour where two modules fail
    together, so MTTR_sys = 1 h. Permanent faults: an outage lasts from the second
    failure to the end of the first repair, on average about half a repair,
    MTTR_sys = R/2. Then, from A = MTBF/(MTBF+MTTR):  MTBF_sys = MTTR_sys (1-P2)/P2."""
    d = TRANSIENT_HOURS if fault_type == "transient" else repair_hours
    if fault_type == "transient":
        q = rates                                  # per-hour failure probability
        p2 = p_two_of_three_down(*q)
        mttr_sys = 1.0
    else:
        u = [d / (1 / r + d) for r in rates]
        p2 = p_two_of_three_down(*u)
        mttr_sys = d / 2
    mtbf = mttr_sys * (1 - p2) / p2
    return {"p_down": p2, "mtbf": mtbf, "mttr": mttr_sys,
            "availability": mtbf / (mtbf + mttr_sys)}


def pooled(results):
    """Pool trials: total up time / total failures (robust when a trial has 0 failures)."""
    fails = sum(r["failures"] for r in results)
    down = sum(r["downtime"] for r in results)
    total = HOURS * len(results)
    finite = [r["mtbf_up"] for r in results if math.isfinite(r["mtbf_up"])]
    return {"failures": fails,
            "mtbf": (total - down) / fails if fails else math.inf,
            "mtbf_total": total / fails if fails else math.inf,
            "mttr": down / fails if fails else 0.0,
            "availability": (total - down) / total,
            "zero_fail_trials": sum(1 for r in results if r["failures"] == 0),
            "fails_min": min(r["failures"] for r in results),
            "fails_max": max(r["failures"] for r in results),
            "mtbf_sd": statistics.stdev(finite) if len(finite) > 1 else 0.0}


def dev(emp, ana):
    return 100 * (emp - ana) / ana


# ---------------------------------------------------------------------------
# Watchdog (Procedure steps 7-9)
# ---------------------------------------------------------------------------
HEARTBEAT_PERIOD = 0.25        # the task kicks the watchdog every 15 minutes


def simulate_watchdog(num_hangs, timeout, restart_time, rng):
    """Inject num_hangs randomly-timed hangs into a periodic task. The task kicks
    the watchdog every HEARTBEAT_PERIOD; after a hang no more kicks arrive and the
    watchdog fires `timeout` after the LAST kick. One MTTR sample per hang =
    detection delay (about the timeout) + forced restart time."""
    samples = []
    for _ in range(num_hangs):
        phase = rng.uniform(0, HEARTBEAT_PERIOD)       # hang time after last kick
        detection = timeout - phase                     # watchdog fires at kick + timeout
        samples.append(detection + restart_time)
    return samples


def simulate_simplex_with_watchdog(hours, rate, timeout, restart_time, rng):
    """End-to-end check: every module failure is a hang recovered by the watchdog;
    downtime per failure comes from the watchdog model (fractional hours)."""
    t, down, fails = 0, 0.0, 0
    while t < hours:
        if rng.random() < rate:
            fails += 1
            mttr = simulate_watchdog(1, timeout, restart_time, rng)[0]
            down += mttr
            t += max(1, math.ceil(mttr))
        else:
            t += 1
    return (hours - down) / hours, fails


def simulate_simplex_fixed_mttr(hours, rate, mttr, rng):
    """Simplex module whose every failure is repaired in exactly `mttr` hours."""
    t, down = 0, 0.0
    while t < hours:
        if rng.random() < rate:
            down += mttr
            t += max(1, math.ceil(mttr))
        else:
            t += 1
    return (hours - down) / hours


# ---------------------------------------------------------------------------
# Lab Task 3: watchdog false positives
# ---------------------------------------------------------------------------
def watchdog_fp_experiment(timeout, n_beats=200_000, hang_prob=0.0005,
                           slow_prob=0.02, slow_mean=0.5, seed=7):
    """Heartbeat gaps are normally one period (+-10% jitter). With probability
    slow_prob a legitimate long job delays the next kick by an exponential extra
    time (mean slow_mean h) - the task is alive, just late. With probability
    hang_prob the task really hangs. The watchdog restarts the task whenever the
    gap exceeds `timeout`: true positive if hung, false positive if merely late."""
    rng = random.Random(seed)
    tp = fp = 0
    for _ in range(n_beats):
        if rng.random() < hang_prob:
            tp += 1                                 # hung: gap is infinite
            continue
        gap = HEARTBEAT_PERIOD * rng.uniform(0.9, 1.1)
        if rng.random() < slow_prob:
            gap += rng.expovariate(1 / slow_mean)
        if gap > timeout:
            fp += 1
    return tp, fp


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def fmt(x, nd=1):
    return "inf" if not math.isfinite(x) else f"{x:,.{nd}f}"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []                         # summary CSV
    print("=" * 78)
    print("MINI PROJECT 5.1 - TMR VOTING, RELIABILITY METRICS AND WATCHDOG RECOVERY")
    print("=" * 78)
    print(f"Per-module failure rate = {FAILURE_RATE}/h, {HOURS:,} h per trial, {TRIALS} trials, "
          f"seed 42")
    print(f"Transient fault: down {TRANSIENT_HOURS} h.  Permanent fault: down until repaired "
          f"({REPAIR_HOURS} h).")

    # ---------------- steps 3-6: simplex vs TMR ----------------
    emp = {}
    for ft in ("transient", "permanent"):
        mttr = TRANSIENT_HOURS if ft == "transient" else REPAIR_HOURS
        s_res, t_res = [], []
        for trial in range(TRIALS):
            seed = random.randrange(10**9)
            s_res.append(run_simplex(HOURS, FAILURE_RATE, ft, random.Random(seed)))
            t_res.append(run_nmr(HOURS, [FAILURE_RATE] * 3, ft,
                                 [seed + 1, seed + 2, seed + 3]))
        S, T = pooled(s_res), pooled(t_res)
        aS, aT = analytic_simplex(FAILURE_RATE, mttr), analytic_tmr([FAILURE_RATE] * 3, ft)
        emp[ft] = (S, T, aS, aT)
        print(f"\n--- {ft.upper()} faults: empirical (pooled over {TRIALS} trials) vs analytical ---")
        print(f"{'config':<8}{'failures':>9}{'MTBF emp':>12}{'MTBF ana':>12}{'dev %':>8}"
              f"{'MTTR emp':>10}{'Avail emp':>11}{'Avail ana':>11}")
        for name, E, Aa in (("Simplex", S, aS), ("TMR", T, aT)):
            print(f"{name:<8}{E['failures']:>9}{fmt(E['mtbf']):>12}{fmt(Aa['mtbf']):>12}"
                  f"{dev(E['mtbf'], Aa['mtbf']):>8.1f}{E['mttr']:>10.2f}"
                  f"{E['availability']:>11.6f}{Aa['availability']:>11.6f}")
            rows.append({"experiment": f"{ft} faults", "config": name,
                         "failures": E["failures"], "mtbf_emp_h": round(E["mtbf"], 1),
                         "mtbf_ana_h": round(Aa["mtbf"], 1),
                         "mtbf_dev_pct": round(dev(E["mtbf"], Aa["mtbf"]), 2),
                         "mttr_emp_h": round(E["mttr"], 2),
                         "avail_emp": round(E["availability"], 6),
                         "avail_ana": round(Aa["availability"], 6)})
        print(f"  TMR/simplex MTBF gain: empirical x{T['mtbf'] / S['mtbf']:,.1f}, "
              f"analytical x{aT['mtbf'] / aS['mtbf']:,.1f}")
        print(f"  Manual's 'hours / failures' simplex MTBF = {fmt(S['mtbf_total'])} h "
              f"(= MTBF + MTTR, the full failure-repair cycle)")
        print(f"  TMR failures per trial: min {T['fails_min']}, max {T['fails_max']}, "
              f"trials with zero failures: {T['zero_fail_trials']}/{TRIALS}")

    # ---------------- steps 7-9: watchdog ----------------
    S_perm = emp["permanent"][0]
    mtbf_emp = S_perm["mtbf"]
    print("\n--- Watchdog recovery (simplex task, a hang = module failure) ---")
    print(f"MTBF fixed at the empirical simplex value {mtbf_emp:,.1f} h "
          f"(analytical {1 / FAILURE_RATE:,.0f} h). Restart time = 0.5 h.")
    print(f"Without watchdog: hang cleared by manual repair, MTTR = {REPAIR_HOURS} h, "
          f"A = {mtbf_emp / (mtbf_emp + REPAIR_HOURS):.6f}")
    rng = random.Random(99)
    print(f"{'timeout(h)':>10}{'MTTR emp':>10}{'A formula(emp MTBF)':>21}"
          f"{'A analytical':>14}{'A simulated':>13}")
    for to in (1, 5, 10, 20):
        samples = simulate_watchdog(200, to, 0.5, rng)
        mttr = statistics.mean(samples)
        a_emp = mtbf_emp / (mtbf_emp + mttr)
        a_ana = (1 / FAILURE_RATE) / (1 / FAILURE_RATE + to + 0.5)   # target MTTR = timeout + restart
        a_sim, _ = simulate_simplex_with_watchdog(HOURS, FAILURE_RATE, to, 0.5,
                                                  random.Random(1000 + to))
        print(f"{to:>10}{mttr:>10.3f}{a_emp:>21.6f}{a_ana:>14.6f}{a_sim:>13.6f}")
        rows.append({"experiment": "watchdog timeout sweep", "config": f"timeout {to} h",
                     "failures": "", "mtbf_emp_h": round(mtbf_emp, 1),
                     "mtbf_ana_h": 1 / FAILURE_RATE, "mtbf_dev_pct": "",
                     "mttr_emp_h": round(mttr, 3), "avail_emp": round(a_sim, 6),
                     "avail_ana": round(a_ana, 6)})

    # ---------------- Lab Task 1: diverse modules ----------------
    print("\n--- Lab Task 1: diverse (non-identical) TMR modules, permanent faults ---")
    t1_trials = max(5, TRIALS // 3)
    print(f"{'module rates (/h)':<26}{'TMR MTBF emp':>13}{'TMR MTBF ana':>13}"
          f"{'gain vs simplex@mean':>21}{'gain vs best':>13}")
    for rates in ((0.001, 0.001, 0.001), (0.0005, 0.001, 0.0015),
                  (0.0002, 0.001, 0.0018), (0.0001, 0.0001, 0.0028)):
        res = [run_nmr(HOURS, list(rates), "permanent",
                       [random.randrange(10**9) for _ in range(3)]) for _ in range(t1_trials)]
        P = pooled(res)
        ana = analytic_tmr(list(rates), "permanent")["mtbf"]
        mean_rate = sum(rates) / 3
        print(f"{str(rates):<26}{fmt(P['mtbf']):>13}{fmt(ana):>13}"
              f"{ana * mean_rate:>20.1f}x{ana * min(rates):>12.2f}x")
        rows.append({"experiment": "Task 1 diverse rates", "config": str(rates),
                     "failures": P["failures"], "mtbf_emp_h": round(P["mtbf"], 1),
                     "mtbf_ana_h": round(ana, 1), "mtbf_dev_pct": round(dev(P["mtbf"], ana), 2),
                     "mttr_emp_h": round(P["mttr"], 2), "avail_emp": round(P["availability"], 6),
                     "avail_ana": ""})
    print("  (all rows have the same mean rate 0.001/h, so the same simplex reference)")

    # ---------------- Lab Task 2: voter overhead ----------------
    print("\n--- Lab Task 2: voter overhead (permanent faults) ---")
    S, T, aS, aT = emp["permanent"]
    print(f"Availability without overhead: simplex {S['availability']:.6f}, "
          f"TMR {T['availability']:.6f}")
    v_star = 1 - S["availability"] / T["availability"]
    print(f"Useful-time availability = A_TMR x (1 - v), v = voter time per hour.")
    for v in (0.001, 0.005, 0.01, 0.02, 0.05):
        a = T["availability"] * (1 - v)
        print(f"  v = {v:<6} ({v * 3600:>5.0f} s/h) -> {a:.6f}  "
              f"{'better' if a > S['availability'] else 'WORSE'} than simplex")
    print(f"  Break-even overhead v* = {v_star:.4f} ({v_star * 3600:.0f} s per hour)")
    # bounded vs unbounded voter time against a per-frame slack budget
    rng = random.Random(5)
    slack = 0.01                                  # 36 s of slack in each 1-h frame
    frames = HOURS
    mean_v = 0.002
    bounded = sum(1 for _ in range(frames) if min(rng.expovariate(1 / mean_v), 0.004) > slack)
    unbounded = sum(1 for _ in range(frames) if rng.expovariate(1 / mean_v) > slack)
    print(f"  Per-frame slack {slack * 3600:.0f} s, mean voter time {mean_v * 3600:.1f} s, "
          f"{frames:,} frames:")
    print(f"    bounded voter (WCET capped at {0.004 * 3600:.1f} s): {bounded} deadline overruns")
    print(f"    unbounded voter (exponential tail)       : {unbounded} deadline overruns "
          f"({100 * unbounded / frames:.3f}% of frames)")

    # ---------------- Lab Task 3: false positives ----------------
    print("\n--- Lab Task 3: watchdog true vs false positives ---")
    print(f"{'timeout(h)':>10}{'true pos':>10}{'false pos':>11}{'FP share':>10}"
          f"{'FP per 1000 beats':>19}")
    for to in (0.5, 1.0, 2.0, 5.0):
        tp, fp = watchdog_fp_experiment(to)
        print(f"{to:>10}{tp:>10}{fp:>11}{100 * fp / (tp + fp):>9.1f}%{1000 * fp / 200_000:>19.3f}")
        rows.append({"experiment": "Task 3 watchdog FP", "config": f"timeout {to} h",
                     "failures": tp, "mtbf_emp_h": "", "mtbf_ana_h": "", "mtbf_dev_pct": "",
                     "mttr_emp_h": "", "avail_emp": f"FP={fp}", "avail_ana": ""})

    # ---------------- Lab Task 4: availability vs MTTR ----------------
    print("\n--- Lab Task 4: availability vs MTTR (MTBF fixed at 1000 h) ---")
    print(f"{'MTTR(h)':>8}{'A analytical':>14}{'A simulated':>13}{'downtime h/yr':>15}"
          f"{'gain vs previous':>18}")
    prev = None
    task4 = []
    for m in (48, 24, 12, 6, 3, 1, 0.5, 0.1):
        a = 1000 / (1000 + m)
        a_sim = simulate_simplex_fixed_mttr(HOURS, FAILURE_RATE, m,
                                            random.Random(int(m * 10) + 3))
        gain = "" if prev is None else f"+{(a - prev) * 8760:.1f} h/yr"
        print(f"{m:>8}{a:>14.6f}{a_sim:>13.6f}{(1 - a) * 8760:>15.1f}{gain:>18}")
        task4.append({"mttr_h": m, "avail_analytical": round(a, 6), "avail_sim": round(a_sim, 6),
                      "downtime_h_per_year": round((1 - a) * 8760, 2)})
        prev = a
    with open(os.path.join(OUT_DIR, "mp5_1_availability_vs_mttr.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(task4[0].keys()))
        w.writeheader()
        w.writerows(task4)
    with open(os.path.join(OUT_DIR, "mp5_1_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    plot_task4(task4)
    print("\nSaved: outputs/mp5_1_summary.csv, outputs/mp5_1_availability_vs_mttr.csv")


def plot_task4(task4):
    """Optional plot (matplotlib) of availability against MTTR, MTBF fixed."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plot (CSV contains the data).")
        return
    xs = [r["mttr_h"] for r in task4]
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=200)
    ax.plot(xs, [r["avail_analytical"] for r in task4], color="#2a78d6", lw=2,
            label="Analytical  A = MTBF / (MTBF + MTTR)")
    ax.plot(xs, [r["avail_sim"] for r in task4], ls="none", marker="o", ms=6, color="#eb6834",
            markeredgecolor="white", label="Simulated (200,000 h)")
    ax.set_xscale("log")
    ax.set_xlabel("MTTR (hours, log scale); MTBF fixed at 1000 h", color="#5f5e5a")
    ax.set_ylabel("Availability", color="#5f5e5a")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "mp5_1_availability_vs_mttr.png"))
    plt.close(fig)
    print("Saved plot: outputs/mp5_1_availability_vs_mttr.png")


if __name__ == "__main__":
    main()
