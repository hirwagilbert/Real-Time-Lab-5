"""
Mini Project 5.3 - Checkpoint and Rollback-Recovery Simulator for a
Long-Running Real-Time Task                     (Real-Time Systems, Unit 5)

Model (all times in minutes = simulated time units):
  * the task needs WORK units of computation and must finish by DEADLINE;
  * every CHECKPOINT_INTERVAL units of progress it pauses CKPT_OVERHEAD units
    to save its state (Section 5.1: a bounded, budgeted redundancy cost);
  * each time unit a fault strikes with probability p = 1/MTBF (Section 5.2),
    whether the task is working, checkpointing or recovering;
  * on a fault the task rolls back to the last COMPLETED checkpoint (or the
    start) and spends RECOVERY_OVERHEAD units restoring state; a fault during
    recovery restarts the recovery.

run_with_checkpointing(..., mode="step") is the literal time-unit-by-time-unit
simulator; mode="fast" jumps straight to the next fault with a geometric draw,
which is statistically identical and much faster. validate() compares them.

Outputs (../outputs/): mp5_3_interval_sweep.csv, mp5_3_optimum_summary.csv and,
if matplotlib is installed, mp5_3_completion_vs_interval.png.
Python 3.8+ standard library only.  Run: python checkpoint_sim.py  (about 5-15 s)
"""

import csv
import math
import os
import random
import statistics
import sys

SEED = 515
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
QUICK = "--quick" in sys.argv

# ---------------- documented parameters (hypothetical mission) ----------------
WORK = 1000                 # minutes of computation the task needs
DEADLINE = 1300             # mission deadline (minutes)
CKPT_OVERHEAD = 5           # minutes to write one checkpoint
RECOVERY_OVERHEAD = 10      # minutes to reload state after a fault
MTBF_VALUES = [250, 500, 1000, 2000]          # assumed MTBF (minutes)
INTERVALS = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100, 125, 150,
             200, 250, 300, 400, 500, 1000]   # 1000 = no checkpoint at all
TRIALS = 300 if QUICK else 3000


# ---------------------------------------------------------------------------
# Simulator (Suggested approach steps 1-3)
# ---------------------------------------------------------------------------
def run_with_checkpointing(checkpoint_interval, mtbf, rng, work=WORK,
                           ckpt=CKPT_OVERHEAD, recovery=RECOVERY_OVERHEAD, mode="fast"):
    """Return (total wall-clock time, faults, checkpoints written, rework units)."""
    p = 1.0 / mtbf
    if mode == "step":
        return _run_step(checkpoint_interval, p, rng, work, ckpt, recovery)
    return _run_fast(checkpoint_interval, p, rng, work, ckpt, recovery)


def _run_step(interval, p, rng, work, ckpt, recovery):
    t = progress = last_cp = faults = n_ckpt = rework = 0
    state, remaining = "work", 0
    while progress < work:
        t += 1
        fault = rng.random() < p
        if state == "work":
            if fault:
                rework += progress - last_cp
                progress, state, remaining = last_cp, "recover", recovery
                faults += 1
                continue
            progress += 1
            if progress < work and progress - last_cp >= interval:
                state, remaining = "ckpt", ckpt
        elif state == "ckpt":
            if fault:                                  # checkpoint not completed
                rework += progress - last_cp
                progress, state, remaining = last_cp, "recover", recovery
                faults += 1
                continue
            remaining -= 1
            if remaining == 0:
                last_cp, state = progress, "work"
                n_ckpt += 1
        else:                                          # recovering
            if fault:
                faults += 1
                remaining = recovery                   # recovery starts again
                continue
            remaining -= 1
            if remaining == 0:
                state = "work"
    return t, faults, n_ckpt, rework


def _run_fast(interval, p, rng, work, ckpt, recovery):
    log1m = math.log1p(-p)

    def clean():                                   # fault-free units before next fault
        return int(math.log(1.0 - rng.random()) / log1m)

    t = progress = last_cp = faults = n_ckpt = rework = 0
    nf = clean()

    def recover():
        nonlocal t, nf, faults
        while True:
            if nf >= recovery:
                t += recovery
                nf -= recovery
                return
            t += nf + 1                            # fault hits during recovery
            faults += 1
            nf = clean()

    while progress < work:
        seg = min(interval - (progress - last_cp), work - progress)
        if nf >= seg:                              # work segment completes
            t += seg
            nf -= seg
            progress += seg
        else:                                      # fault during work
            t += nf + 1
            rework += progress + nf - last_cp
            progress = last_cp
            faults += 1
            nf = clean()
            recover()
            continue
        if progress >= work:
            break
        if nf >= ckpt:                             # checkpoint completes
            t += ckpt
            nf -= ckpt
            last_cp = progress
            n_ckpt += 1
        else:                                      # fault while checkpointing
            t += nf + 1
            rework += progress - last_cp
            progress = last_cp
            faults += 1
            nf = clean()
            recover()
    return t, faults, n_ckpt, rework


# ---------------------------------------------------------------------------
# Analytical models (Suggested approach step 6)
# ---------------------------------------------------------------------------
def first_order_time(tau, mtbf, work=WORK, c=CKPT_OVERHEAD, r=RECOVERY_OVERHEAD):
    """T = W + (W/tau)*C  [checkpoint overhead]  + (T/MTBF)*(tau/2 + R)  [rework].
    Solving for T:  T = W (1 + C/tau) / (1 - (tau/2 + R)/MTBF)."""
    denom = 1 - (tau / 2 + r) / mtbf
    return work * (1 + c / tau) / denom if denom > 0 else math.inf


def young_optimum(mtbf, c=CKPT_OVERHEAD):
    """d/dtau of W*C/tau + W*tau/(2*MTBF) = 0  ->  tau* = sqrt(2 * C * MTBF)."""
    return math.sqrt(2 * c * mtbf)


def renewal_time(tau, mtbf, work=WORK, c=CKPT_OVERHEAD, r=RECOVERY_OVERHEAD):
    """Exact expected time with exponential faults: a segment of length L that
    must run fault-free, with restart cost R also exposed to faults, takes
    E = MTBF * e^(R/MTBF) * (e^(L/MTBF) - 1). Segments: all full ones carry a
    checkpoint (tau + C); the last one does not."""
    k = math.ceil(work / tau)
    last = work - (k - 1) * tau
    seg = lambda L: mtbf * math.exp(r / mtbf) * math.expm1(L / mtbf)
    return (k - 1) * seg(tau + c) + seg(last)


def numeric_min(fn, mtbf):
    taus = [x / 2 for x in range(4, 2001)]        # 2 .. 1000 in steps of 0.5
    vals = [(fn(t, mtbf), t) for t in taus]
    return min(vals)[1]


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def validate():
    rng1, rng2 = random.Random(1), random.Random(2)
    a = [run_with_checkpointing(50, 500, rng1, mode="step")[0] for _ in range(1500)]
    b = [run_with_checkpointing(50, 500, rng2, mode="fast")[0] for _ in range(1500)]
    print(f"Validation (interval 50, MTBF 500, 1500 runs each): mean completion "
          f"step-by-step {statistics.mean(a):.1f}, fast {statistics.mean(b):.1f}, "
          f"renewal formula {renewal_time(50, 500):.1f}")


def sweep():
    rng = random.Random(SEED)
    rows, summary = [], []
    for mtbf in MTBF_VALUES:
        print(f"\n--- Assumed MTBF = {mtbf} min (fault probability {1 / mtbf:.4f} per minute) ---")
        print(f"{'interval':>8}{'mean T':>9}{'95% CI':>17}{'sd':>8}{'faults':>8}{'ckpts':>7}"
              f"{'rework':>8}{'P(T<=D)':>9}{'1st-order':>10}{'renewal':>9}")
        best = None
        for tau in INTERVALS:
            res = [run_with_checkpointing(tau, mtbf, rng) for _ in range(TRIALS)]
            times = [r[0] for r in res]
            mean = statistics.mean(times)
            sd = statistics.stdev(times)
            half = 1.96 * sd / math.sqrt(TRIALS)
            p_dead = sum(1 for x in times if x <= DEADLINE) / TRIALS
            fo, rn = first_order_time(tau, mtbf), renewal_time(tau, mtbf)
            row = {"mtbf": mtbf, "interval": tau, "trials": TRIALS,
                   "mean_completion": round(mean, 2), "ci95_low": round(mean - half, 2),
                   "ci95_high": round(mean + half, 2), "sd": round(sd, 2),
                   "p95_completion": sorted(times)[int(0.95 * TRIALS) - 1],
                   "mean_faults": round(statistics.mean(r[1] for r in res), 3),
                   "mean_checkpoints": round(statistics.mean(r[2] for r in res), 2),
                   "mean_rework": round(statistics.mean(r[3] for r in res), 2),
                   "p_meet_deadline": round(p_dead, 4),
                   "first_order_model": round(fo, 2) if math.isfinite(fo) else "inf",
                   "renewal_model": round(rn, 2)}
            rows.append(row)
            if best is None or mean < best[0]:
                best = (mean, tau, half)
            print(f"{tau:>8}{mean:>9.1f}  [{mean - half:>6.1f},{mean + half:>7.1f}]{sd:>8.1f}"
                  f"{row['mean_faults']:>8.2f}{row['mean_checkpoints']:>7.1f}"
                  f"{row['mean_rework']:>8.1f}{p_dead:>9.3f}"
                  f"{(f'{fo:.1f}' if math.isfinite(fo) else 'inf'):>10}{rn:>9.1f}")
        # intervals statistically indistinguishable from the best (overlapping CIs)
        near = [r["interval"] for r in rows if r["mtbf"] == mtbf
                and r["ci95_low"] <= best[0] + best[2]]
        y, fo_opt, rn_opt = young_optimum(mtbf), numeric_min(first_order_time, mtbf), \
            numeric_min(renewal_time, mtbf)
        print(f"Empirical optimum: interval {best[1]} (mean {best[0]:.1f} min); "
              f"statistically tied: {near}")
        print(f"Analytical optimum: Young sqrt(2*C*MTBF) = {y:.1f};  first-order model minimum = "
              f"{fo_opt:.1f};  renewal model minimum = {rn_opt:.1f}")
        summary.append({"mtbf": mtbf, "empirical_opt": best[1],
                        "empirical_opt_mean": round(best[0], 1),
                        "tied_range": f"{min(near)}-{max(near)}",
                        "young_sqrt_2CM": round(y, 1), "first_order_opt": fo_opt,
                        "renewal_opt": rn_opt})
    return rows, summary


def memo(rows):
    """Recommendation for the hypothetical mission at MTBF = 500 min."""
    m = 500
    pts = [r for r in rows if r["mtbf"] == m]
    best_mean = min(pts, key=lambda r: r["mean_completion"])
    best_dead = max(pts, key=lambda r: (r["p_meet_deadline"], -r["mean_completion"]))
    none = [r for r in pts if r["interval"] == max(INTERVALS)][0]
    print(f"\n--- Recommendation (mission: {WORK} min of work, deadline {DEADLINE} min, "
          f"MTBF {m} min, C = {CKPT_OVERHEAD}, R = {RECOVERY_OVERHEAD}) ---")
    print(f"  No checkpointing        : mean {none['mean_completion']:.0f} min, "
          f"P(meet deadline) = {none['p_meet_deadline']:.3f}")
    print(f"  Minimum mean time       : interval {best_mean['interval']} -> "
          f"mean {best_mean['mean_completion']:.0f} min, 95th pct {best_mean['p95_completion']} min, "
          f"P(meet deadline) = {best_mean['p_meet_deadline']:.3f}")
    print(f"  Best deadline probability: interval {best_dead['interval']} -> "
          f"mean {best_dead['mean_completion']:.0f} min, 95th pct {best_dead['p95_completion']} min, "
          f"P(meet deadline) = {best_dead['p_meet_deadline']:.3f}")


def plot(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plot (CSV contains all data).")
        return
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    marks = ["o", "s", "^", "D"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
    for c, mk, mtbf in zip(colors, marks, MTBF_VALUES):
        pts = [r for r in rows if r["mtbf"] == mtbf]
        x = [r["interval"] for r in pts]
        ax.plot(x, [r["mean_completion"] for r in pts], color=c, lw=2, marker=mk, ms=5,
                markeredgecolor="white", label=f"MTBF {mtbf} simulated")
        ax.plot(x, [r["renewal_model"] for r in pts], color=c, lw=1, ls="--", alpha=0.8)
    ax.axhline(DEADLINE, color="#7a7974", lw=1, ls=":")
    ax.text(5.5, DEADLINE + 15, f"deadline {DEADLINE} min", color="#5f5e5a", fontsize=8)
    ax.set_xscale("log")
    ax.set_ylim(1000, 2200)                 # zoom on the optimum region
    ax.set_xlabel("Checkpoint interval (minutes of work, log scale)", color="#5f5e5a")
    ax.set_ylabel("Mean completion time (min)", color="#5f5e5a")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, title="dashed = renewal model (curves above 2200 cut off)",
              title_fontsize=8, loc="upper center")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "mp5_3_completion_vs_interval.png"))
    plt.close(fig)
    print("Saved plot: outputs/mp5_3_completion_vs_interval.png")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 96)
    print("MINI PROJECT 5.3 - CHECKPOINT AND ROLLBACK-RECOVERY SIMULATOR")
    print("=" * 96)
    print(f"Work {WORK} min, deadline {DEADLINE} min, checkpoint overhead C = {CKPT_OVERHEAD} min, "
          f"recovery R = {RECOVERY_OVERHEAD} min, {TRIALS} trials per point, seed {SEED}")
    validate()
    rows, summary = sweep()
    for name, data in (("mp5_3_interval_sweep.csv", rows), ("mp5_3_optimum_summary.csv", summary)):
        with open(os.path.join(OUT_DIR, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
    print("\n--- Optimal interval vs MTBF ---")
    print(f"{'MTBF':>6}{'empirical':>11}{'tied range':>12}{'Young':>8}{'1st-order':>11}{'renewal':>9}")
    for s in summary:
        print(f"{s['mtbf']:>6}{s['empirical_opt']:>11}{s['tied_range']:>12}"
              f"{s['young_sqrt_2CM']:>8}{s['first_order_opt']:>11}{s['renewal_opt']:>9}")
    memo(rows)
    plot(rows)
    print("Saved: outputs/mp5_3_interval_sweep.csv, outputs/mp5_3_optimum_summary.csv")


if __name__ == "__main__":
    main()
