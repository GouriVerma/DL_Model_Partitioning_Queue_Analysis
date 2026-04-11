"""
TP-5: Split-Execution of a Deep Learning Model on a Decentralized Mesh Network
===============================================================================
Grid-based static optimal partition analysis.

Architecture:
  - 10 x 8 grid = 80 devices total
  - S partitions, each gets floor(80/S) parallel devices (M/M/c queue)
  - Tasks arrive at top-left device, flow left -> right through partitions
  - 8 DL layers: 5 Conv + 3 BiLSTM (atomic, sequential)
  - Goal: find optimal grouping of layers into S partitions (S=1..8)

Dependencies: pip install numpy scipy matplotlib simpy
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import combinations
import simpy
import random

# ─────────────────────────────────────────────────────────
# SECTION 1 — LAYER SPECIFICATIONS
# ─────────────────────────────────────────────────────────

LAYERS = [
    {"name": "Conv-1",   "flops_M":   8.0},
    {"name": "Conv-2",   "flops_M":  16.0},
    {"name": "Conv-3",   "flops_M":  32.0},
    {"name": "Conv-4",   "flops_M":  16.0},
    {"name": "Conv-5",   "flops_M":   8.0},
    {"name": "BiLSTM-1", "flops_M": 120.0},
    {"name": "BiLSTM-2", "flops_M": 120.0},
    {"name": "BiLSTM-3", "flops_M":  80.0},
]

# ─────────────────────────────────────────────────────────
# SECTION 2 — SIMULATION PARAMETERS
# ─────────────────────────────────────────────────────────

TOTAL_DEVICES  = 80       # 10 rows x 8 cols grid
COMPUTE_GFLOPS = 0.5      # per-device compute capacity (GFLOPS)
LAMBDA         = 20.0      # default task arrival rate (tasks/sec)
SIM_TIME       = 500.0    # SimPy simulation duration (seconds)
RANDOM_SEED    = 42

def devices_per_partition(S):
    """Equal device allocation: floor(80 / S) per partition."""
    return TOTAL_DEVICES // S

# ─────────────────────────────────────────────────────────
# SECTION 3 — M/M/c ANALYTICAL MODEL (ERLANG-C)
# ─────────────────────────────────────────────────────────

def erlang_c_queue(flops_M, lam, c, compute=COMPUTE_GFLOPS):
    """
    M/M/c queue analysis for one partition stage.

    Parameters
    ----------
    flops_M : float  — total FLOPs (millions) of layers in this partition
    lam     : float  — task arrival rate (tasks/sec)
    c       : int    — number of parallel servers (devices in this partition)
    compute : float  — per-device compute speed (GFLOPS)

    Returns dict with rho, Lq, Wq, E_T, stable, etc.
    """
    flops_G   = flops_M / 1000.0
    mu_single = compute / flops_G      # service rate of one device (tasks/sec)
    mu_total  = c * mu_single          # total partition throughput
    a         = lam / mu_single        # offered load (Erlangs)
    rho       = lam / mu_total         # per-server utilisation

    if rho >= 1.0:
        return {
            "mu_single": mu_single, "mu_total": mu_total, "c": c,
            "rho": rho, "stable": False,
            "C_erlang": 1.0, "Lq": float("inf"),
            "Wq": float("inf"), "E_service": 1.0 / mu_single,
            "E_T": float("inf"),
        }

    # Erlang-C: P(arriving task has to wait)
    try:
        factC = math.factorial(c)
        num   = (a**c / factC) * (c / (c - a))
        dsum  = sum(a**k / math.factorial(k) for k in range(c))
        denom = dsum + num
        C_erl = num / denom
    except (OverflowError, ZeroDivisionError):
        C_erl = 1.0

    Lq  = C_erl * rho / (1 - rho)   # mean queue length
    Wq  = Lq / lam                   # mean wait time
    E_s = 1.0 / mu_single            # mean service time
    E_T = Wq + E_s                   # mean sojourn time

    return {
        "mu_single": mu_single, "mu_total": mu_total, "c": c,
        "rho": rho, "stable": True,
        "C_erlang": C_erl, "Lq": Lq,
        "Wq": Wq, "E_service": E_s, "E_T": E_T,
    }


# ─────────────────────────────────────────────────────────
# SECTION 4 — PARTITION SEARCH
# ─────────────────────────────────────────────────────────

def split_into_S(S, n=8):
    """
    All contiguous groupings of n sequential layers into S groups.
    Returns list of partitions; each partition = list of S groups of layer indices.
    """
    cuts = list(combinations(range(1, n), S - 1))
    partitions = []
    for cut in cuts:
        groups, prev = [], 0
        for c in cut:
            groups.append(list(range(prev, c)))
            prev = c
        groups.append(list(range(prev, n)))
        partitions.append(groups)
    return partitions


def evaluate_partition(groups, lam=LAMBDA):
    """
    Evaluate a specific layer grouping.
    Devices are split equally: c = floor(80 / S) per partition.
    """
    S      = len(groups)
    c      = devices_per_partition(S)   # equal allocation
    total_T  = 0.0
    total_Lq = 0.0
    max_rho  = 0.0
    all_stable = True
    stages = []

    for k, group in enumerate(groups):
        total_flops = sum(LAYERS[i]["flops_M"] for i in group)
        m = erlang_c_queue(total_flops, lam, c)

        total_T  += m["E_T"]
        total_Lq += m["Lq"] if m["stable"] else 1e6
        max_rho   = max(max_rho, m["rho"])
        if not m["stable"]:
            all_stable = False

        stages.append({
            "col":      k + 1,
            "layers":   [LAYERS[i]["name"] for i in group],
            "flops_M":  total_flops,
            "c":        c,
            "rho":      m["rho"],
            "Lq":       m["Lq"],
            "Wq":       m["Wq"],
            "E_T":      m["E_T"],
            "stable":   m["stable"],
            "C_erlang": m.get("C_erlang", 0),
        })

    return {
        "stages":    stages,
        "total_T":  total_T,
        "total_Lq": total_Lq,
        "max_rho":  max_rho,
        "stable":   all_stable,
        "S":        S,
        "c":        c,
    }


def find_optimal_partition(lam=LAMBDA):
    """
    Sweep S = 1..8, find best stable partition for each S,
    then return overall optimal S* minimising E[T].
    """
    sweep = {}
    for S in range(1, 9):
        parts = split_into_S(S)
        best  = None

        for p in parts:
            r = evaluate_partition(p, lam)
            if not r["stable"]:
                continue
            score = r["total_T"] + 0.5 * r["total_Lq"]
            if best is None or score < (best[0]["total_T"] + 0.5 * best[0]["total_Lq"]):
                best = (r, p)

        if best is None:
            lb = min(
                [(evaluate_partition(p, lam), p) for p in parts],
                key=lambda x: x[0]["max_rho"],
            )
            sweep[S] = {"result": lb[0], "partition": lb[1], "feasible": False}
        else:
            sweep[S] = {"result": best[0], "partition": best[1], "feasible": True}

    stable = [(S, v) for S, v in sweep.items() if v["feasible"]]
    if stable:
        S_star = min(stable, key=lambda x: x[1]["result"]["total_T"])[0]
    else:
        S_star = min(sweep, key=lambda S: sweep[S]["result"]["max_rho"])

    return sweep, S_star


# ─────────────────────────────────────────────────────────
# SECTION 5 — SIMPY DISCRETE-EVENT SIMULATION
# ─────────────────────────────────────────────────────────

class Stats:
    def __init__(self, S):
        self.completion_times      = []
        self.stage_waits           = [[] for _ in range(S)]
        self.stage_queue_lengths   = [[] for _ in range(S)]
        self.n_completed           = 0


def run_simpy(groups, lam=LAMBDA, sim_time=SIM_TIME, seed=RANDOM_SEED):
    """
    Discrete-event simulation of the S-stage M/M/c pipeline.
    Each partition is a SimPy Resource with capacity = floor(80/S).
    Poisson arrivals, exponential service times.
    """
    random.seed(seed)
    np.random.seed(seed)

    env   = simpy.Environment()
    S     = len(groups)
    c     = devices_per_partition(S)
    stats = Stats(S)

    resources  = [simpy.Resource(env, capacity=c) for _ in range(S)]
    mu_singles = []
    for group in groups:
        fm = sum(LAYERS[i]["flops_M"] for i in group)
        mu_singles.append(COMPUTE_GFLOPS / (fm / 1000.0))

    def task(env, task_id):
        arrival = env.now
        for k in range(S):
            stats.stage_queue_lengths[k].append(len(resources[k].queue))
            t_req = env.now
            with resources[k].request() as req:
                yield req
                stats.stage_waits[k].append(env.now - t_req)
                svc = np.random.exponential(1.0 / mu_singles[k])
                yield env.timeout(svc)
        stats.completion_times.append(env.now - arrival)
        stats.n_completed += 1

    def arrivals(env):
        task_id = 0
        while True:
            yield env.timeout(np.random.exponential(1.0 / lam))
            env.process(task(env, task_id))
            task_id += 1

    env.process(arrivals(env))
    env.run(until=sim_time)
    return stats


# ─────────────────────────────────────────────────────────
# SECTION 6 — PLOTTING
# ─────────────────────────────────────────────────────────

def plot_sweep(sweep, S_star, lam, save=True):
    """Plot 1: E[T] and max rho vs S."""
    S_vals   = list(range(1, 9))
    T_vals   = [sweep[S]["result"]["total_T"]  for S in S_vals]
    rho_vals = [sweep[S]["result"]["max_rho"]  for S in S_vals]
    stable   = [sweep[S]["feasible"]           for S in S_vals]
    c_vals   = [devices_per_partition(S)       for S in S_vals]

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twinx()

    colors = ["#185FA5" if s else "#E24B4A" for s in stable]
    ax1.bar(S_vals, T_vals, color=colors, alpha=0.75)
    ax2.plot(S_vals, rho_vals, "o-", color="#854F0B", lw=1.5, label="max ρ")
    ax2.axhline(1.0, color="#E24B4A", lw=0.8, ls="--", alpha=0.6)

    ax1.axvline(S_star, color="#185FA5", lw=1.5, ls=":", alpha=0.8)
    ax1.text(S_star + 0.1, max(T_vals) * 0.92, f"S* = {S_star}", color="#185FA5", fontsize=9)

    # Annotate devices per partition
    for S in S_vals:
        ax1.text(S, -0.015 * max(T_vals), f"c={c_vals[S-1]}",
                 ha="center", va="top", fontsize=7, color="#5F5E5A")

    ax1.set_xlabel("Number of partitions S  (c = devices per partition)")
    ax1.set_ylabel("Mean task completion time E[T] (s)")
    ax2.set_ylabel("Max server utilisation ρ")
    ax2.set_ylim(0, 1.3)
    ax1.set_xticks(S_vals)

    p1 = mpatches.Patch(color="#185FA5", alpha=0.75, label="E[T] – stable")
    p2 = mpatches.Patch(color="#E24B4A", alpha=0.75, label="E[T] – unstable")
    p3 = plt.Line2D([0], [0], color="#854F0B", marker="o", label="max ρ")
    ax1.legend(handles=[p1, p2, p3], fontsize=8, loc="upper left")

    plt.title(
        f"Partition sweep  (λ={lam} tasks/s, {TOTAL_DEVICES} total devices, "
        f"equal allocation: c=80/S)"
    )
    plt.tight_layout()
    if save:
        plt.savefig("plot_sweep.png", dpi=150, bbox_inches="tight")
        print("[saved] plot_sweep.png")
    plt.show()


def plot_per_stage(result_star, S_star, save=True):
    """Plot 2: Per-stage rho and Lq for optimal partition."""
    stages = result_star["stages"]
    c      = result_star["c"]
    labels = [f"P{s['col']}\n" + "\n".join(s["layers"]) for s in stages]
    rho_v  = [s["rho"] for s in stages]
    lq_v   = [min(s["Lq"], 50) for s in stages]

    col_colors = [
        "#85B7EB" if all(l.startswith("Conv") for l in s["layers"])
        else "#AFA9EC" if all(l.startswith("Bi") for l in s["layers"])
        else "#5DCAA5"
        for s in stages
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    x = range(len(stages))

    ax1.bar(x, rho_v, color=col_colors, edgecolor="#fff", lw=0.5)
    ax1.axhline(1.0,  color="#E24B4A", lw=0.8, ls="--", label="unstable (ρ=1)")
    ax1.axhline(0.85, color="#EF9F27", lw=0.8, ls=":",  label="warning (ρ=0.85)")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=7)
    ax1.set_ylabel("Server utilisation ρ"); ax1.set_ylim(0, 1.1)
    ax1.set_title(f"Per-stage utilisation  (S*={S_star}, c={c} devices each)")
    ax1.legend(fontsize=8)

    ax2.bar(x, lq_v, color=col_colors, edgecolor="#fff", lw=0.5)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_ylabel("Mean queue length Lq")
    ax2.set_title("Per-stage queue length Lq")

    p1 = mpatches.Patch(color="#85B7EB", label="Conv layers")
    p2 = mpatches.Patch(color="#AFA9EC", label="BiLSTM layers")
    fig.legend(handles=[p1, p2], fontsize=8, loc="lower center", ncol=2)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    if save:
        plt.savefig("plot_stages.png", dpi=150, bbox_inches="tight")
        print("[saved] plot_stages.png")
    plt.show()


def plot_simpy_vs_theory(theory, sim_stats, S_star, save=True):
    """Plot 3: Theory vs SimPy comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ct = sim_stats.completion_times
    ax1.hist(ct, bins=40, color="#85B7EB", edgecolor="#185FA5",
             lw=0.4, density=True, alpha=0.8)
    ax1.axvline(np.mean(ct), color="#185FA5", lw=1.5, ls="--",
                label=f"Sim mean = {np.mean(ct):.4f} s")
    ax1.axvline(theory["total_T"], color="#854F0B", lw=1.5, ls=":",
                label=f"Theory  = {theory['total_T']:.4f} s")
    ax1.set_xlabel("Task completion time (s)")
    ax1.set_ylabel("Density")
    ax1.set_title(f"Completion time distribution  (S*={S_star})")
    ax1.legend(fontsize=8)

    stages = theory["stages"]
    x      = range(len(stages))
    th_wq  = [s["Wq"] for s in stages]
    sim_wq = [np.mean(w) if w else 0 for w in sim_stats.stage_waits]
    w = 0.35
    ax2.bar([i - w/2 for i in x], th_wq,  width=w,
            color="#85B7EB", label="Theory Wq",  alpha=0.85)
    ax2.bar([i + w/2 for i in x], sim_wq, width=w,
            color="#854F0B", label="SimPy Wq",   alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"P{s['col']}" for s in stages], fontsize=8)
    ax2.set_ylabel("Mean wait time Wq (s)")
    ax2.set_title("Theory vs SimPy: queue wait per partition")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    if save:
        plt.savefig("plot_theory_vs_sim.png", dpi=150, bbox_inches="tight")
        print("[saved] plot_theory_vs_sim.png")
    plt.show()


def plot_lambda_sensitivity(save=True):
    """Plot 4: Optimal E[T] and S* as lambda varies."""
    lam_range = list(range(2, 201, 5))
    T_opt, rho_opt, s_opt, feasible_list = [], [], [], []

    for lam in lam_range:
        sw, ss = find_optimal_partition(lam)
        r = sw[ss]["result"]
        T_opt.append(r["total_T"] if sw[ss]["feasible"] else None)
        rho_opt.append(r["max_rho"])
        s_opt.append(ss)
        feasible_list.append(sw[ss]["feasible"])

    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(11, 4))
    ax2 = ax1.twinx()

    T_plot = [t if t is not None else float("nan") for t in T_opt]
    ax1.plot(lam_range, T_plot,  "-o", color="#185FA5", ms=3, lw=1.5, label="E[T] (s)")
    ax2.plot(lam_range, rho_opt, "-", color="#854F0B",  ms=3, lw=1.2, ls="--", label="max ρ")
    ax2.axhline(1.0, color="#E24B4A", lw=0.8, ls=":")

    for i, (lam, f) in enumerate(zip(lam_range, feasible_list)):
        if not f:
            ax1.axvspan(lam - 2.5, lam + 2.5, color="#FCEBEB", alpha=0.4)

    ax1.set_xlabel("Task arrival rate λ (tasks/s)")
    ax1.set_ylabel("Optimal E[T] (s)")
    ax2.set_ylabel("Max ρ")
    ax1.set_title("λ sensitivity: optimal E[T] and ρ")
    ax1.legend(loc="upper left",  fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    ax3.step(lam_range, s_opt, color="#185FA5", lw=1.5, where="mid")
    ax3.fill_between(lam_range, s_opt, step="mid", alpha=0.15, color="#185FA5")
    ax3.set_xlabel("Task arrival rate λ (tasks/s)")
    ax3.set_ylabel("Optimal S*")
    ax3.set_yticks(range(1, 9))
    ax3.set_title("Optimal partition count S* vs λ")

    plt.tight_layout()
    if save:
        plt.savefig("plot_lambda.png", dpi=150, bbox_inches="tight")
        print("[saved] plot_lambda.png")
    plt.show()


def plot_grid_map(sweep, S_star, lam, save=True):
    """Plot 5: Device grid coloured by partition assignment."""
    result = sweep[S_star]["result"]
    groups = sweep[S_star]["partition"]
    S      = S_star
    c      = devices_per_partition(S)   # devices per partition
    rows   = TOTAL_DEVICES // S         # = c
    cols   = S

    conv_cmap   = plt.cm.Blues
    bilstm_cmap = plt.cm.Purples

    fig, ax = plt.subplots(figsize=(min(14, 2 * S + 2), 5))
    cw, rh = 1.0, 0.75

    group_colors = []
    for g_idx, group in enumerate(groups):
        is_bi = all(LAYERS[i]["name"].startswith("Bi") for i in group)
        cmap  = bilstm_cmap if is_bi else conv_cmap
        group_colors.append(cmap(0.35 + 0.4 * (g_idx / max(S - 1, 1))))

    for g_idx, group in enumerate(groups):
        col_c = group_colors[g_idx]
        rho   = result["stages"][g_idx]["rho"]
        for row in range(c):
            rect = mpatches.FancyBboxPatch(
                (g_idx * cw + 0.05, row * rh + 0.05),
                cw - 0.1, rh - 0.1,
                boxstyle="round,pad=0.02", lw=0.5,
                facecolor=col_c, edgecolor="white"
            )
            ax.add_patch(rect)
            dev_id = g_idx * c + row
            ax.text(g_idx * cw + cw / 2, row * rh + rh / 2,
                    f"D{dev_id}", ha="center", va="center",
                    fontsize=max(5, 7 - S // 3), color="#222")

        # Partition label top
        layer_str = "+".join(LAYERS[i]["name"] for i in group)
        ax.text(g_idx * cw + cw / 2, c * rh + 0.12,
                layer_str, ha="center", va="bottom",
                fontsize=max(6, 8 - S // 3), fontweight="bold")
        # rho label bottom
        ax.text(g_idx * cw + cw / 2, -0.18,
                f"ρ={rho:.3f}  c={c}",
                ha="center", va="top", fontsize=7,
                color="#854F0B" if rho > 0.7 else "#185FA5")

    ax.annotate("Tasks\nenter →", xy=(0, c * rh / 2),
                xytext=(-0.75, c * rh / 2),
                arrowprops=dict(arrowstyle="->", color="#185FA5", lw=1.5),
                fontsize=8, ha="center", va="center", color="#185FA5")

    ax.set_xlim(-1.1, S * cw + 0.2)
    ax.set_ylim(-0.5, c * rh + 0.9)
    ax.axis("off")
    ax.set_title(
        f"Device grid — S*={S_star}, {c} devices per partition  "
        f"(λ={lam} tasks/s)\n"
        f"Blue=Conv partitions   Purple=BiLSTM partitions",
        fontsize=10
    )
    plt.tight_layout()
    if save:
        plt.savefig("plot_grid.png", dpi=150, bbox_inches="tight")
        print("[saved] plot_grid.png")
    plt.show()


# ─────────────────────────────────────────────────────────
# SECTION 7 — MAIN RUNNER
# ─────────────────────────────────────────────────────────

def print_results(sweep, S_star, lam):
    print("\n" + "=" * 72)
    print(f"TP-5 RESULTS  (λ={lam}, {TOTAL_DEVICES} devices, equal allocation c=80/S)")
    print("=" * 72)
    for S in range(1, 9):
        d = sweep[S]
        r = d["result"]
        c = devices_per_partition(S)
        tag = " ← S*" if S == S_star else ""
        ok  = "STABLE" if d["feasible"] else "UNSTABLE"
        print(f"\nS={S}  c={c} devices/partition  [{ok}]{tag}")
        print(f"  E[T]={r['total_T']:.4f}s  Lq={r['total_Lq']:.4f}  max ρ={r['max_rho']:.4f}")
        for st in r["stages"]:
            lyrs = "+".join(st["layers"])
            print(f"    P{st['col']}: {lyrs:40s} ρ={st['rho']:.4f}  Lq={st['Lq']:.5f}  E[T]={st['E_T']:.5f}s")

    print("\n" + "=" * 72)
    print(f"OPTIMAL  S* = {S_star}  (c = {devices_per_partition(S_star)} devices per partition)")
    r = sweep[S_star]["result"]
    print(f"  E[T]     = {r['total_T']:.5f} s")
    print(f"  Total Lq = {r['total_Lq']:.5f}")
    print(f"  Max ρ    = {r['max_rho']:.5f}")
    print("\nDevice Mapping:")
    c = devices_per_partition(S_star)
    for st in r["stages"]:
        devs = f"{(st['col']-1)*c}–{st['col']*c - 1}"
        lyrs = " + ".join(st["layers"])
        print(f"  P{st['col']} (devices {devs}): {lyrs}")
        print(f"     FLOPs={st['flops_M']:.0f}M  ρ={st['rho']:.5f}  E[T]={st['E_T']:.5f}s")
    print()


def main():
    lam = LAMBDA  # change this to explore other arrival rates

    print("Running theoretical M/M/c analysis (equal device allocation)...")
    sweep, S_star = find_optimal_partition(lam)
    print_results(sweep, S_star, lam)

    result_star    = sweep[S_star]["result"]
    partition_star = sweep[S_star]["partition"]

    print(f"Running SimPy simulation (S*={S_star}, "
          f"c={devices_per_partition(S_star)} devices/partition, T={SIM_TIME}s)...")
    sim_stats = run_simpy(partition_star, lam=lam, sim_time=SIM_TIME)
    print(f"  Tasks completed : {sim_stats.n_completed}")
    print(f"  Sim  mean E[T]  : {np.mean(sim_stats.completion_times):.5f} s")
    print(f"  Theory E[T]     : {result_star['total_T']:.5f} s")
    err = (abs(np.mean(sim_stats.completion_times) - result_star["total_T"])
           / result_star["total_T"] * 100)
    print(f"  Error           : {err:.2f}%")

    print("\nGenerating plots...")
    plot_sweep(sweep, S_star, lam)
    plot_per_stage(result_star, S_star)
    plot_simpy_vs_theory(result_star, sim_stats, S_star)
    plot_lambda_sensitivity()
    plot_grid_map(sweep, S_star, lam)
    print("\nDone. All plots saved as PNG files.")


if __name__ == "__main__":
    main()