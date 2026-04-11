"""
TP-5: Split-Execution of a Deep Learning Model on a Decentralized Mesh Network
===============================================================================
Grid-based static optimal partition analysis.

Architecture:
  - 10 x 8 grid = 80 devices total
  - 8 columns, each with 10 parallel devices (M/M/10 queue)
  - Tasks arrive at top-left device (device 0), flow left -> right
  - 8 DL layers: 5 Conv + 3 BiLSTM (atomic, sequential)
  - Goal: find optimal grouping of layers into S columns (S=1..8)

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
    {"name": "Conv-1",   "flops_M": 8.0,   "act_KB": 64.0},
    {"name": "Conv-2",   "flops_M": 16.0,  "act_KB": 32.0},
    {"name": "Conv-3",   "flops_M": 32.0,  "act_KB": 16.0},
    {"name": "Conv-4",   "flops_M": 16.0,  "act_KB":  8.0},
    {"name": "Conv-5",   "flops_M":  8.0,  "act_KB":  4.0},
    {"name": "BiLSTM-1", "flops_M": 120.0, "act_KB": 16.0},
    {"name": "BiLSTM-2", "flops_M": 120.0, "act_KB":  8.0},
    {"name": "BiLSTM-3", "flops_M":  80.0, "act_KB":  4.0},
]

# ─────────────────────────────────────────────────────────
# SECTION 2 — SIMULATION PARAMETERS
# ─────────────────────────────────────────────────────────

GRID_ROWS       = 10        # devices per column (parallel servers per stage)
GRID_COLS       = 8         # columns = max partitions
COMPUTE_GFLOPS  = 0.5       # per-device compute capacity (GFLOPS)
BANDWIDTH_MBps  = 100.0     # inter-column link bandwidth
LAMBDA          = 20       # default task arrival rate (tasks/sec)
SIM_TIME        = 500.0     # SimPy simulation duration (seconds)
RANDOM_SEED     = 42

# ─────────────────────────────────────────────────────────
# SECTION 3 — M/M/c ANALYTICAL MODEL (ERLANG-C)
# ─────────────────────────────────────────────────────────

def erlang_c_queue(flops_M, lam, c=GRID_ROWS, compute=COMPUTE_GFLOPS):
    """
    M/M/c queue analysis for one column stage.

    Parameters
    ----------
    flops_M : float   — total FLOPs (millions) of layers in this column
    lam     : float   — task arrival rate (tasks/sec)
    c       : int     — number of parallel servers (devices per column)
    compute : float   — per-device compute speed (GFLOPS)

    Returns dict with:
        mu_single  : service rate of one device (tasks/sec)
        mu_total   : total service rate of column
        rho        : per-server utilisation (must be < 1 for stability)
        stable     : bool
        C_erlang   : P(task must wait), Erlang-C probability
        Lq         : mean number of tasks waiting in queue
        Wq         : mean waiting time in queue (sec)
        E_service  : mean service time (sec)
        E_T        : mean sojourn time = Wq + E_service (sec)
    """
    flops_G   = flops_M / 1000.0
    mu_single = compute / flops_G      # tasks a single device handles per sec
    mu_total  = c * mu_single          # total column throughput
    a         = lam / mu_single        # offered load (Erlangs)
    rho       = lam / mu_total         # per-server utilisation

    if rho >= 1.0:
        return {
            "mu_single": mu_single, "mu_total": mu_total,
            "rho": rho, "stable": False,
            "C_erlang": 1.0, "Lq": float("inf"),
            "Wq": float("inf"), "E_service": 1.0 / mu_single,
            "E_T": float("inf"),
        }

    # Erlang-C formula
    try:
        factC = math.factorial(c)
        num   = (a**c / factC) * (c / (c - a))
        dsum  = sum(a**k / math.factorial(k) for k in range(c))
        denom = dsum + num
        C_erl = num / denom
    except (OverflowError, ZeroDivisionError):
        C_erl = 1.0

    Lq    = C_erl * rho / (1 - rho)
    Wq    = Lq / lam
    E_s   = 1.0 / mu_single
    E_T   = Wq + E_s

    return {
        "mu_single": mu_single, "mu_total": mu_total,
        "rho": rho, "stable": True,
        "C_erlang": C_erl, "Lq": Lq,
        "Wq": Wq, "E_service": E_s, "E_T": E_T,
    }


def comm_delay_sec(act_KB, bw=BANDWIDTH_MBps):
    """Transmission delay for inter-column activation transfer (seconds)."""
    return (act_KB / 1024.0) / bw


# ─────────────────────────────────────────────────────────
# SECTION 4 — PARTITION SEARCH
# ─────────────────────────────────────────────────────────

def split_into_S(S, n=8):
    """
    Generate all contiguous groupings of n sequential layers into S groups.
    Returns list of partitions; each partition is a list of S groups,
    where each group is a list of layer indices.
    """
    cuts = list(combinations(range(1, n), S - 1))
    partitions = []
    for cut in cuts:
        groups = []
        prev = 0
        for c in cut:
            groups.append(list(range(prev, c)))
            prev = c
        groups.append(list(range(prev, n)))
        partitions.append(groups)
    return partitions


def evaluate_partition(groups, lam=LAMBDA):
    """
    Evaluate a specific grouping of layers across columns.
    Returns metrics dict including per-stage breakdown.
    """
    total_T  = 0.0
    total_Lq = 0.0
    max_rho  = 0.0
    all_stable = True
    stages = []

    for k, group in enumerate(groups):
        total_flops = sum(LAYERS[i]["flops_M"] for i in group)
        last_act_KB = LAYERS[group[-1]]["act_KB"]
        m = erlang_c_queue(total_flops, lam)

        total_T  += m["E_T"]
        total_Lq += m["Lq"] if m["stable"] else 1e6
        max_rho   = max(max_rho, m["rho"])
        if not m["stable"]:
            all_stable = False

        comm = comm_delay_sec(last_act_KB) if k < len(groups) - 1 else 0.0
        total_T += comm

        stages.append({
            "col":        k + 1,
            "layers":     [LAYERS[i]["name"] for i in group],
            "flops_M":    total_flops,
            "rho":        m["rho"],
            "Lq":         m["Lq"],
            "Wq":         m["Wq"],
            "E_T":        m["E_T"],
            "comm_sec":   comm,
            "stable":     m["stable"],
            "C_erlang":   m.get("C_erlang", 0),
        })

    return {
        "stages":    stages,
        "total_T":  total_T,
        "total_Lq": total_Lq,
        "max_rho":  max_rho,
        "stable":   all_stable,
        "S":        len(groups),
    }


def find_optimal_partition(lam=LAMBDA):
    """
    Sweep S = 1..8, find best stable partition for each S,
    then return the overall optimal S* minimising E[T].
    """
    sweep = {}
    for S in range(1, GRID_COLS + 1):
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
            # All partitions unstable — keep least bad
            lb = min(
                [(evaluate_partition(p, lam), p) for p in parts],
                key=lambda x: x[0]["max_rho"],
            )
            sweep[S] = {"result": lb[0], "partition": lb[1], "feasible": False}
        else:
            sweep[S] = {"result": best[0], "partition": best[1], "feasible": True}

    # S* = lowest E[T] among stable options
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
    """Collects per-task and per-stage statistics during SimPy run."""
    def __init__(self, S):
        self.S             = S
        self.completion_times = []
        self.stage_waits   = [[] for _ in range(S)]   # Wq per stage
        self.stage_queue_lengths = [[] for _ in range(S)]
        self.n_completed   = 0


def run_simpy(groups, lam=LAMBDA, sim_time=SIM_TIME, seed=RANDOM_SEED):
    """
    Run discrete-event simulation of the S-stage M/M/c pipeline.

    Each column is modelled as a SimPy Resource with capacity = GRID_ROWS.
    Poisson arrivals, exponential service times.
    """
    random.seed(seed)
    np.random.seed(seed)

    env   = simpy.Environment()
    S     = len(groups)
    stats = Stats(S)

    # Build resources (one per column)
    resources = [simpy.Resource(env, capacity=GRID_ROWS) for _ in range(S)]

    # Pre-compute service rates per stage
    mu_singles = []
    for group in groups:
        fm = sum(LAYERS[i]["flops_M"] for i in group)
        mu_singles.append(COMPUTE_GFLOPS / (fm / 1000.0))

    comm_delays = []
    for k, group in enumerate(groups):
        last_KB = LAYERS[group[-1]]["act_KB"]
        comm_delays.append(comm_delay_sec(last_KB) if k < S - 1 else 0.0)

    def task(env, task_id):
        arrival = env.now
        for k in range(S):
            # Wait for a free device in this column
            q_before = len(resources[k].queue)
            stats.stage_queue_lengths[k].append(q_before)
            t_req = env.now
            with resources[k].request() as req:
                yield req
                wait = env.now - t_req
                stats.stage_waits[k].append(wait)
                # Exponential service time
                svc = np.random.exponential(1.0 / mu_singles[k])
                yield env.timeout(svc)
            # Inter-column communication delay
            if comm_delays[k] > 0:
                yield env.timeout(comm_delays[k])

        stats.completion_times.append(env.now - arrival)
        stats.n_completed += 1

    def arrivals(env):
        task_id = 0
        while True:
            iat = np.random.exponential(1.0 / lam)
            yield env.timeout(iat)
            env.process(task(env, task_id))
            task_id += 1

    env.process(arrivals(env))
    env.run(until=sim_time)
    return stats


# ─────────────────────────────────────────────────────────
# SECTION 6 — PLOTTING
# ─────────────────────────────────────────────────────────

def plot_sweep(sweep, S_star, lam, save=True):
    """Plot 1: E[T] and max ρ vs S (theory)."""
    S_vals  = list(range(1, GRID_COLS + 1))
    T_vals  = [sweep[S]["result"]["total_T"]  for S in S_vals]
    rho_vals= [sweep[S]["result"]["max_rho"]  for S in S_vals]
    stable  = [sweep[S]["feasible"]           for S in S_vals]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()

    colors = ["#185FA5" if s else "#E24B4A" for s in stable]
    bars = ax1.bar(S_vals, T_vals, color=colors, alpha=0.75, label="E[T] (s)")
    ax2.plot(S_vals, rho_vals, "o-", color="#854F0B", lw=1.5, label="max ρ")
    ax2.axhline(1.0, color="#E24B4A", lw=0.8, ls="--", alpha=0.6)

    ax1.axvline(S_star, color="#185FA5", lw=1.5, ls=":", alpha=0.8)
    ax1.text(S_star + 0.1, max(T_vals) * 0.95, f"S* = {S_star}", color="#185FA5", fontsize=9)

    ax1.set_xlabel("Number of partitions S")
    ax1.set_ylabel("Mean task completion time E[T] (s)")
    ax2.set_ylabel("Max server utilisation ρ")
    ax2.set_ylim(0, 1.2)
    ax1.set_xticks(S_vals)

    p1 = mpatches.Patch(color="#185FA5", alpha=0.75, label="E[T] – stable")
    p2 = mpatches.Patch(color="#E24B4A", alpha=0.75, label="E[T] – unstable")
    p3 = plt.Line2D([0], [0], color="#854F0B", marker="o", label="max ρ")
    ax1.legend(handles=[p1, p2, p3], fontsize=8, loc="upper right")

    plt.title(f"Partition sweep  (λ = {lam} tasks/s, {GRID_ROWS}×{GRID_COLS} grid, {GRID_ROWS} servers/col)")
    plt.tight_layout()
    if save:
        plt.savefig("plot_sweep.png", dpi=150, bbox_inches="tight")
        print("[saved] plot_sweep.png")
    plt.show()


def plot_per_stage(result_star, S_star, save=True):
    """Plot 2: Per-stage ρ and Lq for the optimal partition."""
    stages = result_star["stages"]
    labels = [f"Col {s['col']}\n" + "\n".join(s["layers"]) for s in stages]
    rho_v  = [s["rho"] for s in stages]
    lq_v   = [min(s["Lq"], 20) for s in stages]   # cap for display

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    x = range(len(stages))

    col_colors = ["#85B7EB" if all(l.startswith("Conv") for l in s["layers"])
                  else "#AFA9EC" if all(l.startswith("Bi") for l in s["layers"])
                  else "#5DCAA5"
                  for s in stages]

    ax1.bar(x, rho_v, color=col_colors, edgecolor="#fff", linewidth=0.5)
    ax1.axhline(1.0, color="#E24B4A", lw=0.8, ls="--")
    ax1.axhline(0.85, color="#EF9F27", lw=0.8, ls=":")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=7)
    ax1.set_ylabel("Server utilisation ρ"); ax1.set_ylim(0, 1.1)
    ax1.set_title(f"Per-stage utilisation  (S* = {S_star})")

    ax2.bar(x, lq_v, color=col_colors, edgecolor="#fff", linewidth=0.5)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_ylabel("Mean queue length Lq"); ax2.set_title("Per-stage queue length Lq")

    p1 = mpatches.Patch(color="#85B7EB", label="Conv layers")
    p2 = mpatches.Patch(color="#AFA9EC", label="BiLSTM layers")
    fig.legend(handles=[p1, p2], fontsize=8, loc="lower center", ncol=2)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    if save:
        plt.savefig("plot_stages.png", dpi=150, bbox_inches="tight")
        print("[saved] plot_stages.png")
    plt.show()


def plot_simpy_vs_theory(theory, sim_stats, S_star, save=True):
    """Plot 3: Theory vs SimPy completion time distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # Histogram of SimPy completion times
    ct = sim_stats.completion_times
    ax1.hist(ct, bins=40, color="#85B7EB", edgecolor="#185FA5", lw=0.4, density=True, alpha=0.8)
    ax1.axvline(np.mean(ct), color="#185FA5", lw=1.5, ls="--", label=f"Sim mean={np.mean(ct):.4f}s")
    ax1.axvline(theory["total_T"], color="#854F0B", lw=1.5, ls=":", label=f"Theory={theory['total_T']:.4f}s")
    ax1.set_xlabel("Task completion time (s)"); ax1.set_ylabel("Density")
    ax1.set_title(f"Completion time distribution  (S* = {S_star})")
    ax1.legend(fontsize=8)

    # Per-stage mean wait: theory vs sim
    stages = theory["stages"]
    x = range(len(stages))
    th_wq = [s["Wq"] for s in stages]
    sim_wq = [np.mean(w) if w else 0 for w in sim_stats.stage_waits]
    w = 0.35
    ax2.bar([i - w/2 for i in x], th_wq,  width=w, color="#85B7EB",  label="Theory Wq", alpha=0.85)
    ax2.bar([i + w/2 for i in x], sim_wq, width=w, color="#854F0B",  label="SimPy Wq",  alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Col {s['col']}" for s in stages], fontsize=8)
    ax2.set_ylabel("Mean wait time Wq (s)")
    ax2.set_title("Theory vs SimPy: queue wait per stage")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    if save:
        plt.savefig("plot_theory_vs_sim.png", dpi=150, bbox_inches="tight")
        print("[saved] plot_theory_vs_sim.png")
    plt.show()


def plot_lambda_sensitivity(save=True):
    """Plot 4: Optimal E[T] and S* as λ varies."""
    lam_range = list(range(2, 46))
    T_opt, rho_opt, s_opt, feasible = [], [], [], []

    for lam in lam_range:
        sw, ss = find_optimal_partition(lam)
        r = sw[ss]["result"]
        T_opt.append(r["total_T"] if sw[ss]["feasible"] else None)
        rho_opt.append(r["max_rho"])
        s_opt.append(ss)
        feasible.append(sw[ss]["feasible"])

    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(11, 4))
    ax2 = ax1.twinx()

    T_plot = [t if t is not None else float("nan") for t in T_opt]
    ax1.plot(lam_range, T_plot,  "-o", color="#185FA5", ms=3, lw=1.5, label="E[T] (s)")
    ax2.plot(lam_range, rho_opt, "-s", color="#854F0B", ms=3, lw=1.2, ls="--", label="max ρ")
    ax2.axhline(1.0, color="#E24B4A", lw=0.8, ls=":")

    # Shade unstable region
    for i, (lam, f) in enumerate(zip(lam_range, feasible)):
        if not f:
            ax1.axvspan(lam - 0.5, lam + 0.5, color="#FCEBEB", alpha=0.4)

    ax1.set_xlabel("Task arrival rate λ (tasks/s)")
    ax1.set_ylabel("Optimal E[T] (s)"); ax2.set_ylabel("Max ρ")
    ax1.set_title("λ sensitivity: optimal E[T] and ρ")
    ax1.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    ax3.step(lam_range, s_opt, color="#185FA5", lw=1.5, where="mid")
    ax3.fill_between(lam_range, s_opt, step="mid", alpha=0.15, color="#185FA5")
    ax3.set_xlabel("Task arrival rate λ (tasks/s)")
    ax3.set_ylabel("Optimal S*"); ax3.set_yticks(range(1, 9))
    ax3.set_title("Optimal partition count S* vs λ")

    plt.tight_layout()
    if save:
        plt.savefig("plot_lambda.png", dpi=150, bbox_inches="tight")
        print("[saved] plot_lambda.png")
    plt.show()


def plot_grid_map(sweep, S_star, lam, save=True):
    """Plot 5: 10x8 device grid coloured by partition assignment."""
    result = sweep[S_star]["result"]
    groups = sweep[S_star]["partition"]

    # Build device -> group mapping
    device_group = {}
    col_to_group = {}
    # Assign visual columns evenly to groups
    layers_per_col = 8 / S_star
    for g_idx, group in enumerate(groups):
        for i in group:
            col_to_group[i] = g_idx

    conv_cmap   = plt.cm.Blues
    bilstm_cmap = plt.cm.Purples

    fig, ax = plt.subplots(figsize=(12, 5))
    col_width  = 1.0
    row_height = 0.8

    group_colors = []
    for g_idx, group in enumerate(groups):
        is_conv = all(LAYERS[i]["name"].startswith("Conv") for i in group)
        is_bi   = all(LAYERS[i]["name"].startswith("Bi")   for i in group)
        rho = result["stages"][g_idx]["rho"]
        intensity = 0.3 + 0.6 * (rho / max(r["max_rho"] for r in [result]))
        if is_conv:
            c = conv_cmap(0.3 + 0.5 * (g_idx / max(S_star, 1)))
        elif is_bi:
            c = bilstm_cmap(0.3 + 0.5 * (g_idx / max(S_star, 1)))
        else:
            c = plt.cm.Greens(0.5)
        group_colors.append(c)

    # Draw cells
    for g_idx, group in enumerate(groups):
        x_base = g_idx * col_width
        c = group_colors[g_idx]
        rho = result["stages"][g_idx]["rho"]
        for row in range(GRID_ROWS):
            rect = mpatches.FancyBboxPatch(
                (x_base + 0.05, row * row_height + 0.05),
                col_width - 0.1, row_height - 0.1,
                boxstyle="round,pad=0.02", linewidth=0.5,
                facecolor=c, edgecolor="white"
            )
            ax.add_patch(rect)
            dev_id = g_idx * GRID_ROWS + row
            ax.text(x_base + col_width / 2, row * row_height + row_height / 2,
                    f"D{dev_id}", ha="center", va="center", fontsize=6.5, color="#333")

        # Column label at top
        layer_names = "+".join(LAYERS[i]["name"] for i in group)
        ax.text(x_base + col_width / 2, GRID_ROWS * row_height + 0.15,
                layer_names, ha="center", va="bottom", fontsize=7, fontweight="bold",
                wrap=True)
        ax.text(x_base + col_width / 2, -0.2,
                f"ρ={rho:.3f}", ha="center", va="top", fontsize=7,
                color="#854F0B" if rho > 0.7 else "#185FA5")

    # Task entry arrow
    ax.annotate("Tasks\nenter", xy=(0, GRID_ROWS * row_height / 2),
                xytext=(-0.7, GRID_ROWS * row_height / 2),
                arrowprops=dict(arrowstyle="->", color="#185FA5", lw=1.5),
                fontsize=8, ha="center", va="center", color="#185FA5")

    ax.set_xlim(-1, S_star * col_width + 0.2)
    ax.set_ylim(-0.5, GRID_ROWS * row_height + 0.8)
    ax.axis("off")
    ax.set_title(
        f"10 × {S_star} active device grid  (S* = {S_star}, λ = {lam} tasks/s)\n"
        f"Blue = Conv layers, Purple = BiLSTM layers",
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
    print("\n" + "=" * 70)
    print(f"TP-5 RESULTS  (λ={lam}, grid={GRID_ROWS}×{GRID_COLS}, c={GRID_ROWS} servers/col)")
    print("=" * 70)

    for S in range(1, GRID_COLS + 1):
        d = sweep[S]
        r = d["result"]
        tag = " ← S*" if S == S_star else ""
        ok  = "STABLE" if d["feasible"] else "UNSTABLE"
        print(f"\nS={S}  [{ok}]{tag}")
        print(f"  E[T]={r['total_T']:.4f}s  |  Lq={r['total_Lq']:.4f}  |  max ρ={r['max_rho']:.4f}")
        for st in r["stages"]:
            lyrs = "+".join(st["layers"])
            print(f"    Col{st['col']}: {lyrs:40s} ρ={st['rho']:.4f}  Lq={st['Lq']:.5f}  E[T]={st['E_T']:.5f}s")

    print("\n" + "=" * 70)
    print(f"OPTIMAL  S* = {S_star}")
    r = sweep[S_star]["result"]
    print(f"  E[T]    = {r['total_T']:.5f} s")
    print(f"  Total Lq = {r['total_Lq']:.5f}")
    print(f"  Max ρ    = {r['max_rho']:.5f}")
    print("\nDevice Mapping:")
    for st in r["stages"]:
        devs = f"{(st['col']-1)*GRID_ROWS}–{st['col']*GRID_ROWS-1}"
        lyrs = " + ".join(st["layers"])
        print(f"  Col {st['col']} (devices {devs}): {lyrs}")
        print(f"    FLOPs={st['flops_M']:.0f}M  ρ={st['rho']:.5f}  E[T]={st['E_T']:.5f}s")
    print()


def main():
    lam = LAMBDA  # change this to test other arrival rates

    # ── Theoretical analysis ───────────────────────────────
    print("Running theoretical M/M/c analysis...")
    sweep, S_star = find_optimal_partition(lam)
    print_results(sweep, S_star, lam)

    result_star = sweep[S_star]["result"]
    partition_star = sweep[S_star]["partition"]

    # ── SimPy simulation ────────────────────────────────────
    print(f"Running SimPy simulation (S*={S_star}, T={SIM_TIME}s)...")
    sim_stats = run_simpy(partition_star, lam=lam, sim_time=SIM_TIME)
    print(f"  Tasks completed: {sim_stats.n_completed}")
    print(f"  Sim mean E[T]:   {np.mean(sim_stats.completion_times):.5f} s")
    print(f"  Theory E[T]:     {result_star['total_T']:.5f} s")
    err = abs(np.mean(sim_stats.completion_times) - result_star["total_T"]) / result_star["total_T"] * 100
    print(f"  Error:           {err:.2f}%")

    # ── Plots ───────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_sweep(sweep, S_star, lam)
    plot_per_stage(result_star, S_star)
    plot_simpy_vs_theory(result_star, sim_stats, S_star)
    plot_lambda_sensitivity()
    plot_grid_map(sweep, S_star, lam)
    print("\nDone. All plots saved as PNG files.")


if __name__ == "__main__":
    main()
