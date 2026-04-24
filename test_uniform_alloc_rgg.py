from simulator_rgg import *
import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

def generate_uniform_interleaved_mapping_rgg(G: nx.Graph, S: int, min_cluster_size: int | None = None) -> dict[int, int]:
    """
    1. Detect communities using Louvain (python-louvain).
    2. Merge any cluster smaller than S into its nearest neighbour cluster.
    3. Within each cluster, sort nodes by their position along the cluster
       subgraph's Fiedler vector (principal eigenvector of the Laplacian).
       This gives a 1-D ordering that respects graph distance inside the cluster.
    4. Assign partition = rank % S along that ordering.

    The Fiedler-vector ordering is the RGG equivalent of the anti-diagonal
    coordinate (i+j) used on regular grids.
    """
    try:
        import community as community_louvain
    except ImportError:
        raise ImportError("pip install python-louvain")

    if min_cluster_size is None:
        min_cluster_size = S

    # --- step 1: Louvain community detection ---
    raw_partition = community_louvain.best_partition(G, random_state=42)
    clusters: dict[int, list[int]] = defaultdict(list)
    for node, cid in raw_partition.items():
        clusters[cid].append(node)

    # --- step 2: merge small clusters ---
    changed = True
    while changed:
        changed = False
        for cid, members in list(clusters.items()):
            if len(members) < min_cluster_size and cid in clusters:
                # find neighbouring cluster with most cross-edges
                neighbour_votes: dict[int, int] = defaultdict(int)
                for n in members:
                    for nb in G.neighbors(n):
                        nc = next((c for c, m in clusters.items() if nb in m and c != cid), None)
                        if nc is not None:
                            neighbour_votes[nc] += 1
                if not neighbour_votes:
                    continue
                target = max(neighbour_votes, key=neighbour_votes.get)
                clusters[target].extend(members)
                del clusters[cid]
                changed = True
                break

    # --- step 3 & 4: eigenvector rank within each cluster ---
    assignment: dict[int, int] = {}
    for cid, members in clusters.items():
        if len(members) == 1:
            assignment[members[0]] = 0
            continue
        sub = G.subgraph(members)
        L = nx.laplacian_matrix(sub, nodelist=members).astype(float)
        if len(members) <= 2:
            # trivial case: just assign sequentially
            for rank, node in enumerate(members):
                assignment[node] = rank % S
            continue
        # Fiedler vector: second smallest eigenvector (index 1)
        k = min(2, len(members) - 1)
        _, vecs = eigsh(L, k=k, which='SM')
        fiedler = vecs[:, -1]          # last of the k smallest = Fiedler
        order = np.argsort(fiedler)    # sort nodes along the 1-D embedding
        for rank, idx in enumerate(order):
            assignment[members[idx]] = rank % S

    return assignment

def validate_partition_hops_and_seq_chain(G: nx.Graph, assignment: dict[int, int], S: int) -> None:
    """
    Compute both coverage and sequential-chain metrics in one pass over the
    precomputed shortest-path information.

    Returns:
      - max_hop_to_replica
      - avg_hop_to_replica
      - worst_chain
      - avg_chain
    """
    max_gap = 0
    total_gap = 0
    coverage_count = 0

    worst = 0
    total_chain = 0
    n = G.number_of_nodes()

    all_pairs = dict(nx.all_pairs_shortest_path_length(G))

    # Coverage: for every source node, how far is the nearest replica of each partition?
    for source in G.nodes():
        lengths = all_pairs[source]
        for p in range(S):
            nearest = min((lengths[n] for n in G.nodes() if assignment[n] == p), default=999)
            max_gap = max(max_gap, nearest)
            total_gap += nearest
            coverage_count += 1

    # Chain cost: origin -> nearest P0 -> nearest P1 -> ... -> nearest P(S-1)
    for origin in G.nodes():
        ci = origin
        chain_cost = 0
        for p in range(S):
            candidates = [node for node in G.nodes() if assignment[node] == p]
            best_dist = min(all_pairs[ci][node] for node in candidates)
            next_node = min(candidates, key=lambda nd: all_pairs[ci][nd])
            chain_cost += best_dist
            ci = next_node
        worst = max(worst, chain_cost)
        total_chain += chain_cost

    print("Coverage metrics:")
    print(f"  max_hop_to_replica = {max_gap}")
    print(f"  avg_hop_to_replica = {round(total_gap / coverage_count, 2)}")
    print("Sequential chain metrics:")
    print(f"  worst_chain = {worst}")
    print(f"  avg_chain = {round(total_chain / n, 2)}")

def plot_assignment(G: nx.Graph, pos_array: np.ndarray, assignment: dict[int, int],
             S: int, save_path: str = "rgg_assignment.png"):
    """
    Plot a single assignment plus coverage stats.
    """
    def partition_colors(S: int) -> list[str]:
        palette = ['#378ADD', '#1D9E75', '#D85A30', '#7F77DD',
                '#BA7517', '#D4537E', '#639922', '#E24B4A']
        return [palette[i % len(palette)] for i in range(S)]


    colors = partition_colors(S)
    pos = {n: tuple(pos_array[n]) for n in G.nodes()}

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    fig.patch.set_facecolor("#fdfdfd")
    ax.set_facecolor("#ffffff")

    node_colors = [colors[assignment[n]] for n in G.nodes()]

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15,
                           edge_color='#555555', width=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=80, linewidths=0.5,
                           edgecolors='#222222')

    legend_handles = [
        mpatches.Patch(color=colors[p], label=f'P{p}') for p in range(S)
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=S, framealpha=0, labelcolor='#aaaaaa',
               fontsize=9, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"Saved → {save_path}")
    else:
        plt.show()
    plt.close()

if __name__=="__main__":
    dl_layers_flops_activation_size = {"conv1":{"flops":442368, "activation_size":[1, 16, 32, 32]},
                                       "conv2":{"flops":4718592, "activation_size":[1, 32, 32, 32]},
                                       "conv3":{"flops":4718592, "activation_size":[1, 64, 16, 16]},
                                       "conv4":{"flops":18874368, "activation_size":[1, 128, 16, 16]},
                                       "conv5":{"flops":18874368, "activation_size":[1, 256, 8, 8]},
                                       "lstm1":{"flops":25165824, "activation_size":[1, 64, 256]},
                                       "lstm2":{"flops":25165824, "activation_size":[1, 64, 256]},
                                       "lstm3":{"flops":10485760, "activation_size":[1, 64, 128]}}
    layer_flops = [layer["flops"] for layer in dl_layers_flops_activation_size.values()]
    layer_activation = [activation_size_bytes(layer["activation_size"]) for layer in dl_layers_flops_activation_size.values()]
    # grid_rows, grid_cols = 4, 4
    # num_devices = grid_rows * grid_cols
    num_devices = 100
    device_caps = [1e8] * num_devices
    comm_radius = 0.22
    random_seed = 42
    sqarea_side_len = 1
    
    sim_duration = 1000
    partition = [[0,1,2],[3,4],[5],[6],[7]]
    print(len(partition))

    rgg_graph, pos_array = build_rgg(num_devices, comm_radius, random_seed, sqarea_side_len)

    mapping = generate_uniform_interleaved_mapping_rgg(rgg_graph, len(partition))
    print(mapping)
    validate_partition_hops_and_seq_chain(rgg_graph, mapping, len(partition))

    print("Partition counts (how many nodes hold each partition):")

    counts = [sum(1 for v in mapping.values() if v == p) for p in range(len(partition))]
    print(f" {counts}")
    
    plot_assignment(rgg_graph, pos_array, mapping, len(partition))

    

    env = simpy.Environment()

    sim = Simulator(
        numLayers               = len(layer_flops),  # Fixed: use actual number of layers
        layersFlops             = layer_flops,
        layersActivationSize    = layer_activation,
        numNodes                = num_devices,
        partition               = partition,
        deviceToPartitionMapping= mapping,
        deviceComputeCapacity   = device_caps,
        env                     = env,
        arrival_rate            = 0.5,
        sim_duration            = sim_duration,
        sampling_interval       = 1,
        task_generating_device_ids = list(range(0,100)),
        input_task_size         = activation_size_bytes([1,3,32,32]),
        global_poisson_stream   = False,
        random_seed             = random_seed,
        next_device_assignment_policy='leastload',
        default_bw              = 1e6/8,
        comm_radius             = comm_radius,
        rgg_random_seed         = random_seed,
        sqarea_side_len         = sqarea_side_len,
        next_device_random_seed = 30
    )
 
    sim.simulate()
    env.run(until=sim_duration)
    print(sim.results())


