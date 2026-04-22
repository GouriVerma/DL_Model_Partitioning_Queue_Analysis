from simulator_multiple_devices_per_partition_multiple_sources import *

def generate_uniform_interleaved_mapping(partition, grid_rows, grid_cols):
    num_partitions = len(partition)
    mapping = {}
    for i in range(grid_rows):
        for j in range(grid_cols):
            block_idx = (i + j) % num_partitions
            device_idx = i * grid_cols + j
            mapping[device_idx] = block_idx
    return mapping


def validate_partition_hops(mapping, grid_rows, grid_cols, num_partitions):
    """
    For each partition k (except last), compute the nearest-hop distance from every
    device in partition k to devices in partition k+1, and print range.
    """
    grid = networkx.grid_2d_graph(grid_rows, grid_cols)

    partition_to_devices = {p: [] for p in range(num_partitions)}
    for d, p in mapping.items():
        partition_to_devices[p].append(d)

    def idx_to_coord(idx):
        return (idx // grid_cols, idx % grid_cols)

    for p in range(num_partitions - 1):
        src_devices = partition_to_devices[p]
        dst_devices = partition_to_devices[p + 1]
        if not src_devices or not dst_devices:
            print(f"Partition {p}->{p+1}: missing devices")
            continue

        nearest_hops = []
        for sd in src_devices:
            s = idx_to_coord(sd)
            best = min(
                networkx.shortest_path_length(grid, source=s, target=idx_to_coord(dd))
                for dd in dst_devices
            )
            nearest_hops.append(best)

        print(
            f"Partition {p}->{p+1} nearest-hop range: "
            f"min={min(nearest_hops)}, max={max(nearest_hops)}"
        )

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
    grid_rows, grid_cols = 4, 4
    num_devices = grid_rows * grid_cols
    device_caps = [1e8] * num_devices
    
    # Create links bandwidth dict for 4x4 grid
    links_bw = {}
    for i in range(grid_rows):
        for j in range(grid_cols):
            if i < grid_rows - 1:
                links_bw[((i, j), (i+1, j))] = 1e6/8  # Vertical links, bandwidth converted from bps => Bytes per second
            if j < grid_cols - 1:
                links_bw[((i, j), (i, j+1))] = 1e6/8  # Horizontal links, bandwidth converted from bps => Bytes per second
    
    sim_duration = 5000
    partition = [[0,1,2],[3,4],[5],[6],[7]]
    mapping = generate_uniform_interleaved_mapping(partition, grid_rows, grid_cols)
    # validate_partition_hops(mapping, grid_rows, grid_cols, len(partition))

    env = simpy.Environment()
    sim = Simulator(
        numLayers               = len(layer_flops),  # Fixed: use actual number of layers
        layersFlops             = layer_flops,
        layersActivationSize    = layer_activation,
        gridSize                = (grid_rows, grid_cols),
        partition               = partition,
        deviceToPartitionMapping= mapping,
        deviceComputeCapacity   = device_caps,
        linksBandwidth          = links_bw,
        env                     = env,
        arrival_rate            = 0.2,
        sim_duration            = sim_duration,
        sampling_interval       = 1,
        task_generating_device_ids = [0, 15],
        input_task_size         = activation_size_bytes([1,3,32,32]),
        global_poisson_stream   = False,
        random_seed             = 42,
        next_device_assignment_policy='nn'
    )
 
    sim.simulate()
    env.run(until=sim_duration)
    print(sim.results())


