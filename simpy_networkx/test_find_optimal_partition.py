from simulator import *
import pandas as pd

def generate_partitions(n):
    """Generate all contiguous partitions of n layers"""
    partitions = []
    
    # binary mask for split points
    # 1 means split after that index
    for mask in range(1 << (n - 1)):
        current = [0]
        result = []
        
        for i in range(n - 1):
            if (mask >> i) & 1:
                result.append(current)
                current = []
            current.append(i + 1)
        
        result.append(current)
        partitions.append(result)
    
    return partitions

def idx_to_coord(idx, cols):
    return (idx // cols, idx % cols)

def get_neighbors(coord, grid_rows, grid_cols):
    x, y = coord
    neighbors = []
    for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
        nx, ny = x+dx, y+dy
        if 0 <= nx < grid_rows and 0 <= ny < grid_cols:
            neighbors.append((nx, ny))
    return neighbors

def generate_paths(start, length, grid_rows, grid_cols):
    paths = []

    def dfs(path):
        if len(path) == length:
            paths.append(path[:])
            return
        
        last = path[-1]
        for nei in get_neighbors(last, grid_rows, grid_cols):
            if nei not in path:  # avoid cycles
                path.append(nei)
                dfs(path)
                path.pop()

    dfs([start])
    return paths

def get_snake_path(grid_rows, grid_cols):
    path = []
    for i in range(grid_rows):
        row = [(i, j) for j in range(grid_cols)]
        if i % 2 == 1:
            row.reverse()  # zig-zag
        path.extend(row)
    return path

def get_center_path(k, grid_rows, grid_cols):
    coords = [(i, j) for i in range(grid_rows) for j in range(grid_cols)]
    center = len(coords) // 2
    return coords[center:center+k]

def coord_to_idx(coord, cols):
    return coord[0]*cols + coord[1]

def build_mapping(partition, path, cols):
    mapping = {}
    for i, layers in enumerate(partition):
        device_idx = coord_to_idx(path[i], cols)
        mapping[device_idx] = layers
    return mapping

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
    
    sim_duration = 1000
    num_layers = len(dl_layers_flops_activation_size)



    partitions = generate_partitions(num_layers)
    # print(partitions)

    # paths = generate_paths((0,0),8,4,4)
    # print(paths)

    results = []

    for partition in partitions:
        k = len(partition)

        # skip if more partitions than devices
        if k > num_devices:
            continue

        full_path = get_snake_path(grid_rows, grid_cols)
        path = full_path[:k]
        mapping = build_mapping(partition, path, grid_cols)

        print(mapping)

        env = simpy.Environment()
        sim = Simulator(
            numLayers               = len(layer_flops),  # Fixed: use actual number of layers
            layersFlops             = layer_flops,
            layersActivationSize    = layer_activation,
            gridSize                = (grid_rows, grid_cols),
            deviceToPartitionMapping= mapping,
            deviceComputeCapacity   = device_caps,
            linksBandwidth          = links_bw,
            env                     = env,
            arrival_rate            = 0.2,
            sim_duration            = sim_duration
        )
    
        sim.simulate()
        env.run(until=sim_duration)
        res = sim.results()
        print(res)

        results.append({
            "partition": mapping,
            "num_devices": k,
            "mean_latency": res["mean_latency"],
            "p95_latency": res["p95_latency"],
            "bottleneck": res["bottleneck_partition"],
            "util": res["max_utilization"],
            "mean_compute_queue":  res["mean_compute_queue"],
            "mean_comm_queue":     res["mean_comm_queue"],
            "mean_overall_queue":  res["mean_overall_queue"],
        })
    
    print(results)
    df = pd.DataFrame(results)
    print(df)
    df.to_csv("results.csv")
    # return results