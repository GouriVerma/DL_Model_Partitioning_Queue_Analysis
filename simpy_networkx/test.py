from simulator import *

if __name__ == "__main__":

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

    # Create device to partition mapping (each device handles one layer)
    # Device to list of layers in device, need to be sequential, cant be 1:[1], 0:[0]
    device_to_partition = {0:[0], 1:[1], 2:[2], 3:[3], 7:[4], 6:[5], 5:[6], 4:[7]}
    
    # Create links bandwidth dict for 4x4 grid, ex- {((0,0),(0,1)):1e6, ...}
    links_bw = {}
    for i in range(grid_rows):
        for j in range(grid_cols):
            if i < grid_rows - 1:
                links_bw[((i, j), (i+1, j))] = 1e6  # Vertical links
            if j < grid_cols - 1:
                links_bw[((i, j), (i, j+1))] = 1e6  # Horizontal links
    
    sim_duration = 1000

    env = simpy.Environment()
    sim = Simulator(
        numLayers               = len(layer_flops),  # Fixed: use actual number of layers
        layersFlops             = layer_flops,
        layersActivationSize    = layer_activation,
        gridSize                = (grid_rows, grid_cols),
        deviceToPartitionMapping= device_to_partition,
        deviceComputeCapacity   = device_caps,
        linksBandwidth          = links_bw,
        env                     = env,
        arrival_rate            = 0.2,
        sim_duration            = sim_duration
    )
 
    sim.simulate()
    env.run(until=sim_duration)
    print(sim.results())
 