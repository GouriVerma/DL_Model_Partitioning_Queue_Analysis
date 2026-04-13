from simulator import *
import simpy

if __name__ == "__main__":

    # -----------------------------
    # ORIGINAL FLOPs
    # -----------------------------
    conv_flops = [
        442368,
        4718592,
        4718592,
        18874368,
        18874368
    ]

    lstm_flops = [
        25165824,
        25165824,
        10485760
    ]

    # -----------------------------
    # SPLIT LSTM INTO:
    # forward, backward, merge
    # -----------------------------
    split_lstm_flops = []
    for f in lstm_flops:
        split_lstm_flops.append(f / 2)        # forward
        split_lstm_flops.append(f / 2)        # backward
        split_lstm_flops.append(0.1 * f)      # merge (small cost)

    # -----------------------------
    # FINAL LAYER FLOPs
    # -----------------------------
    layer_flops = conv_flops + split_lstm_flops

    # -----------------------------
    # ACTIVATION SIZES (approx)
    # -----------------------------
    dl_layers_activation_shapes = [
        [1, 16, 32, 32],
        [1, 32, 32, 32],
        [1, 64, 16, 16],
        [1, 128, 16, 16],
        [1, 256, 8, 8],
    ]

    # LSTM activations (same replicated)
    lstm_activation_shapes = [
        [1, 64, 256],
        [1, 64, 256],
        [1, 64, 128]
    ]

    split_lstm_activations = []
    for shape in lstm_activation_shapes:
        split_lstm_activations.append(shape)  # forward
        split_lstm_activations.append(shape)  # backward
        split_lstm_activations.append(shape)  # merge

    all_shapes = dl_layers_activation_shapes + split_lstm_activations

    layer_activation = [activation_size_bytes(s) for s in all_shapes]

    # -----------------------------
    # GRID
    # -----------------------------
    grid_rows, grid_cols = 4, 4
    num_devices = grid_rows * grid_cols

    device_caps = [1e8] * num_devices

    # -----------------------------
    # DEVICE MAPPING
    # -----------------------------
    # Each layer → one device (pipeline)
    device_to_partition = {
    # Conv layers
    0: [0],
    1: [1],
    2: [2],
    3: [3],
    4: [4],

    # LSTM 1
    5: [5],   # forward
    6: [6],   # backward
    7: [7],   # merge

    # LSTM 2
    8: [8],
    9: [9],
    10: [10],

    # LSTM 3
    11: [11],
    12: [12],
    13: [13],
}

    # -----------------------------
    # LINKS
    # -----------------------------
    links_bw = {}
    for i in range(grid_rows):
        for j in range(grid_cols):
            if i < grid_rows - 1:
                links_bw[((i, j), (i+1, j))] = 1e6
            if j < grid_cols - 1:
                links_bw[((i, j), (i, j+1))] = 1e6

    # -----------------------------
    # SIMULATION
    # -----------------------------
    sim_duration = 1000

    env = simpy.Environment()

    sim = Simulator(
        numLayers               = len(layer_flops),
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

    results = sim.results()
    print(results)
