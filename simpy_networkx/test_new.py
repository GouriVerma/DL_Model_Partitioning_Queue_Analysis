from simpy_networkx.new_simulator import *
import simpy

if __name__ == "__main__":

    # ----------------------------------------------------------------
    # FLOPs
    # ----------------------------------------------------------------
    conv_flops = [
        442368,
        4718592,
        4718592,
        18874368,
        18874368,
    ]

    lstm_flops = [
        25165824,   # LSTM 1
        25165824,   # LSTM 2
        10485760,   # LSTM 3
    ]

    # Each LSTM layer is split into:
    #   forward  → half FLOPs
    #   backward → half FLOPs
    #   merge    → 10 % FLOPs (cheap reduction/concat)
    split_lstm_flops = []
    for f in lstm_flops:
        split_lstm_flops.append(f / 2)       # forward
        split_lstm_flops.append(f / 2)       # backward
        split_lstm_flops.append(0.1 * f)     # merge

    layer_flops = conv_flops + split_lstm_flops
    # Index layout:
    #   0-4   → conv layers
    #   5,6,7 → LSTM-1  (forward, backward, merge)
    #   8,9,10→ LSTM-2
    #  11,12,13→ LSTM-3

    # ----------------------------------------------------------------
    # Activation sizes
    # ----------------------------------------------------------------
    conv_activation_shapes = [
        [1, 16, 32, 32],
        [1, 32, 32, 32],
        [1, 64, 16, 16],
        [1, 128, 16, 16],
        [1, 256, 8, 8],
    ]

    lstm_activation_shapes = [
        [1, 64, 256],   # LSTM 1
        [1, 64, 256],   # LSTM 2
        [1, 64, 128],   # LSTM 3
    ]

    split_lstm_activations = []
    for shape in lstm_activation_shapes:
        split_lstm_activations.append(shape)  # forward  output
        split_lstm_activations.append(shape)  # backward output
        split_lstm_activations.append(shape)  # merge    output

    all_shapes      = conv_activation_shapes + split_lstm_activations
    layer_activation = [activation_size_bytes(s) for s in all_shapes]

    # ----------------------------------------------------------------
    # Grid  (4×4 = 16 devices; we use 14 of them)
    # ----------------------------------------------------------------
    grid_rows, grid_cols = 4, 4
    num_devices          = grid_rows * grid_cols          # 16
    device_caps          = [1e8] * num_devices

    # ----------------------------------------------------------------
    # Device → partition mapping
    #
    # Convention used by device_worker:
    #   Normal device  : [layer_idx]
    #   Forward branch : ["forward",  layer_idx]
    #   Backward branch: ["backward", layer_idx]
    #   Merge device   : ["merge",    layer_idx]
    #
    # The Simulator.__init__ detects LSTM groups automatically by
    # scanning for consecutive (forward, backward, merge) triples and
    # records:
    #   lstm_groups[fan_out_device]  = {forward, backward, merge device indices}
    #   merge_sources[merge_device]  = {forward, backward device indices}
    #
    # Pipeline order (keys must be in execution order):
    #   conv0 → conv1 → conv2 → conv3 → conv4
    #                                      ↓  (fan-out)
    #                               ┌──── fwd5 ────┐
    #                               └──── bwd6 ────┘
    #                                      ↓  (merge, then continue)
    #                                   merge7
    #                                      ↓
    #                               ┌──── fwd8 ────┐
    #                               └──── bwd9 ────┘
    #                                      ↓
    #                                  merge10
    #                                      ↓
    #                               ┌──── fwd11 ───┐
    #                               └──── bwd12 ───┘
    #                                      ↓
    #                                  merge13  (final device)
    # ----------------------------------------------------------------
    device_to_partition = {
        # Conv layers — normal sequential pipeline
        0:  [0],
        1:  [1],
        2:  [2],
        3:  [3],
        4:  [4],          # <-- fan-out device for LSTM-1 (auto-detected)

        # LSTM 1
        5:  ["forward",  5],
        6:  ["backward", 6],
        7:  ["merge",    7],   # <-- fan-out device for LSTM-2

        # LSTM 2
        8:  ["forward",  8],
        9:  ["backward", 9],
        10: ["merge",    10],  # <-- fan-out device for LSTM-3

        # LSTM 3
        11: ["forward",  11],
        12: ["backward", 12],
        13: ["merge",    13],  # final device
    }

    # ----------------------------------------------------------------
    # Link bandwidths  (all edges in the 4×4 grid)
    # ----------------------------------------------------------------
    links_bw = {}
    for i in range(grid_rows):
        for j in range(grid_cols):
            if i < grid_rows - 1:
                links_bw[((i, j), (i + 1, j))] = 1e6
            if j < grid_cols - 1:
                links_bw[((i, j), (i, j + 1))] = 1e6

    # ----------------------------------------------------------------
    # Run simulation
    # ----------------------------------------------------------------
    sim_duration = 1000

    env = simpy.Environment()
    sim = Simulator(
        numLayers                = len(layer_flops),
        layersFlops              = layer_flops,
        layersActivationSize     = layer_activation,
        gridSize                 = (grid_rows, grid_cols),
        deviceToPartitionMapping = device_to_partition,
        deviceComputeCapacity    = device_caps,
        linksBandwidth           = links_bw,
        env                      = env,
        arrival_rate             = 0.2,
        sim_duration             = sim_duration,
        sampling_interval        = 1,
    )

    sim.simulate()
    env.run(until=sim_duration)

    results = sim.results()

    # ----------------------------------------------------------------
    # Pretty-print summary
    # ----------------------------------------------------------------
    print("\n===== SIMULATION RESULTS =====")
    print(f"Tasks completed  : {results['tasks_completed']}")
    print(f"Mean latency     : {results['mean_latency']:.4f}")
    print(f"P95  latency     : {results['p95_latency']:.4f}")
    print(f"Max  latency     : {results['max_latency']:.4f}")
    print(f"Bottleneck       : {results['bottleneck_partition']}")
    print(f"Max utilization  : {results['max_utilization']:.4f}")
    print(f"Mean compute Q   : {results['mean_compute_queue']:.4f}")
    print(f"Mean comm    Q   : {results['mean_comm_queue']:.4f}")
    print(f"Mean overall Q   : {results['mean_overall_queue']:.4f}")
    print("\n--- Per-stage metrics ---")
    for stage, m in results["stage_metrics"].items():
        if m["tasks_processed"] > 0 or m["utilization"] > 0:
            print(
                f"  {stage:40s}  util={m['utilization']:.4f}  "
                f"tasks={m['tasks_processed']}  "
                f"mean_q={m['mean_queue_length']:.3f}"
            )
