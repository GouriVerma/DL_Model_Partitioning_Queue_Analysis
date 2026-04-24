# DL Model Partitioning Queue Analysis

This project simulates distributed execution of a DL model across networked devices and studies latency, utilization, and queue behavior under different placement policies.

## Main workflow

1. Define model compute/activation sizes (or estimate FLOPs).
2. Run a simulator test script (grid, RGG, or forward/backward LSTM split).
3. Collect CSV outputs (for partition sweep experiments).
4. Generate plots from CSV results.

---

## File guide

- [FLOPS_Calculation.py](FLOPS_Calculation.py)  
	Estimates per-layer FLOPs for the Conv+BiLSTM model using forward hooks.

- [simulator_grid.py](simulator_grid.py)  
	Core simulator on a 2D grid topology.

- [test_uniform_alloc_grid.py](test_uniform_alloc_grid.py)  
	Driver for `simulator_grid`; lets you set grid size, device count, partitioning, source devices, and routing/replica assignment policy.

- [simulator_rgg.py](simulator_rgg.py)  
	Core simulator on a Random Geometric Graph (RGG) topology.

- [test_uniform_alloc_rgg.py](test_uniform_alloc_rgg.py)  
	Driver for `simulator_rgg`; lets you set node count, communication radius, bandwidth defaults/mapping, and assignment policy.

- [test_find_optimal_partition.py](test_find_optimal_partition.py)  
	Enumerates all contiguous partitions of 8 layers, runs simulation for each, and writes:
	- `results.csv`
	- `results_queue_details.csv`

- [analyse_results.py](analyse_results.py)  
	Reads the two CSV files and generates plots (utilization, latency, queue metrics, and per-device/per-link queue bars for best partition).

- [simulator_forward_backward_lstm_split.py](simulator_forward_backward_lstm_split.py)  
	Simulator variant where each LSTM block is modeled as forward branch + backward branch + merge stage.

- [test_forward_backward_lstm_split.py](test_forward_backward_lstm_split.py)  
	Test/driver for forward-backward-merge LSTM split mapping.

---

## How to run

From [DL_Model_Partitioning_Queue_Analysis](.):
- FLOPs estimation: `python3 FLOPS_Calculation.py`
- Grid run: `python3 test_uniform_alloc_grid.py`
- RGG run: `python3 test_uniform_alloc_rgg.py`
- Exhaustive partition search: `python3 test_find_optimal_partition.py`
- Plot generation from CSVs generated from `test_find_optimal_partition.py`: `python3 analyse_results.py`
- Forward/backward LSTM split test: `python3 test_forward_backward_lstm_split.py`

---

## Parameters you can change (by script)

## 1) Grid experiments

Configured mainly in [test_uniform_alloc_grid.py](test_uniform_alloc_grid.py) and passed into [simulator_grid.py](simulator_grid.py).

Key parameters:

- `grid_rows`, `grid_cols`: topology size.
- `num_devices`: usually `grid_rows * grid_cols`.
- `device_caps`: per-device compute capacities.
- `partition`: list of layer groups (pipeline stages).
- `mapping`: device-to-partition assignment.
- `links_bw`: per-grid-edge bandwidth map.
- `arrival_rate`: Poisson input rate.
- `sim_duration`: total simulation time.
- `sampling_interval`: queue/utilization sampling period.
- `task_generating_device_ids`: source devices.
- `input_task_size`: input message size (bytes).
- `global_poisson_stream`: one global source stream vs independent source streams.
- `random_seed`, `next_device_random_seed`: reproducibility.
- `next_device_assignment_policy`: This controls how the next device is selected when a partition has multiple replicas. In [simulator_grid.py](simulator_grid.py), supported policies are:

    - `"random"`: random replica selection.
    - `"nn"`: nearest-neighbor style selection (network-distance aware).
    - `"rr"`: round-robin among replicas.
    - `"leastload"`: picks the least loaded replica based on queue/load state.

    This option strongly affects latency and queue build-up and is one of the most important knobs for experiments.

---

## 2) RGG experiments

Configured mainly in [test_uniform_alloc_rgg.py](test_uniform_alloc_rgg.py) and passed into [simulator_rgg.py](simulator_rgg.py).

Key parameters:

- `num_devices` / `numNodes`: number of graph nodes.
- `comm_radius`: geometric communication radius (changes graph connectivity and hop lengths).
- `sqarea_side_len`: physical area side length for node placement.
- `rgg_random_seed`: seed for graph generation.
- `device_caps`: per-device compute capacities.
- `partition`: partition definition.
- `deviceToPartitionMapping`: assignment of partitions to nodes.
- `task_generating_device_ids`: source nodes.
- `input_task_size`: input message size (bytes).
- `global_poisson_stream`: one global source stream vs independent source streams.
- `random_seed`, `next_device_random_seed`: reproducibility.
- `default_bw`: bandwidth used for edges not explicitly listed.
- `next_device_assignment_policy`: This controls how the next device is selected when a partition has multiple replicas. In [simulator_rgg.py](simulator_rgg.py), supported policies are:

    - `"random"`: random replica selection.
    - `"nn"`: nearest-neighbor style selection (network-distance aware).
    - `"rr"`: round-robin among replicas.
    - `"leastload"`: picks the least loaded replica based on queue/load state.

---

## 3) Optimal partition sweep

In [test_find_optimal_partition.py](test_find_optimal_partition.py):

- Enumerates all contiguous partitions of 8 layers.
- Runs simulation for each valid partition/device count combination.
- Exports:
	- `results.csv` (global metrics per partition)
	- `results_queue_details.csv` (device/link-level queue details)

Then [analyse_results.py](analyse_results.py) is used to create plots from these CSVs.

---

## 4) Forward/backward LSTM split experiment

[simulator_forward_backward_lstm_split.py](simulator_forward_backward_lstm_split.py) models each LSTM block with three stages:

- forward branch
- backward branch
- merge stage

[test_forward_backward_lstm_split.py](test_forward_backward_lstm_split.py) sets:

- split FLOPs and activations,
- role-tagged device mapping (normal / `"forward"` / `"backward"` / `"merge"`),
- network bandwidth and simulation parameters.

Use this to evaluate branch synchronization and merge effects in queueing/latency.

---

## Output artifacts

- CSV results for sweeps: `results.csv`, `results_queue_details.csv`
- Plot images generated under the chosen output directory (default `plots/`)
- Logs (based on simulator variant log file names)

---

