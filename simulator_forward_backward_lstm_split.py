import simpy
import networkx
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import logging
logging.basicConfig(filename='simulator.log', level=logging.DEBUG, filemode='w')

def normalize_edge(u, v):
    return tuple(sorted([u, v]))

def activation_size_bytes(shape, dtype_bytes=4):
    num_elements = np.prod(shape)
    return (num_elements * dtype_bytes)

def get_layer_idx(layers_on_device):
    """Extract the integer layer index from a layers list that may contain string tags."""
    return next(x for x in layers_on_device if isinstance(x, int))

def get_device_role(layers_on_device):
    """Return the string role tag ('forward'/'backward'/'merge') or None for normal devices."""
    for x in layers_on_device:
        if isinstance(x, str):
            return x
    return None


@dataclass
class TaskRecord:
    """Stores timing breakdowns for one completed task."""
    task_id:          int
    arrival_time:     float
    origin:           Tuple[int, int] = (0, 0)
    completion_time:  float = 0.0
    stage_breakdown:  List[Tuple] = field(default_factory=list)

    @property
    def total_latency(self):
        return self.completion_time - self.arrival_time


@dataclass
class StageMetrics:
    """Running stats for one partition stage."""
    partition_idx:   int
    busy_time:       float = 0.0
    total_wait_time: float = 0.0
    tasks_processed: int   = 0
    queue_samples:   List[float] = field(default_factory=list)

    def utilization(self, sim_duration: float) -> float:
        return self.busy_time / sim_duration if sim_duration > 0 else 0.0

    def mean_queue_length(self) -> float:
        return float(np.mean(self.queue_samples)) if self.queue_samples else 0.0

    def max_queue_length(self) -> float:
        return float(np.max(self.queue_samples)) if self.queue_samples else 0.0


class Device:

    def __init__(self, compute_cap, env):
        self.compute_capacity = compute_cap
        self.compute_resource = simpy.Resource(env, capacity=1)
        self.input_queue = simpy.Store(env)
        self.env = env

    def run(self, flops=0):
        delay = flops / self.compute_capacity
        start_wait = self.env.now
        with self.compute_resource.request() as req:
            yield req
            wait_time = self.env.now - start_wait
            yield self.env.timeout(delay)
        return wait_time


class Link:

    def __init__(self, bandwidth, env):
        self.bandwidth = bandwidth
        self.transmission_queue = simpy.Store(env)
        self.env = env
        self.linkResource = simpy.Resource(env, capacity=1)
        self.busy_time = 0.0

    def latency(self, message, message_size):
        delay = message_size / self.bandwidth
        with self.linkResource.request() as req:
            yield req
            logging.info(f"Message gets the link resource for task {message.task_id} at {self.env.now}")
            yield self.env.timeout(delay)
            self.busy_time += delay
            yield self.transmission_queue.put(message)

    def put(self, message_size, message):
        self.env.process(self.latency(message, message_size))

    def get(self):
        return self.transmission_queue.get()


class Simulator:

    def __init__(self, numLayers, layersFlops, layersActivationSize, gridSize,
                 deviceToPartitionMapping, deviceComputeCapacity, linksBandwidth,
                 env, arrival_rate=1, default_bw=1, sim_duration=50, sampling_interval=1):

        self.numLayers = numLayers
        self.layerFlops = layersFlops
        self.layersActivationSize = layersActivationSize
        self.gridSize = gridSize
        self.deviceToPartitionMapping = deviceToPartitionMapping
        self.deviceComputeCapacity = deviceComputeCapacity
        self.linksBandwidth = linksBandwidth
        self.arrivalRate = arrival_rate
        self.simDuration = sim_duration
        self.samplingInterval = sampling_interval
        self.env = env

        # Initialise Grid
        self.grid = networkx.grid_2d_graph(gridSize[0], gridSize[1])

        # Initialise Devices
        for idx, coord in enumerate(list(self.grid.nodes)):
            self.grid.nodes[coord]["device"] = Device(
                compute_cap=deviceComputeCapacity[idx], env=self.env
            )

        # Initialise Links
        for edge in self.grid.edges:
            bw = linksBandwidth.get(edge, default_bw)
            self.grid.edges[edge]["link"] = Link(bw, env)

        # Create reverse mapping: integer layer index -> device_idx
        # Skip string role tags ("forward", "backward", "merge")
        self.layer_to_device = {}
        for device_idx, layers in self.deviceToPartitionMapping.items():
            for layer in layers:
                if isinstance(layer, int):
                    self.layer_to_device[layer] = device_idx

        # ----------------------------------------------------------------
        # Detect LSTM groups automatically by scanning device_keys in order.
        #
        # lstm_groups:   { fan_out_device_idx -> {"forward": dev, "backward": dev, "merge": dev} }
        #   - The device immediately before a forward/backward/merge triple is the fan-out device.
        #
        # merge_sources: { merge_device_idx -> {"forward": dev, "backward": dev} }
        #   - Used by the forward branch to know where to deliver the task.
        # ----------------------------------------------------------------
        self.lstm_groups:   Dict[int, Dict[str, int]] = {}
        self.merge_sources: Dict[int, Dict[str, int]] = {}

        device_keys = list(self.deviceToPartitionMapping.keys())
        i = 0
        while i < len(device_keys):
            dev  = device_keys[i]
            role = get_device_role(self.deviceToPartitionMapping[dev])
            if role == "forward" and i + 2 < len(device_keys):
                bwd_dev   = device_keys[i + 1]
                merge_dev = device_keys[i + 2]
                bwd_role   = get_device_role(self.deviceToPartitionMapping[bwd_dev])
                merge_role = get_device_role(self.deviceToPartitionMapping[merge_dev])
                if bwd_role == "backward" and merge_role == "merge":
                    fan_out_dev = device_keys[i - 1] if i > 0 else None
                    if fan_out_dev is not None:
                        self.lstm_groups[fan_out_dev] = {
                            "forward":  dev,
                            "backward": bwd_dev,
                            "merge":    merge_dev,
                        }
                    self.merge_sources[merge_dev] = {
                        "forward":  dev,
                        "backward": bwd_dev,
                    }
                    i += 3
                    continue
            i += 1

        logging.info("LSTM groups detected:")
        for fo, grp in self.lstm_groups.items():
            logging.info(f"  fan-out device {fo}: {grp}")
        logging.info("Merge sources:")
        for m, src in self.merge_sources.items():
            logging.info(f"  merge device {m}: {src}")

        # Create edges to index dict
        self.edge_to_idx = {
            normalize_edge(u, v): idx
            for idx, (u, v) in enumerate(self.grid.edges())
        }

        # join_buffer: {task_id: {"forward": bool, "backward": bool}}
        self.join_buffer: Dict[int, Dict[str, bool]] = {}

        self.task_records: List[TaskRecord] = []
        self.stage_metrics_device: Dict[int, StageMetrics] = {
            p: StageMetrics(partition_idx=p) for p in range(len(deviceComputeCapacity))
        }
        self.stage_metrics_link: Dict[int, StageMetrics] = {
            p: StageMetrics(partition_idx=p) for p in range(self.grid.number_of_edges())
        }

        logging.info("Device to Partition Mapping:")
        for device_idx, layers in self.deviceToPartitionMapping.items():
            coord = self.get_device_coord(device_idx)
            logging.info(f"  Device {device_idx} at {coord}: layers {layers}")

    # ------------------------------------------------------------------
    def get_device_coord(self, device_idx):
        """Get grid coordinate of device by its index."""
        return list(self.grid.nodes)[device_idx]

    def send_message(self, task, src_coord, dest_coord, message_size, dest_device):
        """Transmit task from src to dest through the shortest path (multi-hop)."""
        logging.info(f"Task {task.task_id}: send {src_coord} -> {dest_coord} at {self.env.now:.4f}")
        path = networkx.shortest_path(self.grid, src_coord, dest_coord)

        current_task = task
        for i in range(len(path) - 1):
            cur = path[i]
            nxt = path[i + 1]
            if (cur, nxt) in self.grid.edges:
                link = self.grid.edges[(cur, nxt)]["link"]
            else:
                link = self.grid.edges[(nxt, cur)]["link"]

            if link is None:
                raise ValueError(f"No link for edge {(cur, nxt)}")

            link.put(message_size, current_task)
            edge_key = normalize_edge(cur, nxt)
            link_idx = self.edge_to_idx[edge_key]
            self.stage_metrics_link[link_idx].tasks_processed += 1
            current_task = yield link.get()
            logging.info(f"Task {current_task.task_id} hop {cur}->{nxt} done at {self.env.now:.4f}")

        yield dest_device.input_queue.put(current_task)
        logging.info(f"Task {task.task_id} delivered to {dest_coord} at {self.env.now:.4f}")

    # ------------------------------------------------------------------
    def simulate(self):
        """Main simulation loop."""

        def task_generator():
            device_0_coord = self.get_device_coord(0)
            device_0 = self.grid.nodes[device_0_coord]["device"]
            task_id = 0
            while True:
                inter_arrival = np.random.exponential(1.0 / self.arrivalRate)
                yield self.env.timeout(inter_arrival)
                record = TaskRecord(task_id=task_id, origin=(0, 0), arrival_time=self.env.now)
                yield device_0.input_queue.put(record)
                logging.info(f"Task {task_id} generated at {self.env.now:.4f}")
                task_id += 1

        # ------------------------------------------------------------------
        def device_worker(device_idx):
            device_coord     = self.get_device_coord(device_idx)
            device           = self.grid.nodes[device_coord]["device"]
            layers_on_device = self.deviceToPartitionMapping[device_idx]
            role             = get_device_role(layers_on_device)
            layer_idx        = get_layer_idx(layers_on_device)

            while True:
                task = yield device.input_queue.get()
                logging.info(
                    f"Device {device_idx} (role={role}): got task {task.task_id} at {self.env.now:.4f}"
                )

                # ------------------------------------------------------
                # LSTM FORWARD branch
                #   1. Compute
                #   2. Flag join_buffer["forward"]
                #   3. Send task to merge device  <-- forward is responsible
                #   4. continue (skip generic pipeline-forward)
                # ------------------------------------------------------
                if role == "forward":
                    flops = self.layerFlops[layer_idx]
                    yield self.env.process(device.run(flops=flops))
                    self.stage_metrics_device[device_idx].tasks_processed += 1
                    self.stage_metrics_device[device_idx].busy_time += (
                        flops / self.deviceComputeCapacity[device_idx]
                    )

                    if task.task_id not in self.join_buffer:
                        self.join_buffer[task.task_id] = {"forward": False, "backward": False}
                    self.join_buffer[task.task_id]["forward"] = True

                    # Deliver task to the merge device for this LSTM group
                    merge_dev_idx = next(
                        (m for m, src in self.merge_sources.items() if src["forward"] == device_idx),
                        None
                    )
                    if merge_dev_idx is not None:
                        merge_coord  = self.get_device_coord(merge_dev_idx)
                        merge_device = self.grid.nodes[merge_coord]["device"]
                        message_size = self.layersActivationSize[layer_idx]
                        self.env.process(self.send_message(
                            task, device_coord, merge_coord, message_size, merge_device
                        ))
                        logging.info(
                            f"Task {task.task_id}: forward {device_idx} -> merge {merge_dev_idx}"
                        )
                    continue

                # ------------------------------------------------------
                # LSTM BACKWARD branch
                #   1. Compute
                #   2. Flag join_buffer["backward"]
                #   3. Do NOT send anything — forward handles merge delivery
                #   4. continue (skip generic pipeline-forward)
                # ------------------------------------------------------
                elif role == "backward":
                    flops = self.layerFlops[layer_idx]
                    yield self.env.process(device.run(flops=flops))
                    self.stage_metrics_device[device_idx].tasks_processed += 1
                    self.stage_metrics_device[device_idx].busy_time += (
                        flops / self.deviceComputeCapacity[device_idx]
                    )

                    if task.task_id not in self.join_buffer:
                        self.join_buffer[task.task_id] = {"forward": False, "backward": False}
                    self.join_buffer[task.task_id]["backward"] = True

                    logging.info(
                        f"Task {task.task_id}: backward {device_idx} done, flagged join_buffer"
                    )
                    continue

                # ------------------------------------------------------
                # LSTM MERGE
                #   1. Busy-wait until both forward and backward flags are set
                #   2. Compute
                #   3. Clean up join_buffer
                #   4. Fall through to generic pipeline-forward
                # ------------------------------------------------------
                elif role == "merge":
                    while True:
                        buf = self.join_buffer.get(task.task_id)
                        if buf and buf["forward"] and buf["backward"]:
                            break
                        yield self.env.timeout(0.01)

                    flops = self.layerFlops[layer_idx]
                    yield self.env.process(device.run(flops=flops))
                    self.stage_metrics_device[device_idx].tasks_processed += 1
                    self.stage_metrics_device[device_idx].busy_time += (
                        flops / self.deviceComputeCapacity[device_idx]
                    )
                    del self.join_buffer[task.task_id]
                    logging.info(
                        f"Task {task.task_id}: merge {device_idx} complete at {self.env.now:.4f}"
                    )
                    # Falls through to PIPELINE FORWARD below

                # ------------------------------------------------------
                # NORMAL layer
                # ------------------------------------------------------
                else:
                    flops = sum(
                        self.layerFlops[l] for l in layers_on_device if isinstance(l, int)
                    )
                    yield self.env.process(device.run(flops=flops))
                    self.stage_metrics_device[device_idx].tasks_processed += 1
                    self.stage_metrics_device[device_idx].busy_time += (
                        flops / self.deviceComputeCapacity[device_idx]
                    )

                # ------------------------------------------------------
                # PIPELINE FORWARD
                # Three cases:
                #   A) Fan-out device → send to BOTH forward and backward branches
                #   B) Normal sequential device → send to next device
                #   C) Last device → mark task complete
                # ------------------------------------------------------
                message_size = self.layersActivationSize[layer_idx]
                device_keys  = list(self.deviceToPartitionMapping.keys())
                pos          = device_keys.index(device_idx)

                if device_idx in self.lstm_groups:
                    # Case A: fan-out — same task goes to both branches in parallel
                    grp = self.lstm_groups[device_idx]
                    for branch_key in ("forward", "backward"):
                        branch_dev_idx = grp[branch_key]
                        branch_coord   = self.get_device_coord(branch_dev_idx)
                        branch_device  = self.grid.nodes[branch_coord]["device"]
                        logging.info(
                            f"Task {task.task_id}: fan-out {device_idx} -> "
                            f"{branch_key} device {branch_dev_idx}"
                        )
                        self.env.process(self.send_message(
                            task, device_coord, branch_coord, message_size, branch_device
                        ))

                elif pos < len(device_keys) - 1:
                    # Case B: normal sequential step — skip forward/backward devices
                    # (they are handled exclusively via fan-out, not sequential indexing)
                    next_device_idx = device_keys[pos + 1]
                    next_role       = get_device_role(
                        self.deviceToPartitionMapping[next_device_idx]
                    )
                    # If the next device in the ordered list is a forward/backward branch,
                    # that means the current device IS the fan-out device but wasn't flagged
                    # in lstm_groups — this shouldn't happen with correct config, but guard it.
                    if next_role in ("forward", "backward"):
                        logging.warning(
                            f"Device {device_idx} sequential-next is a branch device "
                            f"({next_device_idx}, role={next_role}). "
                            f"Check device_to_partition config."
                        )
                    next_coord  = self.get_device_coord(next_device_idx)
                    next_device = self.grid.nodes[next_coord]["device"]
                    self.env.process(self.send_message(
                        task, device_coord, next_coord, message_size, next_device
                    ))

                else:
                    # Case C: last device — task complete
                    task.completion_time = self.env.now
                    self.task_records.append(task)
                    logging.info(
                        f"Task {task.task_id} COMPLETE, latency={task.total_latency:.4f}"
                    )

        # ------------------------------------------------------------------
        def periodic_sampler():
            while True:
                yield self.env.timeout(self.samplingInterval)
                for edge, link_idx in self.edge_to_idx.items():
                    link = self.grid.edges[edge]["link"]
                    queue_len = (
                        len(link.linkResource.queue) +
                        len(link.transmission_queue.items)
                    )
                    self.stage_metrics_link[link_idx].queue_samples.append(queue_len)
                for dev_idx in range(len(self.deviceComputeCapacity)):
                    dev_coord = self.get_device_coord(dev_idx)
                    dev = self.grid.nodes[dev_coord]["device"]
                    queue_len = (
                        len(dev.input_queue.items) + len(dev.compute_resource.queue)
                    )
                    self.stage_metrics_device[dev_idx].queue_samples.append(queue_len)

        # ------------------------------------------------------------------
        self.env.process(task_generator())

        # Start one worker per device that appears in the mapping
        for device_idx in sorted(self.deviceToPartitionMapping.keys()):
            self.env.process(device_worker(device_idx))

        self.env.process(periodic_sampler())

    # ------------------------------------------------------------------
    def results(self) -> Dict:
        """Return a summary dict with all key metrics. Call after simulate()."""
        completed = [r for r in self.task_records if r.completion_time > 0]
        print(f"Tasks completed: {len(completed)}")
        if not completed:
            return {"error": "no tasks completed"}

        latencies = [r.total_latency for r in completed]

        stage_summary = {}
        for p, sm in self.stage_metrics_device.items():
            stage_summary[f"device_{p}"] = {
                "utilization":       sm.utilization(self.simDuration),
                "mean_queue_length": sm.mean_queue_length(),
                "max_queue_length":  sm.max_queue_length(),
                "tasks_processed":   sm.tasks_processed,
            }

        for p, sm in self.stage_metrics_link.items():
            edge = list(self.grid.edges)[p]
            link = self.grid.edges[edge]["link"]
            sm.busy_time = link.busy_time
            stage_summary[f"link_{edge}"] = {
                "utilization":       sm.utilization(self.simDuration),
                "mean_queue_length": sm.mean_queue_length(),
                "max_queue_length":  sm.max_queue_length(),
                "tasks_processed":   sm.tasks_processed,
            }

        bottleneck = max(stage_summary, key=lambda p: stage_summary[p]["utilization"])

        active_devices     = set(self.deviceToPartitionMapping.keys())
        device_queue_means = [
            self.stage_metrics_device[d].mean_queue_length() for d in active_devices
        ]
        mean_compute_queue = float(np.mean(device_queue_means)) if device_queue_means else 0.0

        link_queue_means = [
            sm.mean_queue_length()
            for sm in self.stage_metrics_link.values()
            if sm.tasks_processed > 0
        ]
        mean_comm_queue = float(np.mean(link_queue_means)) if link_queue_means else 0.0

        all_queues         = device_queue_means + link_queue_means
        mean_overall_queue = float(np.mean(all_queues)) if all_queues else 0.0

        return {
            "stage_metrics":        stage_summary,
            "tasks_completed":      len(completed),
            "mean_latency":         float(np.mean(latencies)),
            "p95_latency":          float(np.percentile(latencies, 95)),
            "max_latency":          float(np.max(latencies)),
            "bottleneck_partition": bottleneck,
            "max_utilization":      stage_summary[bottleneck]["utilization"],
            "mean_compute_queue":   mean_compute_queue,
            "mean_comm_queue":      mean_comm_queue,
            "mean_overall_queue":   mean_overall_queue,
        }


# ----------------------------------------------------------------------
if __name__ == "__main__":
    layer_flops      = [10, 20]
    layer_activation = [30, 40]

    grid_rows, grid_cols = 2, 2
    links_bw = {
        ((0, 0), (1, 0)): 5,
        ((0, 0), (0, 1)): 5,
        ((0, 1), (1, 1)): 5,
        ((1, 0), (1, 1)): 5,
    }
    device_caps         = [10, 10, 10, 10]
    device_to_partition = {0: [0], 1: [1]}

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
        sim_duration             = 1000,
        sampling_interval        = 1,
    )
    sim.simulate()
    env.run(until=sim_duration)
    print(sim.results())