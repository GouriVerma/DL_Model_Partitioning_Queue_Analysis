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
    return num_elements * dtype_bytes


# ---------------------------------------------------------------------------
# Data-recording helpers
# ---------------------------------------------------------------------------

@dataclass
class TaskRecord:
    """Stores timing breakdowns for one completed task."""
    task_id:         int
    arrival_time:    float
    origin:          Tuple[int, int] = (0, 0)
    completion_time: float = 0.0
    stage_breakdown: List[Tuple] = field(default_factory=list)

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


# ---------------------------------------------------------------------------
# Hardware primitives
# ---------------------------------------------------------------------------

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
            logging.info(
                f"Message gets link resource for task {message.task_id} at {self.env.now}"
            )
            yield self.env.timeout(delay)
            self.busy_time += delay
            yield self.transmission_queue.put(message)

    def put(self, message_size, message):
        self.env.process(self.latency(message, message_size))

    def get(self):
        return self.transmission_queue.get()


# ---------------------------------------------------------------------------
# Main simulator
# ---------------------------------------------------------------------------

class Simulator:
    """
    Pipeline-parallel simulator supporting **multiple devices per partition**.

    Parameters
    ----------
    partitionToDevicesMapping : dict[int, list[int]]
        Maps each partition index to a list of device indices assigned to it.
        Example (2 partitions, 2 devices each on a 2×2 grid):
            {0: [0, 1], 1: [2, 3]}
    dispatch_policy : str
        How to pick which replica handles the next task.
        'round_robin'  – rotate evenly regardless of load.
        'least_loaded' – always pick the replica whose input_queue is shortest.
    """

    def __init__(
        self,
        numLayers,
        layersFlops,
        layersActivationSize,
        gridSize,
        partitionToDevicesMapping: Dict[int, List[int]],
        deviceComputeCapacity,
        linksBandwidth,
        env,
        arrival_rate=1,
        default_bw=1,
        sim_duration=50,
        sampling_interval=1,
        dispatch_policy="least_loaded",
    ):
        self.numLayers = numLayers
        self.layerFlops = layersFlops
        self.layersActivationSize = layersActivationSize
        self.gridSize = gridSize
        self.partitionToDevicesMapping = partitionToDevicesMapping   # {partition: [device_idx, ...]}
        self.deviceComputeCapacity = deviceComputeCapacity
        self.linksBandwidth = linksBandwidth
        self.arrivalRate = arrival_rate
        self.simDuration = sim_duration
        self.samplingInterval = sampling_interval
        self.dispatch_policy = dispatch_policy

        self.env = env

        # ---- Build flat device→partition reverse map ----
        self.device_to_partition: Dict[int, int] = {}
        for part_idx, dev_list in partitionToDevicesMapping.items():
            for dev_idx in dev_list:
                self.device_to_partition[dev_idx] = part_idx

        # ---- Ordered partition list ----
        self.partition_order = sorted(partitionToDevicesMapping.keys())

        # ---- Build grid ----
        self.grid = networkx.grid_2d_graph(gridSize[0], gridSize[1])
        self.device_coords = list(self.grid.nodes)

        for idx, coord in enumerate(self.device_coords):
            self.grid.nodes[coord]["device"] = Device(
                compute_cap=deviceComputeCapacity[idx], env=self.env
            )

        for edge in self.grid.edges:
            bw = linksBandwidth.get(edge, default_bw)
            self.grid.edges[edge]["link"] = Link(bw, env)

        # ---- Edge index helpers ----
        self.edge_to_idx = {
            normalize_edge(u, v): i for i, (u, v) in enumerate(self.grid.edges())
        }

        # ---- Per-partition dispatch round-robin counters ----
        self._rr_counter: Dict[int, int] = {p: 0 for p in self.partition_order}

        # ---- Metrics ----
        self.task_records: List[TaskRecord] = []
        self.stage_metrics_device: Dict[int, StageMetrics] = {
            d: StageMetrics(partition_idx=d) for d in range(len(deviceComputeCapacity))
        }
        self.stage_metrics_link: Dict[int, StageMetrics] = {
            p: StageMetrics(partition_idx=p) for p in range(self.grid.number_of_edges())
        }

        self._validate_configuration()

        # ---- Logging ----
        logging.info("Partition to Devices Mapping:")
        for part_idx, dev_list in partitionToDevicesMapping.items():
            for d in dev_list:
                coord = self.get_device_coord(d)
                logging.info(f"  Partition {part_idx}: device {d} at {coord}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_device_coord(self, device_idx: int):
        return self.device_coords[device_idx]

    def get_device(self, device_idx: int) -> Device:
        return self.grid.nodes[self.get_device_coord(device_idx)]["device"]

    def _pick_replica(self, partition_idx: int) -> int:
        """Return the device_idx of the chosen replica for *partition_idx*."""
        replicas = self.partitionToDevicesMapping[partition_idx]
        if len(replicas) == 1:
            return replicas[0]

        if self.dispatch_policy == "round_robin":
            chosen = replicas[self._rr_counter[partition_idx] % len(replicas)]
            self._rr_counter[partition_idx] += 1
            return chosen

        # Default: least_loaded (queue length + compute queue)
        def load(dev_idx):
            dev = self.get_device(dev_idx)
            return len(dev.input_queue.items) + len(dev.compute_resource.queue)

        loads = {dev_idx: load(dev_idx) for dev_idx in replicas}
        min_load = min(loads.values())
        candidates = [dev_idx for dev_idx, cur_load in loads.items() if cur_load == min_load]
        chosen = candidates[self._rr_counter[partition_idx] % len(candidates)]
        self._rr_counter[partition_idx] += 1
        return chosen

    # ------------------------------------------------------------------
    # Network transmission
    # ------------------------------------------------------------------

    def send_message(self, task, src_coord, dest_coord, message_size, dest_device):
        """Route task from src_coord to dest_coord along shortest path."""
        logging.info(
            f"Task {task.task_id}: routing {src_coord} → {dest_coord} at {self.env.now}"
        )
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
                raise ValueError(f"No link for edge ({cur}, {nxt})")

            link.put(message_size, current_task)
            edge_key = normalize_edge(cur, nxt)
            link_idx = self.edge_to_idx[edge_key]
            self.stage_metrics_link[link_idx].tasks_processed += 1

            current_task = yield link.get()
            logging.info(f"Task {current_task.task_id}: hop {cur}→{nxt} done at {self.env.now}")

        yield dest_device.input_queue.put(current_task)
        logging.info(f"Task {task.task_id}: delivered to {dest_coord} at {self.env.now}")

    # ------------------------------------------------------------------
    # Simulation processes
    # ------------------------------------------------------------------

    def simulate(self):
        """Spawn all SimPy processes."""

        # ---- Task generator ----
        def task_generator():
            first_partition = self.partition_order[0]
            task_id = 0
            while True:
                inter_arrival = np.random.exponential(1.0 / self.arrivalRate)
                yield self.env.timeout(inter_arrival)

                record = TaskRecord(
                    task_id=task_id,
                    origin=(0, 0),
                    arrival_time=self.env.now,
                )

                # Dispatch to a replica within the first partition
                chosen_dev = self._pick_replica(first_partition)
                dest_device = self.get_device(chosen_dev)
                yield dest_device.input_queue.put(record)
                logging.info(
                    f"Task {task_id} → partition {first_partition} device {chosen_dev} at {self.env.now}"
                )
                task_id += 1

        # ---- Per-device worker ----
        def device_worker(device_idx: int):
            partition_idx = self.device_to_partition[device_idx]
            device = self.get_device(device_idx)
            device_coord = self.get_device_coord(device_idx)

            # Layers assigned to this *partition* (all replicas share the same layers)
            # For now, partitions share a single set of layers; you can extend this
            # to per-device layer lists if needed.
            # We derive partition → layers from the inverse mapping.
            layers_on_partition = self._partition_layers(partition_idx)
            total_flops = sum(self.layerFlops[l] for l in layers_on_partition)
            last_layer  = max(layers_on_partition)
            message_size = self.layersActivationSize[last_layer]

            # Index of this partition in the pipeline order
            part_pos = self.partition_order.index(partition_idx)
            is_last_partition = (part_pos == len(self.partition_order) - 1)

            while True:
                task = yield device.input_queue.get()
                logging.info(
                    f"Device {device_idx} (partition {partition_idx}): "
                    f"received task {task.task_id} at {self.env.now}"
                )

                # ---- Compute ----
                wait_time = yield self.env.process(device.run(flops=total_flops))
                self.stage_metrics_device[device_idx].tasks_processed += 1
                self.stage_metrics_device[device_idx].total_wait_time += wait_time
                self.stage_metrics_device[device_idx].busy_time += (
                    total_flops / self.deviceComputeCapacity[device_idx]
                )
                task.stage_breakdown.append(
                    (
                        "compute",
                        partition_idx,
                        device_idx,
                        wait_time,
                        total_flops / self.deviceComputeCapacity[device_idx],
                        self.env.now,
                    )
                )
                logging.info(
                    f"Device {device_idx}: computed task {task.task_id} at {self.env.now}"
                )

                if is_last_partition:
                    # ---- Pipeline complete ----
                    task.completion_time = self.env.now
                    logging.info(
                        f"Task {task.task_id} completed at {self.env.now:.4f}, "
                        f"latency {task.total_latency:.4f}"
                    )
                    self.task_records.append(task)
                else:
                    # ---- Forward to next partition ----
                    next_partition = self.partition_order[part_pos + 1]
                    next_dev_idx   = self._pick_replica(next_partition)
                    next_dev       = self.get_device(next_dev_idx)
                    next_coord     = self.get_device_coord(next_dev_idx)

                    self.env.process(
                        self.send_message(
                            task, device_coord, next_coord, message_size, next_dev
                        )
                    )

        # ---- Periodic queue sampler ----
        def periodic_sampler():
            while True:
                yield self.env.timeout(self.samplingInterval)
                # Sample link queues
                for edge, link_idx in self.edge_to_idx.items():
                    link = self.grid.edges[edge]["link"]
                    queue_len = (
                        len(link.linkResource.queue)
                        + len(link.transmission_queue.items)
                    )
                    self.stage_metrics_link[link_idx].queue_samples.append(queue_len)
                # Sample device queues
                for dev_idx in range(len(self.deviceComputeCapacity)):
                    dev = self.get_device(dev_idx)
                    queue_len = (
                        len(dev.input_queue.items)
                        + len(dev.compute_resource.queue)
                    )
                    self.stage_metrics_device[dev_idx].queue_samples.append(queue_len)

        # ---- Start processes ----
        self.env.process(task_generator())
        for dev_idx in sorted(self.device_to_partition.keys()):
            self.env.process(device_worker(dev_idx))
        self.env.process(periodic_sampler())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _partition_layers(self, partition_idx: int) -> List[int]:
        """
        Return the list of layer indices assigned to *partition_idx*.

        We derive this from the flattened device→layers information embedded in
        the original per-partition layer assignment.  The caller is expected to
        supply a ``layerToPartitionMapping`` kwarg **or** we fall back to
        evenly splitting layers across partitions.
        """
        if hasattr(self, "_layer_partition_map"):
            return self._layer_partition_map.get(partition_idx, [])

        # Fallback: evenly distribute layers across partitions
        n_parts = len(self.partition_order)
        n_layers = self.numLayers
        chunk = n_layers // n_parts
        remainder = n_layers % n_parts
        start = 0
        mapping = {}
        for i, p in enumerate(self.partition_order):
            extra = 1 if i < remainder else 0
            mapping[p] = list(range(start, start + chunk + extra))
            start += chunk + extra
        self._layer_partition_map = mapping
        return mapping[partition_idx]

    def _validate_configuration(self):
        num_devices = len(self.deviceComputeCapacity)
        grid_devices = len(self.device_coords)
        if num_devices != grid_devices:
            raise ValueError(
                f"deviceComputeCapacity has {num_devices} entries, but gridSize={self.gridSize} "
                f"creates {grid_devices} devices."
            )

        seen_devices = set()
        for part_idx, dev_list in self.partitionToDevicesMapping.items():
            if not dev_list:
                raise ValueError(f"Partition {part_idx} has no devices assigned.")
            for dev_idx in dev_list:
                if dev_idx < 0 or dev_idx >= num_devices:
                    raise ValueError(
                        f"Partition {part_idx} references invalid device index {dev_idx}."
                    )
                if dev_idx in seen_devices:
                    raise ValueError(
                        f"Device {dev_idx} is assigned to more than one partition."
                    )
                seen_devices.add(dev_idx)

        for part_idx in self.partition_order:
            layers = self._partition_layers(part_idx)
            if not layers:
                raise ValueError(
                    f"Partition {part_idx} has no layers assigned. "
                    "Provide an explicit mapping or reduce the number of partitions."
                )
            for layer_idx in layers:
                if layer_idx < 0 or layer_idx >= self.numLayers:
                    raise ValueError(
                        f"Partition {part_idx} references invalid layer index {layer_idx}."
                    )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def results(self) -> Dict:
        """Return summary metrics. Call after env.run()."""
        completed = [r for r in self.task_records if r.completion_time > 0]
        if not completed:
            return {"error": "no tasks completed"}

        latencies = [r.total_latency for r in completed]

        stage_summary: Dict = {}

        # ---- Device metrics ----
        for dev_idx, sm in self.stage_metrics_device.items():
            part_idx = self.device_to_partition.get(dev_idx, -1)
            stage_summary[f"device_{dev_idx}_partition_{part_idx}"] = {
                "utilization":       sm.utilization(self.simDuration),
                "mean_wait_time":    (sm.total_wait_time / sm.tasks_processed) if sm.tasks_processed else 0.0,
                "mean_queue_length": sm.mean_queue_length(),
                "max_queue_length":  sm.max_queue_length(),
                "tasks_processed":   sm.tasks_processed,
            }

        # ---- Per-partition aggregate metrics ----
        for part_idx, dev_list in self.partitionToDevicesMapping.items():
            total_tasks = sum(
                self.stage_metrics_device[d].tasks_processed for d in dev_list
            )
            mean_util = float(np.mean([
                self.stage_metrics_device[d].utilization(self.simDuration)
                for d in dev_list
            ]))
            stage_summary[f"partition_{part_idx}_aggregate"] = {
                "devices":           dev_list,
                "total_tasks":       total_tasks,
                "mean_utilization":  mean_util,
            }

        # ---- Link metrics ----
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

        # ---- Bottleneck (highest device utilisation) ----
        device_keys = [k for k in stage_summary if k.startswith("device_")]
        bottleneck = max(device_keys, key=lambda k: stage_summary[k]["utilization"])

        # ---- Queue summary ----
        active_devices = list(self.device_to_partition.keys())
        device_queue_means = [
            self.stage_metrics_device[d].mean_queue_length() for d in active_devices
        ]
        link_queue_means = [
            self.stage_metrics_link[p].mean_queue_length()
            for p in self.stage_metrics_link
            if self.stage_metrics_link[p].tasks_processed > 0
        ]
        all_queues = device_queue_means + link_queue_means

        return {
            "stage_metrics":         stage_summary,
            "tasks_completed":       len(completed),
            "mean_latency":          float(np.mean(latencies)),
            "p95_latency":           float(np.percentile(latencies, 95)),
            "max_latency":           float(np.max(latencies)),
            "bottleneck_device":     bottleneck,
            "max_utilization":       stage_summary[bottleneck]["utilization"],
            "mean_compute_queue":    float(np.mean(device_queue_means)) if device_queue_means else 0.0,
            "mean_comm_queue":       float(np.mean(link_queue_means))   if link_queue_means   else 0.0,
            "mean_overall_queue":    float(np.mean(all_queues))         if all_queues         else 0.0,
        }


# ---------------------------------------------------------------------------
# Example: provide explicit layer→partition mapping
# ---------------------------------------------------------------------------

class SimulatorWithLayerMap(Simulator):
    """
    Thin subclass that lets callers pass an explicit
    ``partitionToLayersMapping`` dict instead of relying on the fallback
    even-split heuristic.

    Parameters
    ----------
    partitionToLayersMapping : dict[int, list[int]]
        e.g. {0: [0, 1], 1: [2, 3]}
    """

    def __init__(self, partitionToLayersMapping: Dict[int, List[int]], **kwargs):
        super().__init__(**kwargs)
        self._layer_partition_map = partitionToLayersMapping


# ---------------------------------------------------------------------------
# Demo / __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # Model definition
    # -----------------------------------------------------------------------
    layer_flops      = [10, 20, 15, 25]      # 4 layers
    layer_activation = [30, 40, 35, 45]

    # -----------------------------------------------------------------------
    # Grid: 5×5 = 25 devices
    #   layout (row, col):
    #   (0,0) (0,1) (0,2) (0,3) (0,4)
    #   (1,0) (1,1) (1,2) (1,3) (1,4)
    #   (2,0) (2,1) (2,2) (2,3) (2,4)
    #   (3,0) (3,1) (3,2) (3,3) (3,4)
    #   (4,0) (4,1) (4,2) (4,3) (4,4)
    # device indices (networkx node ordering, row-major):
    #   0  1  2  3  4
    #   5  6  7  8  9
    #  10 11 12 13 14
    #  15 16 17 18 19
    #  20 21 22 23 24
    # -----------------------------------------------------------------------
    grid_rows, grid_cols = 5, 5

    device_caps = [10] * (grid_rows * grid_cols)   # uniform compute

    links_bw = {}   # leave empty → default_bw used for all edges

    # -----------------------------------------------------------------------
    # Use all 25 devices by splitting the grid into 4 adjacent partitions.
    # Each partition owns one model layer, and neighboring partitions stay
    # physically close to reduce communication distance between stages.
    # -----------------------------------------------------------------------
    partition_to_devices = {
        0: [0, 1, 2, 3, 4, 5, 6],
        1: [7, 8, 9, 10, 11, 12],
        2: [13, 14, 15, 16, 17, 18],
        3: [19, 20, 21, 22, 23, 24],
    }

    # Explicit layer assignment per partition: one layer per partition
    partition_to_layers = {
        0: [0],
        1: [1],
        2: [2],
        3: [3],
    }

    sim_duration = 1000

    env = simpy.Environment()
    sim = SimulatorWithLayerMap(
        partitionToLayersMapping = partition_to_layers,
        numLayers                = len(layer_flops),
        layersFlops              = layer_flops,
        layersActivationSize     = layer_activation,
        gridSize                 = (grid_rows, grid_cols),
        partitionToDevicesMapping= partition_to_devices,
        deviceComputeCapacity    = device_caps,
        linksBandwidth           = links_bw,
        env                      = env,
        arrival_rate             = 0.5,
        default_bw               = 5,
        sim_duration             = sim_duration,
        sampling_interval        = 1,
        dispatch_policy          = "least_loaded",
    )

    sim.simulate()
    env.run(until=sim_duration)

    import json
    res = sim.results()
    print(json.dumps(res, indent=2))
