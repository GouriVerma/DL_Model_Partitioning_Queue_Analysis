import simpy
import networkx
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union

import logging
logging.basicConfig(filename='simulator_rgg.log', level=logging.DEBUG, filemode='w')

def normalize_edge(u, v):
    return tuple(sorted([u, v]))

def activation_size_bytes(shape, dtype_bytes=4):
    num_elements = np.prod(shape)
    return (num_elements * dtype_bytes)

def build_rgg(n_nodes: int = 100, radius: float = 0.9, seed: int = 42, sqarea_side_len = 1) -> tuple[networkx.Graph, np.ndarray]:
    """
    Build a Random Geometric Graph.
    Nodes are placed uniformly in [0,1]^2; edges connect pairs within `radius`.
    Retries with slightly larger radius until the graph is connected.
    Returns (graph, positions array of shape [n, 2]).
    """
    rng = np.random.default_rng(seed)
    r = radius
    for _ in range(20):
        pos_array = rng.uniform(0, sqarea_side_len, size=(n_nodes, 2))
        G = networkx.random_geometric_graph(n_nodes, r, pos=dict(enumerate(map(tuple, pos_array))), seed=seed)
        if networkx.is_connected(G):
            return G, pos_array
        r *= 1.05
    raise RuntimeError(f"Could not build a connected RGG after 20 retries (final r={r:.3f})")

@dataclass
class TaskRecord:
    """Stores timing breakdowns for one completed task."""
    task_id:                int
    arrival_time:           float
    origin:                 Union[int, Tuple[int, int]] = 0
    completion_time:        float = 0.0
    # per-partition: (partition_idx, compute_wait, compute_service, tx_wait, tx_service)
    stage_breakdown:        List[Tuple] = field(default_factory=list)
    compute_delay:          float = 0.0
    communication_delay:    float = 0.0
    next_partition_idx:     int = 0
 
    @property
    def total_latency(self):
        return self.completion_time - self.arrival_time
 
 
@dataclass
class StageMetrics:
    """Running stats for one partition stage."""
    partition_idx:   int
    busy_time:       float = 0.0        # total seconds the compute slot was in use
    total_wait_time: float = 0.0        # total time tasks spent waiting (not being served)
    tasks_processed: int   = 0
    queue_samples:   List[float] = field(default_factory=list)  # sampled queue lengths
 
    def utilization(self, sim_duration: float) -> float:
        return self.busy_time / sim_duration if sim_duration > 0 else 0.0
 
    def mean_queue_length(self) -> float:
        return float(np.mean(self.queue_samples)) if self.queue_samples else 0.0
    
    def max_queue_length(self) -> float:
        return float(np.max(self.queue_samples)) if self.queue_samples else 0.0
 


class Device:

    def __init__(self, compute_cap, env):
        self.compute_capacity = compute_cap
        self.compute_resource = simpy.Resource(env,capacity=1)
        self.input_queue = simpy.Store(env) # infinite capacity
        self.generated_tasks_queue = simpy.Store(env) # infinite capacity
        self.env = env
    
    def run(self, flops=0):
        delay = flops/self.compute_capacity
        start_wait = self.env.now
        with self.compute_resource.request() as req:
            yield req
            wait_time = self.env.now - start_wait
            yield self.env.timeout(delay)
        return wait_time

class Link:
    
    def __init__(self,bandwidth,env):
        self.bandwidth = bandwidth
        self.transmission_queue = simpy.Store(env)
        self.env = env
        self.linkResource = simpy.Resource(env, capacity=1)
        self.busy_time = 0.0  # Track actual transmission time
    
    def latency(self, message, message_size):
        delay = message_size/self.bandwidth
        with self.linkResource.request() as req:
            yield req
            logging.info(f"Message gets the link resource for task {message.task_id} at {self.env.now}")
            yield self.env.timeout(delay)
            self.busy_time += delay  # Add actual transmission time
            message.communication_delay += delay
            yield self.transmission_queue.put(message)

    def put(self, message_size, message):
        self.env.process(self.latency(message, message_size))

    def get(self):
        return self.transmission_queue.get()

class Simulator:

    def __init__(self, numLayers, layersFlops, layersActivationSize, numNodes, deviceToPartitionMapping, deviceComputeCapacity, env, partition: Optional[Union[Dict[int, List[int]], List[List[int]]]]=None, arrival_rate=1, default_bw=1, sim_duration=50, sampling_interval=1, task_generating_device_ids=None, input_task_size=32, global_poisson_stream=False, random_seed: Optional[int]=None, next_device_assignment_policy="random", next_device_random_seed: Optional[int]=None, rgg_random_seed: Optional[int]=None, comm_radius=0.1, sqarea_side_len=1):
        self.numLayers = numLayers
        self.layerFlops = layersFlops
        self.layersActivationSize = layersActivationSize
        # self.gridSize = gridSize
        self.deviceComputeCapacity = deviceComputeCapacity # Rowwise 1D array

        self.partitions = self._normalize_partitions(partition, deviceToPartitionMapping)
        self.numPartitions = len(self.partitions)
        self.deviceToPartitionMapping = self._normalize_device_partition_mapping(deviceToPartitionMapping) # {device_id: [partition_idx, ...]}
        self.partitionToDevices = self._build_partition_to_devices(self.deviceToPartitionMapping)
        self.partitionFlops = {
            p_idx: sum(self.layerFlops[layer_idx] for layer_idx in layers)
            for p_idx, layers in self.partitions.items()
        }
        
        self.arrivalRate = arrival_rate
        self.simDuration = sim_duration
        self.samplingInterval = sampling_interval
        self.taskGeneratingDeviceIds = task_generating_device_ids
        self.nextTaskId = 0
        self.totalTasksGenerated = 0
        self.initialTaskMessageSize = input_task_size
        self.globalPoissonStream = global_poisson_stream # False then each device in the task_generating_device_ids generates task independently in its tasks queue, if true; global poisson stream is used to generate task and put into the queue of any of the device in task generating devices
        self.randomSeed = random_seed
        self.rng = np.random.default_rng(self.randomSeed)
        self.nextDeviceAssignmentPolicy = next_device_assignment_policy
        self.nextDeviceRandomSeed = next_device_random_seed if next_device_random_seed is not None else self.randomSeed
        self.nextDeviceRng = np.random.default_rng(self.nextDeviceRandomSeed)
        self.rggRandomSeed = rgg_random_seed if rgg_random_seed is not None else self.randomSeed
        self.env = env
        self.sqareaSideLen = sqarea_side_len

        self._rr_counter: Dict[int, int] = {p: 0 for p in range(self.numPartitions)}

        # Initialise Grid
        # self.grid = networkx.grid_2d_graph(gridSize[0], gridSize[1])
        self.commRadius = comm_radius
        # self.grid = networkx.random_geometric_graph(numNodes,comm_radius,seed=self.rggRandomSeed)
        # if not networkx.is_connected(self.grid):
        #     raise ValueError(
        #         "Random geometric graph is disconnected. "
        #         "Increase comm_radius, change rgg_random_seed, or use a larger numNodes."
        #     )
        self.grid, self.pos_array = build_rgg(numNodes, comm_radius, self.rggRandomSeed, self.sqareaSideLen)
        # print(list(self.grid.edges()))

        # Initialise Devices
        for idx, coord in enumerate(list(self.grid.nodes)):
            self.grid.nodes[coord]["device"] = Device(compute_cap=deviceComputeCapacity[idx], env=self.env)
    
        # Initialise Links
        for edge in self.grid.edges:
            self.grid.edges[edge]["link"] = Link(default_bw, env)
        
        # Create reverse mapping: layer -> partition
        self.layer_to_partition = {}
        for p_idx, layers in self.partitions.items():
            for layer in layers:
                self.layer_to_partition[layer] = p_idx
        
        # Log the device to partition mapping with coordinates
        logging.info("Device to Partition Mapping:")
        for device_idx, partition_idx in self.deviceToPartitionMapping.items():
            coord = self.get_device_coord(device_idx)
            layers = self.partitions[partition_idx]
            logging.info(f"  Device {device_idx} at {coord}: partition {partition_idx}, layers {layers}")
        
        
        # Create edges to index dict
        # self.edge_to_idx = {edge: i for i, edge in enumerate(self.grid.edges())}
        self.edge_to_idx = {
            normalize_edge(u,v): i for i, (u,v) in enumerate(self.grid.edges())
        }
        

        self.task_records:  List[TaskRecord]          = []
        self.stage_metrics_device: Dict[int, StageMetrics]   = {
            p: StageMetrics(partition_idx=p) for p in range(len(deviceComputeCapacity))
        }
        self.stage_metrics_link: Dict[int, StageMetrics]   = {
            p: StageMetrics(partition_idx=p) for p in range(self.grid.number_of_edges())
        }

        if self.taskGeneratingDeviceIds is None:
            self.taskGeneratingDeviceIds = [0]

        # first_device_layers = self.deviceToPartitionMapping[self.first_partition_device_idx]
        # first_layer_idx = min(first_device_layers) if first_device_layers else 0
        # self.initial_task_message_size = self.layersActivationSize[first_layer_idx] if self.layersActivationSize else 0

        for device_idx in self.taskGeneratingDeviceIds:
            if device_idx < 0 or device_idx >= len(self.grid.nodes):
                raise ValueError(f"Invalid task generating device id: {device_idx}")

    def _normalize_partitions(self, partition_input, device_to_partition_input):
        """
        Normalize partition definition to {partition_idx: [layer_idx, ...]} with sequential partition_idx.
        Accepts:
          - dict like {0:[0,1], 1:[2,3]}
          - list like [[0,1], [2,3]]
        If not provided, attempts backward-compatible inference from device mapping values if they look like layer lists.
        """
        if partition_input is None:
            if not isinstance(device_to_partition_input, dict):
                raise ValueError("partition must be provided when deviceToPartitionMapping is not a dict")
            inferred = []
            seen = set()
            for _, value in sorted(device_to_partition_input.items(), key=lambda x: x[0]):
                if isinstance(value, int):
                    continue
                if isinstance(value, list) and value and all(isinstance(x, int) for x in value):
                    key = tuple(value)
                    if key not in seen:
                        seen.add(key)
                        inferred.append(list(value))
            if not inferred:
                raise ValueError("Could not infer partitions. Provide partition input explicitly.")
            partition_input = inferred

        if isinstance(partition_input, dict):
            ordered = [list(partition_input[k]) for k in sorted(partition_input.keys())]
        elif isinstance(partition_input, list):
            ordered = [list(p) for p in partition_input]
        else:
            raise ValueError("partition must be either dict[int, list[int]] or list[list[int]]")

        partitions = {idx: layers for idx, layers in enumerate(ordered)}
        for idx, layers in partitions.items():
            if not layers:
                raise ValueError(f"Partition {idx} is empty")
            if not all(isinstance(layer, int) for layer in layers):
                raise ValueError(f"Partition {idx} contains non-integer layer IDs")
        return partitions

    def _normalize_device_partition_mapping(self, mapping_input):
        """
        Normalize device mapping to {device_id: partition_idx}.
        Each device holds exactly one partition.
        Accepts mapping values as:
          - partition id (integer), e.g. 0, 1, 2
        """
        if not isinstance(mapping_input, dict):
            raise ValueError("deviceToPartitionMapping must be a dict")

        partition_ids = set(self.partitions.keys())
        normalized = {}

        for device_idx, partition_idx in mapping_input.items():
            if device_idx < 0 or device_idx >= len(self.deviceComputeCapacity):
                raise ValueError(f"Invalid device id in mapping: {device_idx}")

            if not isinstance(partition_idx, int):
                raise ValueError(
                    f"Device {device_idx} mapping must be a partition index (integer), got {type(partition_idx).__name__}"
                )

            if partition_idx not in partition_ids:
                raise ValueError(f"Device {device_idx} references unknown partition {partition_idx}")

            normalized[device_idx] = partition_idx

        return normalized

    def _build_partition_to_devices(self, device_to_partition):
        """Build reverse mapping: partition_idx -> list of device_ids that hold it."""
        partition_to_devices = {p: [] for p in self.partitions.keys()}
        for device_idx, partition_idx in device_to_partition.items():
            partition_to_devices[partition_idx].append(device_idx)

        missing = [p for p, devices in partition_to_devices.items() if not devices]
        if missing:
            raise ValueError(f"No device assigned to partition(s): {missing}")

        return partition_to_devices
    
    def get_device_coord(self, device_idx):
        """Get grid coordinate of device by its index."""
        coords = list(self.grid.nodes)
        return coords[device_idx]

    def _get_next_task_id(self):
        task_id = self.nextTaskId
        self.nextTaskId += 1
        self.totalTasksGenerated += 1
        return task_id
    
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

    def _select_next_device_for_partition(self, device_idx, next_partition_idx):
        replicas = self.partitionToDevices[next_partition_idx]
        if len(replicas) == 1:
            return replicas[0]
        
        if self.nextDeviceAssignmentPolicy=="random":
            next_device_idx = int(self.nextDeviceRng.choice(self.partitionToDevices[next_partition_idx]))
            return next_device_idx
        elif self.nextDeviceAssignmentPolicy=="nn":
            next_partition_devices = self.partitionToDevices[next_partition_idx]
            current_coord = self.get_device_coord(device_idx)

            # Choose device(s) with minimum hop distance from current device.
            min_hops = None
            nearest_devices = []
            for candidate_device_idx in next_partition_devices:
                candidate_coord = self.get_device_coord(candidate_device_idx)
                try:
                    hops = networkx.shortest_path_length(self.grid, source=current_coord, target=candidate_coord)
                except networkx.NetworkXNoPath:
                    continue

                if (min_hops is None) or (hops < min_hops):
                    min_hops = hops
                    nearest_devices = [candidate_device_idx]
                elif hops == min_hops:
                    nearest_devices.append(candidate_device_idx)

            if not nearest_devices:
                raise ValueError(
                    f"No reachable device found for next partition {next_partition_idx} from device {device_idx}."
                )

            # Tie-break among equally near devices in a reproducible way.
            next_device_idx = int(self.nextDeviceRng.choice(nearest_devices))
            return next_device_idx
            
        elif self.nextDeviceAssignmentPolicy=="rr":
            chosen_device = replicas[self._rr_counter[next_partition_idx] % len(replicas)]
            self._rr_counter[next_partition_idx] += 1
            return chosen_device

        elif self.nextDeviceAssignmentPolicy=="leastload":
            def load(device_idx):
                device_coord = self.get_device_coord(device_idx)
                device = self.grid.nodes[device_coord]["device"]
                return len(device.input_queue.items) + len(device.compute_resource.queue)

            loads = {dev_idx: load(dev_idx) for dev_idx in replicas}
            min_load = min(loads.values())
            candidates = [dev_idx for dev_idx, cur_load in loads.items() if cur_load == min_load]
            chosen = candidates[self._rr_counter[next_partition_idx] % len(candidates)]
            # chosen = self.nextDeviceRng.choice(candidates)
            self._rr_counter[next_partition_idx] += 1
            return chosen
        else:
            print("Setting Next Device Assignment Policy to Default i.e Random")
            next_device_idx = int(self.nextDeviceRng.choice(self.partitionToDevices[next_partition_idx]))
            return next_device_idx
        

    def _task_generator_global(self):
        """
        Generate tasks with a single global Poisson arrival stream,
        randomly assigned to one of the source devices.
        """
        while True:
            # Exponential inter-arrival time (global Poisson process)
            inter_arrival = self.rng.exponential(1.0 / self.arrivalRate)
            yield self.env.timeout(inter_arrival)
            
            # Randomly select a source device for this task
            source_device_idx = self.rng.choice(self.taskGeneratingDeviceIds)
            source_device_coord = self.get_device_coord(source_device_idx)
            source_device = self.grid.nodes[source_device_coord]["device"]
            
            task_id = self._get_next_task_id()
            record = TaskRecord(
                task_id      = task_id,
                origin       = source_device_coord,
                arrival_time = self.env.now,
            )
            yield source_device.generated_tasks_queue.put(record)
            logging.info(f"Task {task_id} generated (global Poisson), assigned to device {source_device_idx} ({source_device_coord}) at time {self.env.now}")
    
    def send_message(self, task, src_coord, dest_coord, message_size, dest_device):
        """Asynchronously transmit a task from source to destination through the shortest path."""
        logging.info(f"Message for task {task.task_id} need to start sending from {src_coord} to {dest_coord} at {self.env.now}")

        # For Multihop
        try:
            path = networkx.shortest_path(self.grid, src_coord, dest_coord)
        except networkx.NetworkXNoPath as e:
            raise ValueError(f"No route from {src_coord} to {dest_coord} for task {task.task_id}") from e
        
        current_task = task
        for i in range(len(path) - 1):
            current_coord = path[i]
            next_coord = path[i + 1]

            if (current_coord, next_coord) in self.grid.edges:
                link = self.grid.edges[(current_coord, next_coord)]["link"]
            else:
                link = self.grid.edges[(next_coord, current_coord)]["link"]
            
            if link is None:
                raise ValueError(f"No link defined for edge {(current_coord, next_coord)}")

            # Put the task on the current link and wait until it arrives at the next node
            logging.info(f"Message for task {current_task.task_id} start sending from {current_coord} to {next_coord} at {self.env.now}")
            link.put(message_size, current_task)

            edge_key = normalize_edge(current_coord, next_coord)
            link_idx = self.edge_to_idx[edge_key]
            # self.stage_metrics_link[link_idx].busy_time += message_size/link.bandwidth
            self.stage_metrics_link[link_idx].tasks_processed += 1
            current_task = yield link.get()
            logging.info(f"Message for task {current_task.task_id} reached from {current_coord} to {next_coord} at {self.env.now}")

        # After the last hop, put the task into the destination device queue
        yield dest_device.input_queue.put(current_task)
        logging.info(f"Message for task {task.task_id} reached from {src_coord} to {dest_coord} at {self.env.now}")
    
    def simulate(self):
        """Main simulation loop that processes tasks through the network."""
        # arrival_rate = 1.0  # Tasks per unit time (Poisson parameter)
        
        def task_generator(device_idx):
            """Generate tasks with Poisson arrivals on the given source device."""
            device_coord = self.get_device_coord(device_idx)
            device = self.grid.nodes[device_coord]["device"]

            while True:
                # Exponential inter-arrival time (Poisson process)
                inter_arrival = self.rng.exponential(1.0 / self.arrivalRate)
                yield self.env.timeout(inter_arrival)
                
                # Put generated task into source device's generated tasks queue
                task_id = self._get_next_task_id()
                record = TaskRecord(
                    task_id      = task_id,
                    origin       = device_coord,
                    arrival_time = self.env.now,
                )
                yield device.generated_tasks_queue.put(record)
                logging.info(f"Task {task_id} generated at device {device_idx} ({device_coord}) at time {self.env.now}")

        def generated_task_worker(device_idx):
            """Handle generated tasks for a device and route them to the first partition device."""
            device_coord = self.get_device_coord(device_idx)
            device = self.grid.nodes[device_coord]["device"]
            first_partition_idx = 0
            first_partition_devices = self.partitionToDevices[first_partition_idx]

            while True:
                task = yield device.generated_tasks_queue.get()
                logging.info(f"Device {device_idx}: Picked generated task {task.task_id} at {self.env.now}")

                first_device_idx = self._select_next_device_for_partition(device_idx, 0)
                # first_device_idx = int(self.rng.choice(first_partition_devices))
                first_device_coord = self.get_device_coord(first_device_idx)
                first_device = self.grid.nodes[first_device_coord]["device"]
                task.next_partition_idx = 0

                if device_idx == first_device_idx:
                    yield device.input_queue.put(task)
                    logging.info(f"Device {device_idx}: Enqueued generated task {task.task_id} directly to input queue at {self.env.now}")
                else:
                    self.env.process(self.send_message(
                        task,
                        device_coord,
                        first_device_coord,
                        self.initialTaskMessageSize,
                        first_device
                    ))
                    logging.info(f"Device {device_idx}: Forwarded generated task {task.task_id} to first partition device {first_device_idx} at {self.env.now}")
        
        def device_worker(device_idx):
            """Worker process that handles tasks on a specific device."""
            device_coord = self.get_device_coord(device_idx)
            device = self.grid.nodes[device_coord]["device"]
            partition_idx = self.deviceToPartitionMapping.get(device_idx)
            if partition_idx is None:
                return
            
            while True:
                # Get task from input buffer
                task = yield device.input_queue.get()
                logging.info(f"Device {device_idx}: Received task {task.task_id}  at {self.env.now}")

                if task.next_partition_idx != partition_idx:
                    raise RuntimeError(
                        f"Task {task.task_id} reached device {device_idx}, but expects partition {task.next_partition_idx} (device holds partition {partition_idx})"
                    )

                layers_on_device = self.partitions[partition_idx]
                total_flops = self.partitionFlops[partition_idx]

                queue_length = len(device.input_queue.items)
                logging.info(f"Device {device_idx} Queue Length {queue_length}")
                # self.stage_metrics_device[device_idx].queue_samples.append(queue_length)
                
                # Request compute resource and execute computation
                wait_time = yield self.env.process(device.run(flops=total_flops))
                # self.stage_metrics_device[device_idx].total_wait_time += wait_time
                self.stage_metrics_device[device_idx].tasks_processed += 1
                self.stage_metrics_device[device_idx].busy_time += total_flops/self.deviceComputeCapacity[device_idx]
                task.compute_delay += total_flops/self.deviceComputeCapacity[device_idx]
                logging.info(f"Device {device_idx}: Computed task {task.task_id}  at {self.env.now}")
                
                # Send output to next partition if not the last one
                task.next_partition_idx += 1

                if task.next_partition_idx < self.numPartitions:
                    next_partition_idx = task.next_partition_idx
                    # next_device_idx = int(self.rng.choice(self.partitionToDevices[next_partition_idx]))
                    next_device_idx = self._select_next_device_for_partition(device_idx, next_partition_idx)
                    next_device_coord = self.get_device_coord(next_device_idx)
                    next_device = self.grid.nodes[next_device_coord]["device"]
                    
                    # Send activation from last layer of current device
                    last_layer_on_device = max(layers_on_device)
                    message_size = self.layersActivationSize[last_layer_on_device]
                    self.env.process(self.send_message(
                        task,
                        device_coord,
                        next_device_coord,
                        message_size,
                        next_device
                    ))
                else:
                    # Task completed at final device - print results
                    task_completion_time = self.env.now - task.arrival_time
                    task.completion_time = self.env.now
                    logging.info(f"Task {task.task_id} completed at time {self.env.now:.2f}, latency: {task_completion_time:.2f}")
                    self.task_records.append(task)
        
        def periodic_sampler():
            while True:
                yield self.env.timeout(self.samplingInterval)
                # Sample link queues
                for edge, link_idx in self.edge_to_idx.items():
                    link = self.grid.edges[edge]["link"]
                    # queue_len = len(link.linkResource.queue)
                    queue_len = (
                        len(link.linkResource.queue)     # waiting for transmission
                        # + len(link.transmission_queue.items) # finished transmission but not consumed
                    )
                    self.stage_metrics_link[link_idx].queue_samples.append(queue_len)
                # Sample device queues (unbiased)
                for device_idx in range(len(self.deviceComputeCapacity)):
                    device_coord = self.get_device_coord(device_idx)
                    device = self.grid.nodes[device_coord]["device"]
                    # queue_len = len(device.input_queue.items)
                    queue_len = len(device.input_queue.items) + len(device.compute_resource.queue) + len(device.generated_tasks_queue.items)
                    self.stage_metrics_device[device_idx].queue_samples.append(queue_len)


        # Start the task generator process on selected source devices
        if self.globalPoissonStream:
            # Single global Poisson stream, randomly assigned to sources
            self.env.process(self._task_generator_global())
        else:
            # Per-device independent Poisson streams
            for source_device_idx in self.taskGeneratingDeviceIds:
                self.env.process(task_generator(source_device_idx))

        # Start generated-task handlers for all devices
        for device_idx in range(len(self.grid.nodes)):
            self.env.process(generated_task_worker(device_idx))
        
        # Start worker processes for each device that holds a partition
        for device_idx in sorted(self.deviceToPartitionMapping.keys()):
            self.env.process(device_worker(device_idx))
        
        self.env.process(periodic_sampler())  
    

    def results(self) -> Dict:
        """
        Return a summary dict with all key metrics.
        Call after simulate().
        """
        print(len(self.task_records))
        completed = [r for r in self.task_records if r.completion_time > 0]
        if not completed:
            return {"error": "no tasks completed"}
 
        latencies = [r.total_latency for r in completed]
        compute_delays = [r.compute_delay for r in completed]
        communication_delays = [r.communication_delay for r in completed]
 
        stage_summary = {}
        for p, sm in self.stage_metrics_device.items():
            stage_summary[f"device_{p}"] = {
                "utilization":       sm.utilization(self.simDuration),
                "mean_queue_length": sm.mean_queue_length(),
                "max_queue_length":  sm.max_queue_length(),
                "tasks_processed":   sm.tasks_processed,
                # "mean_wait_time":    (sm.total_wait_time / sm.tasks_processed
                #                       if sm.tasks_processed else 0),
            }
        
        for p, sm in self.stage_metrics_link.items():
            edge = list(self.grid.edges)[p]
            link = self.grid.edges[edge]["link"]
            sm.busy_time = link.busy_time  # Update with actual transmission time
            stage_summary[f"link_{edge}"] = {
                "utilization":       sm.utilization(self.simDuration),
                "mean_queue_length": sm.mean_queue_length(),
                "max_queue_length":  sm.max_queue_length(),
                "tasks_processed":   sm.tasks_processed,
                # "mean_wait_time":    (sm.total_wait_time / sm.tasks_processed
                #                       if sm.tasks_processed else 0),
            }
 
        # Identify bottleneck (highest utilisation) across devices and links
        bottleneck = max(stage_summary, key=lambda p: stage_summary[p]["utilization"])

        # -------- Compute queue metrics --------
        active_devices = set(self.deviceToPartitionMapping.keys())

        device_queue_means = [
            self.stage_metrics_device[d].mean_queue_length()
            for d in active_devices
        ]

        mean_compute_queue = (
            float(np.mean(device_queue_means)) if device_queue_means else 0.0
        )

        # -------- Communication queue metrics --------
        link_queue_means = []
        for p, sm in self.stage_metrics_link.items():
            if sm.tasks_processed > 0:  # only used links
                link_queue_means.append(sm.mean_queue_length())

        mean_comm_queue = (
            float(np.mean(link_queue_means)) if link_queue_means else 0.0
        )


        # -------- Overall queue --------
        all_queues = device_queue_means + link_queue_means

        mean_overall_queue = (
            float(np.mean(all_queues)) if all_queues else 0.0
        )
 
        return {
        "stage_metrics":       stage_summary,
        "total_tasks_generated": self.totalTasksGenerated,
        "tasks_completed":     len(completed),
        "mean_latency":        float(np.mean(latencies)),
        "p95_latency":         float(np.percentile(latencies, 95)),
        "max_latency":         float(np.max(latencies)),
        "mean_compute_delay":  float(np.mean(compute_delays)),
        "mean_communication_delay":  float(np.mean(communication_delays)),
        "bottleneck_partition": bottleneck,
        "max_utilization":     stage_summary[bottleneck]["utilization"],
        "mean_compute_queue":  mean_compute_queue,
        "mean_comm_queue":     mean_comm_queue,
        "mean_overall_queue":  mean_overall_queue,
    }


if __name__ == "__main__":
    layer_flops = [
        10,20
    ]
    layer_activation = [
        30,40
    ]
 
    # RGG with 4 nodes
    num_devices = 100

    # For RGG, edge keys depend on generated topology; use default_bw unless explicitly mapping by node-id edge pairs.
    links_bw = {}
    device_caps = [10, 10, 10, 10]  # capacity for all 4 devices
    device_caps = [10]*num_devices
    # print(device_caps)
    partition = [[0], [1]] # list of list of layers in partition
    device_to_partition = {0: 0, 1: 1}  # device_id: partition_idx
    
    sim_duration = 1000

    env = simpy.Environment()
    sim = Simulator(
        numLayers               = len(layer_flops),  # Fixed: use actual number of layers
        layersFlops             = layer_flops,
        layersActivationSize    = layer_activation,
        numNodes                = num_devices,
        partition               = partition,
        deviceToPartitionMapping= device_to_partition,
        deviceComputeCapacity   = device_caps,
        env                     = env,
        arrival_rate            = 0.2,
        sim_duration            = sim_duration,
        sampling_interval       = 1,
        task_generating_device_ids = [0,54,90],
        input_task_size         = 5,
        global_poisson_stream   = False,
        random_seed             = 42,
        next_device_assignment_policy='nn',
        default_bw              = 5,
        comm_radius             = 0.9,
        rgg_random_seed         = 42,
    )
 
    sim.simulate()
    env.run(until=sim_duration)
    print(sim.results())
 
 
    