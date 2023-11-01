import time
import numpy as np
import multiprocessing as mp
from lithops import FunctionExecutor
from lithops.multiprocessing import Array, Barrier, Lock
from utility_service import sigmoid
from utility_service import get_datasetAndLabels
from utility_service import partition_dataset


class LogisticRegression:

    def __init__(self, learning_rate=0.001, max_iter=100000, n_features=2, n_workers=1, n_processes=1):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.n_features = n_features
        self.n_workers = n_workers
        self.n_processes = n_processes
        self.workers = []

    def launch_worker(self, obj):
        worker_id = obj.part - 1
        self.workers[worker_id].breakdown["launchTime"] = time.time()
        loadDataset_start = time.time()
        X, y = get_datasetAndLabels(obj.data_stream.read().decode("utf-8"))
        loadDataset_end = time.time()
        self.workers[worker_id].set_dataset(X, y)
        self.workers[worker_id].breakdown["loadDataset"] = (loadDataset_end - loadDataset_start)
        return self.workers[worker_id].run()

    def initialize_workers(self):
        for worker_idx in range(self.n_workers):
            worker = Worker(worker_idx, self.n_processes, self.learning_rate, self.max_iter, self.weights, self.gradients, self.barrier, self.lock)
            self.workers.append(worker)

    def fit(self, dataset):
        start_time = time.time()

        self.weights = Array('d', [0 for _ in range(self.n_features + 1)])
        self.gradients = Array('d', [0 for _ in range(self.n_features + 1)])
        self.barrier = Barrier(self.n_workers)
        self.lock = Lock()

        self.initialize_workers()

        fexec = FunctionExecutor()
        launch_start = time.time()
        fexec.map(self.launch_worker, dataset, obj_chunk_number=self.n_workers)
        workers_results = fexec.get_result()
        receivedResultsTime = time.time()

        self.weights = self.weights[:]

        end_time = time.time()
        self.total_duration = end_time - start_time

        times_workers = [worker_results[0] for worker_results in workers_results]
        min_start_time_worker = min([time[0] for time in times_workers])
        max_end_time_worker = max([time[1] for time in times_workers])
        self.total_duration_workers = max_end_time_worker - min_start_time_worker

        breakdown = [worker_results[1] for worker_results in workers_results]

        launchDurations = []
        for breakdown_item in breakdown:
            launchDuration = breakdown_item["launchTime"] - launch_start
            launchDurations.append(launchDuration)

        workers_endTimes = [time[1] for time in times_workers]
        sendResultsDurations = []
        for worker_endTime in workers_endTimes:
            sendResultsDuration = receivedResultsTime - worker_endTime
            sendResultsDurations.append(sendResultsDuration)

        self.breakdown = breakdown[0]
        self.breakdown["launchDuration"] = sum(launchDurations) / len(launchDurations)
        self.breakdown["sendResultsDuration"] = sum(sendResultsDurations) / len(sendResultsDurations)


    def predict(self, X):
        X = np.c_[np.ones((np.shape(X)[0], 1)), X]
        z = np.dot(X, self.weights)
        y_pred = sigmoid(z)
        predictions = [1 if i > 0.5 else 0 for i in y_pred]
        return predictions


class Worker:
    def __init__(self, worker_id, n_processes, learning_rate, max_iter, weights, gradients, barrier, lock):
        self.worker_id = worker_id
        self.n_processes = n_processes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = weights
        self.gradients = gradients
        self.empty_gradients = [0 for _ in range(len(gradients))]
        self.barrier = barrier
        self.lock = lock
        self.times = []  # contains start time and end time of worker
        self.breakdown = {}

    def set_dataset(self, X, y):
        self.X = np.c_[np.ones((np.shape(X)[0], 1)), X]
        self.y = y

    def compute_gradients(self, conn, X, y, weights):
        innerWorker_startExecutionTime = time.time()
        z = np.dot(X, weights)
        y_pred = sigmoid(z)
        errors = y_pred - y
        gradients = np.dot(X.T, errors)
        innerWorker_endExecutionTime = time.time()
        conn.send([gradients, innerWorker_startExecutionTime, innerWorker_endExecutionTime])
        conn.close()

    def run(self):
        print(f"Worker[{self.worker_id}] - execution started")
        self.times.append(time.time())

        self.breakdown["fetch_sharedState"] = 0
        self.breakdown["update_sharedState"] = 0
        self.breakdown["synchronisation"] = 0
        self.breakdown["aggregate_sharedState"] = 0
        self.breakdown["createPipeConnections"] = 0
        self.breakdown["createProcesses"] = 0
        self.breakdown["closePipes"] = 0
        self.breakdown["closeProcesses"] = 0
        self.breakdown["innerWorker_launchDuration"] = 0
        self.breakdown["innerWorker_computeDuration"] = 0
        self.breakdown["innerWorker_sendResultsDuration"] = 0
        self.breakdown["partitionDataset"] = 0
        self.breakdown["aggregateInnerWorkers"] = 0

        partitionDataset_start = time.time()
        partitioned_X = partition_dataset(self.X, self.n_processes)
        partitioned_y = partition_dataset(self.y, self.n_processes)
        partitionDataset_end = time.time()
        self.breakdown["partitionDataset"] += (partitionDataset_end - partitionDataset_start)

        processes = []
        parent_connections = []
        child_connections = []
        processesStartTimes = []
        innerWorker_launchDurations = []
        innerWorker_computeDurations = []
        innerWorker_sendResultsDurations = []

        createPipeConnections_start = time.time()
        for process_idx in range(self.n_processes):
            parent_conn, child_conn = mp.Pipe()
            parent_connections.append(parent_conn)
            child_connections.append(child_conn)
        createPipeConnections_end = time.time()
        self.breakdown["createPipeConnections"] += (createPipeConnections_end - createPipeConnections_start)

        for iter in range(self.max_iter):
            print(f"Worker[{self.worker_id}] - started iteration[{iter}]")

            fetch_sharedState_start = time.time()
            weights_local = self.weights[:]
            fetch_sharedState_end = time.time()
            self.breakdown["fetch_sharedState"] += (fetch_sharedState_end - fetch_sharedState_start)

            processes.clear()
            processesStartTimes.clear()
            for process_idx in range(self.n_processes):
                createProcesses_start = time.time()
                process = mp.Process(target=self.compute_gradients, args=(child_connections[process_idx], partitioned_X[process_idx], partitioned_y[process_idx], weights_local,))
                createProcesses_end = time.time()
                self.breakdown["createProcesses"] += (createProcesses_end - createProcesses_start)
                processes.append(process)
                processesStartTimes.append(time.time())
                process.start()

            gradients_local = 0
            innerWorker_computeDurations.clear()
            innerWorker_sendResultsDurations.clear()
            innerWorker_launchDurations.clear()
            for index, parent_connection in enumerate(parent_connections):
                results = parent_connection.recv()
                innerWorker_receiveResultsTime = time.time()
                innerWorker_gradients = results[0]
                innerWorker_startExecutionTime = results[1]
                innerWorker_endExecutionTime = results[2]
                innerWorker_computeDuration = innerWorker_endExecutionTime - innerWorker_startExecutionTime
                innerWorker_sendResultsDuration = innerWorker_receiveResultsTime - innerWorker_endExecutionTime
                innerWorker_launchDuration = innerWorker_startExecutionTime - processesStartTimes[index]
                innerWorker_computeDurations.append(innerWorker_computeDuration)
                innerWorker_sendResultsDurations.append(innerWorker_sendResultsDuration)
                innerWorker_launchDurations.append(innerWorker_launchDuration)
                aggregateInnerWorkers_start = time.time()
                gradients_local += innerWorker_gradients
                aggregateInnerWorkers_end = time.time()
                self.breakdown["aggregateInnerWorkers"] += (aggregateInnerWorkers_end - aggregateInnerWorkers_start)

            closeProcesses_start = time.time()
            for process in processes:
                process.join()
                process.close()
            closeProcesses_end = time.time()
            self.breakdown["closeProcesses"] += (closeProcesses_end - closeProcesses_start)

            self.breakdown["innerWorker_launchDuration"] += sum(innerWorker_launchDurations) / len(innerWorker_launchDurations)
            self.breakdown["innerWorker_computeDuration"] += sum(innerWorker_computeDurations) / len(innerWorker_computeDurations)
            self.breakdown["innerWorker_sendResultsDuration"] += sum(innerWorker_sendResultsDurations) / len(innerWorker_sendResultsDurations)

            update_sharedState_start = time.time()
            with self.lock:
                self.gradients[:] += gradients_local
            update_sharedState_end = time.time()
            self.breakdown["update_sharedState"] += (update_sharedState_end - update_sharedState_start)

            synchronisation_start = time.time()
            self.barrier.wait()
            synchronisation_end = time.time()
            self.breakdown["synchronisation"] += (synchronisation_end - synchronisation_start)

            aggregate_sharedState_start = time.time()
            if self.worker_id == 0:
                gradients_total = np.array(self.gradients[:])
                weights_local = weights_local - self.learning_rate * gradients_total
                self.weights[:] = weights_local
                self.gradients[:] = self.empty_gradients

            aggregate_sharedState_end = time.time()
            self.breakdown["aggregate_sharedState"] += (aggregate_sharedState_end - aggregate_sharedState_start)

            synchronisation_start = time.time()
            self.barrier.wait()
            synchronisation_end = time.time()
            self.breakdown["synchronisation"] += (synchronisation_end - synchronisation_start)

        closePipes_start = time.time()
        for process_idx in range(self.n_processes):
            parent_connections[process_idx].close()
            child_connections[process_idx].close()
        closePipes_end = time.time()
        self.breakdown["closePipes"] += (closePipes_end - closePipes_start)

        self.times.append(time.time())

        return (self.times, self.breakdown)