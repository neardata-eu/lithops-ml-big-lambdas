import time
import numpy as np
from lithops import FunctionExecutor
from lithops.multiprocessing import Array, Barrier, Lock
from utility_service import sigmoid
from utility_service import get_datasetAndLabels


class LogisticRegression:

    def __init__(self, learning_rate=0.001, max_iter=100000, n_features=2, n_workers=1):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.n_features = n_features
        self.n_workers = n_workers
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
            worker = Worker(worker_idx, self.learning_rate, self.max_iter, self.weights, self.gradients, self.barrier, self.lock)
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
    def __init__(self, worker_id, learning_rate, max_iter, weights, gradients, barrier, lock):
        self.worker_id = worker_id
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

    def run(self):
        print(f"Worker[{self.worker_id}] - execution started")
        self.times.append(time.time())

        self.breakdown["fetch_sharedState"] = 0
        self.breakdown["compute"] = 0
        self.breakdown["update_sharedState"] = 0
        self.breakdown["synchronisation"] = 0
        self.breakdown["aggregate_sharedState"] = 0

        for iter in range(self.max_iter):
            print(f"Worker[{self.worker_id}] - started iteration[{iter}]")

            fetch_sharedState_start = time.time()
            weights_local = self.weights[:]
            fetch_sharedState_end = time.time()
            self.breakdown["fetch_sharedState"] += (fetch_sharedState_end - fetch_sharedState_start)

            compute_start = time.time()
            z = np.dot(self.X, weights_local)
            y_pred = sigmoid(z)
            errors = y_pred - self.y
            gradients_local = np.dot(self.X.T, errors)
            compute_end = time.time()
            self.breakdown["compute"] += (compute_end - compute_start)

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

        self.times.append(time.time())

        return (self.times, self.breakdown)