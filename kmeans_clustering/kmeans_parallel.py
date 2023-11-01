import time
import numpy as np
import multiprocessing as mp
from lithops import FunctionExecutor
from lithops.multiprocessing import Array, Barrier, RawValue, Lock
from utility_service import get_dataset
from utility_service import partition_dataset

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, n_workers=1, n_processes=1, init=None, ignore_last_column=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_workers = n_workers
        self.n_processes = n_processes
        self.workers = []
        self.init = init
        self.centroids_initialized = (self.init is not None)
        self.ignore_last_column = ignore_last_column

    def launch_worker(self, obj):
        worker_id = obj.part - 1
        dataset = get_dataset(obj.data_stream.read().decode("utf-8"), self.ignore_last_column)
        self.workers[worker_id].set_dataset(dataset)
        return self.workers[worker_id].run()

    def initialize_workers(self):
        locks_clusters = [Lock() for _ in range(self.n_clusters)]
        for worker_idx in range(self.n_workers):
            worker = Worker(worker_idx, self.n_processes, self.n_clusters, self.max_iter, self.centroids_initialized, self.cluster_centers, self.clusters_totals, self.clusters_counters, self.barrier, self.has_converged, locks_clusters)
            self.workers.append(worker)

    def fit(self, X):
        start_time = time.time()

        self.clusters_totals = Array('d', [0 for _ in range(self.n_clusters)])
        self.clusters_counters = Array('i', [0 for _ in range(self.n_clusters)])
        self.has_converged = RawValue('i', 0)
        self.barrier = Barrier(self.n_workers)

        if self.centroids_initialized == False:
            self.cluster_centers = Array('d', [0 for _ in range(self.n_clusters)])
        else:
            self.cluster_centers = Array('d', self.init)

        self.initialize_workers()

        fexec = FunctionExecutor()
        fexec.map(self.launch_worker, X, obj_chunk_number=self.n_workers)
        workers_results = fexec.get_result()

        self.labels_ = np.concatenate([worker_results[0] for worker_results in workers_results], axis=0)
        self.cluster_centers_ = np.array(self.cluster_centers[:])

        end_time = time.time()
        self.total_duration = end_time - start_time

        times_workers = [worker_results[1] for worker_results in workers_results]
        min_start_time_worker = min([time[0] for time in times_workers])
        max_end_time_worker = max([time[1] for time in times_workers])
        self.total_duration_workers = max_end_time_worker - min_start_time_worker

        return self


class Worker:
    def __init__(self, worker_id, n_processes, n_clusters, max_iter, centroids_initialized, cluster_centers, clusters_totals, clusters_counters, barrier, has_converged, locks_clusters):
        self.worker_id = worker_id
        self.n_processes = n_processes
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids_initialized = centroids_initialized
        self.cluster_centers = cluster_centers
        self.clusters_totals = clusters_totals
        self.clusters_counters = clusters_counters
        self.barrier = barrier
        self.has_converged = has_converged
        self.locks_clusters = locks_clusters
        self.times = []  # contains start time and end time of worker
        self.X = None
        self.labels = None

    def set_dataset(self, X):
        self.X = X

    def compute_clusters(self, conn, X, cluster_centers):
        labels = []
        clusters_counters = [0 for _ in range(len(cluster_centers))]
        clusters_totals = [0 for _ in range(len(cluster_centers))]
        for x_idx, x_value in enumerate(X):
            cluster_idx = np.argmin(((cluster_centers - x_value) ** 2).sum(axis=1))
            labels.append(cluster_idx)
            clusters_counters[cluster_idx] += 1
            clusters_totals[cluster_idx] += x_value
        conn.send([labels, clusters_counters, clusters_totals])
        conn.close()

    def run(self):
        print(f"Worker[{self.worker_id}] - execution started")
        self.times.append(time.time())

        if self.centroids_initialized == False:
            if self.worker_id == 0:
                initial_cluster_centers_idx = np.random.choice(len(self.X), self.n_clusters, replace=False)
                initial_cluster_centers = [self.X[idx] for idx in initial_cluster_centers_idx]
                self.cluster_centers[:] = initial_cluster_centers
            self.barrier.wait()

        clusters_counters_local = [0 for _ in range(self.n_clusters)]
        clusters_totals_local = [0 for _ in range(self.n_clusters)]

        partitioned_X = partition_dataset(self.X, self.n_processes)

        labels_local = []
        processes = []
        parent_connections = []
        child_connections = []

        for process_idx in range(self.n_processes):
            parent_conn, child_conn = mp.Pipe()
            parent_connections.append(parent_conn)
            child_connections.append(child_conn)

        for iter in range(self.max_iter):
            print(f"Worker[{self.worker_id}] - started iteration[{iter}]")
            cluster_centers_local = self.cluster_centers[:]

            processes.clear()
            for process_idx in range(self.n_processes):
                process = mp.Process(target=self.compute_clusters, args=(child_connections[process_idx], partitioned_X[process_idx], cluster_centers_local,))
                processes.append(process)
                process.start()

            labels_local.clear()
            for parent_connection in parent_connections:
                results = parent_connection.recv()
                labels_local.extend(results[0])
                for cluster_idx in range(self.n_clusters):
                    clusters_counters_local[cluster_idx] += results[1][cluster_idx]
                    clusters_totals_local[cluster_idx] += results[2][cluster_idx]

            for process in processes:
                process.join()
                process.close()

            for cluster_idx in range(self.n_clusters):
                with self.locks_clusters[cluster_idx]:
                    self.clusters_counters[cluster_idx] += clusters_counters_local[cluster_idx]
                    self.clusters_totals[cluster_idx] += clusters_totals_local[cluster_idx]
                clusters_counters_local[cluster_idx] = 0
                clusters_totals_local[cluster_idx] = 0

            self.barrier.wait()

            if self.worker_id == 0:
                cluster_centers_new = []
                for cluster_idx in range(self.n_clusters):
                    clusters_counters = self.clusters_counters[cluster_idx]
                    clusters_totals = self.clusters_totals[cluster_idx]
                    if clusters_counters > 0:
                        cluster_centers_new.append(clusters_totals / clusters_counters)
                    else:
                        cluster_centers_new.append(cluster_centers_local[cluster_idx])
                    self.cluster_centers[cluster_idx] = cluster_centers_new[cluster_idx]
                    self.clusters_counters[cluster_idx] = 0
                    self.clusters_totals[cluster_idx] = 0

                if np.allclose(cluster_centers_new, cluster_centers_local):
                    self.has_converged.value = 1

            self.barrier.wait()

            if self.has_converged.value == 1:
                break

        for process_idx in range(self.n_processes):
            parent_connections[process_idx].close()
            child_connections[process_idx].close()

        self.labels = np.array(labels_local, dtype=int)

        self.times.append(time.time())

        return (self.labels, self.times)