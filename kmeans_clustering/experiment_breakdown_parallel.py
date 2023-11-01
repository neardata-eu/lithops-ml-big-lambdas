import argparse
from utility_service import get_centroids
from kmeans_parallel_breakdown import KMeans

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--max_iter', type=int, required=True)
    parser.add_argument('--n_workers', type=int, required=True)
    parser.add_argument('--n_processes', type=int, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--centroids_init', type=int, required=False)
    parser.add_argument('--ignore_last_column', type=int, required=False)
    args = parser.parse_args()

    centroids_initialized = (args.centroids_init == 1)
    ignore_last_column = (args.ignore_last_column == 1)
    centroids = get_centroids(centroids_initialized)

    kmeans = KMeans(n_clusters=args.k, max_iter=args.max_iter, n_workers=args.n_workers, n_processes=args.n_processes, init=centroids, ignore_last_column=ignore_last_column)
    kmeans.fit([args.dataset])

    print("\nNumber clusters K: " + str(args.k))
    print("Number workers: " + str(args.n_workers))
    print("Number processes: " + str(args.n_processes))
    print("Centroids initialized: " + str(centroids_initialized))
    print("Ignore last column: " + str(ignore_last_column))
    print("Dataset X: " + str(args.dataset))
    print("Total duration: " + str(kmeans.total_duration))
    print("Total duration workers: " + str(kmeans.total_duration_workers))
    print("Breakdown: " + str(kmeans.breakdown))