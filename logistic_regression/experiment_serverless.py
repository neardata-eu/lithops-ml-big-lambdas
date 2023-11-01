import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, required=True)
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--max_iter', type=int, required=True)
    parser.add_argument('--n_features', type=int, required=True)
    parser.add_argument('--n_workers', type=int, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--n_processes', type=int, required=False)
    args = parser.parse_args()

    if args.algorithm == "logisticRegression":
        from logisticRegression import LogisticRegression
        logisticRegression = LogisticRegression(learning_rate=args.learning_rate, max_iter=args.max_iter, n_features=args.n_features, n_workers=args.n_workers)
    elif args.algorithm == "logisticRegression_parallel":
        from logisticRegression_parallel import LogisticRegression
        logisticRegression = LogisticRegression(learning_rate=args.learning_rate, max_iter=args.max_iter, n_features=args.n_features, n_workers=args.n_workers, n_processes=args.n_processes)

    logisticRegression.fit([args.dataset])

    print("\nAlgorithm: " + args.algorithm)
    print("Dataset X: " + str(args.dataset))
    print("Number workers: " + str(args.n_workers))
    print("Number processes: " + str(args.n_processes))
    print("Number features: " + str(args.n_features))
    print("Learning rate: " + str(args.learning_rate))
    print("Max iterations: " + str(args.max_iter))
    print("Total duration: " + str(logisticRegression.total_duration))
    print("Total duration workers: " + str(logisticRegression.total_duration_workers))