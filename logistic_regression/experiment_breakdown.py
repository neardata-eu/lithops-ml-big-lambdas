import argparse
from logisticRegression_breakdown import LogisticRegression

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--max_iter', type=int, required=True)
    parser.add_argument('--n_features', type=int, required=True)
    parser.add_argument('--n_workers', type=int, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    logisticRegression = LogisticRegression(learning_rate=args.learning_rate, max_iter=args.max_iter, n_features=args.n_features, n_workers=args.n_workers)
    logisticRegression.fit([args.dataset])

    print("\nDataset X: " + str(args.dataset))
    print("Number workers: " + str(args.n_workers))
    print("Number features: " + str(args.n_features))
    print("Learning rate: " + str(args.learning_rate))
    print("Max iterations: " + str(args.max_iter))
    print("Total duration: " + str(logisticRegression.total_duration))
    print("Total duration workers: " + str(logisticRegression.total_duration_workers))
    print("Breakdown: " + str(logisticRegression.breakdown))