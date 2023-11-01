The Logistic Regression scripts are described in the following:

- **logisticRegression.py** - serverless implementation of the Logistic Regression algorithm based on Lithops

- **logisticRegression_parallel.py** - implementation with inner workers of the algorithm from <em>logisticRegression.py</em>

- **logisticRegression_breakdown.py** - implementation which includes the breakdown of execution times for the algorithm from <em>logisticRegression.py</em>

- **logisticRegression_parallel_breakdown.py** - implementation which includes the breakdown of execution times for the algorithm from <em>logisticRegression_parallel.py</em>

- **experiment_serverless.py** - executes the serverless Logistic Regression algorithm with or without inner workers and outputs the total execution times of the algorithm and of the workers

- **experiment_breakdown.py** - executes <em>logisticRegression_breakdown.py</em> and outputs the breakdown and the total execution times of the algorithm and of the workers

- **experiment_breakdown_parallel.py** - executes <em>logisticRegression_parallel_breakdown.py</em> and outputs the breakdown and the total execution times of the algorithm and of the workers

- **utility_service.py** - includes utility functions used by the Logistic Regression algorithms
