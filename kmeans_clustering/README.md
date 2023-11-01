The K-Means scripts are described in the following:

- **kmeans.py** - serverless implementation of the K-Means algorithm based on Lithops

- **kmeans_parallel.py** - implementation with inner workers of the algorithm from <em>kmeans.py</em>

- **kmeans_breakdown.py** - implementation which includes the breakdown of execution times for the algorithm from <em>kmeans.py</em>

- **kmeans_parallel_breakdown.py** - implementation which includes the breakdown of execution times for the algorithm from <em>kmeans_parallel.py</em>

- **experiment_serverless.py** - executes the serverless K-Means algorithm with or without inner workers and outputs the total execution times of the algorithm and of the workers

- **experiment_breakdown.py** - executes <em>kmeans_breakdown.py</em> and outputs the breakdown and the total execution times of the algorithm and of the workers

- **experiment_breakdown_parallel.py** - executes <em>kmeans_parallel_breakdown.py</em> and outputs the breakdown and the total execution times of the algorithm and of the workers

- **utility_service.py** - includes utility functions used by the K-Means algorithms
