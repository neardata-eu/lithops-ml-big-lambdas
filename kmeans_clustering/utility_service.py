import math
import numpy as np

def get_dataset(data, ignore_last_column):
    data_lines = data.splitlines()
    number_samples = len(data_lines)
    number_dimensions = len(data_lines[0].split(','))
    if ignore_last_column:
        number_dimensions -= 1
    dataset = np.zeros([number_samples, number_dimensions])
    for line_idx, line in enumerate(data_lines):
        dimensions = line.split(',')
        dataset[line_idx] = np.array(list(map(lambda x: float(x), dimensions[:number_dimensions])))
    return dataset

def get_datasetList(data):
    dataset = []
    data_lines = data.splitlines()
    for line in data_lines:
        dimensions = line.split(',')
        dataset_item = np.array(list(map(lambda x: float(x), dimensions)))
        dataset.append(dataset_item)
    return dataset

def partition_dataset(x, size):
    size = math.ceil(len(x)/size)
    return np.split(x, np.arange(size, len(x), size))

def get_centroids(centroids_initialized):
    if centroids_initialized:
        with open('centroids.csv') as f:
            centroids_data = f.read()
            return get_datasetList(centroids_data)
    return None