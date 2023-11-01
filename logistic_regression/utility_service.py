import math
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def partition_dataset(x, size):
    size = math.ceil(len(x)/size)
    return np.split(x, np.arange(size, len(x), size))

def get_datasetAndLabels(data):
    data_lines = data.splitlines()
    number_samples = len(data_lines)
    number_dimensions = len(data_lines[0].split(','))
    dataset = np.zeros([number_samples, number_dimensions-1])
    labels = []
    for line_idx, line in enumerate(data_lines):
        dimensions = line.split(',')
        dataset[line_idx] = np.array(list(map(lambda x: float(x), dimensions[:number_dimensions-1])))
        labels.append(dimensions[-1])
    return dataset, np.array(labels, dtype="float64")