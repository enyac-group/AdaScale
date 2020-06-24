import sys
import cPickle
import numpy as np
import matplotlib.pyplot as plt

def expected_latency(optimal_file):
    """
    Take an optimal.pkl as input and output theoretical latency.
    Notice that the data structure in optimal.pkl should be a dictionary
    dict({int(image_id): tuple(shortest side, longest side)})
    Args:
        optimal_file (str): The path to find input pickle file
    Returns:
        None
    """

    profiled_latency = [20.9, 22.9, 26.4, 31.7, 38.7, 45.2, 49.6, 60.8, 69.6, 77.2]

    with open(optimal_file) as f:
        d = cPickle.load(f)

    res = []
    for key in d:
        res.append(d[key][0])
    hist = np.histogram(res)
    expected_runtime = np.matmul(hist[0]/float(len(res)), profiled_latency)
    print('Expected Runtime: {:.2f} ms'.format(expected_runtime))
    plt.Figure(dpi=600)
    plt.bar(np.arange(hist[0].shape[0])+0.1, hist[0]/float(len(res)), align='edge')
    plt.xticks(np.arange(hist[1].shape[0]), ['{:.2f}'.format(x) for x in hist[1]], rotation=30)
    plt.xlabel('Shortest side of the image')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(sys.argv[2])
    plt.close()

expected_latency(sys.argv[1])