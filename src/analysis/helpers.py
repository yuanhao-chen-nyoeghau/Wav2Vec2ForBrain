import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import describe


def show_spike_hist(data, data_name: str, log_scale: bool = False):
    if log_scale:
        data = np.log1p(data)

    # Create a histogram
    plt.hist(data, bins=30, density=True, alpha=0.7, color="blue", edgecolor="black")

    # Add labels and title
    plt.xlabel("Spike Power")
    plt.ylabel("Frequency")
    plt.title(
        f"{'Log-Scale-' if log_scale else ''}Distribution of Spike Power over all nodes {data_name}"
    )

    plt.show()


def show_statistics(data, data_name):
    describe_result = describe(data)

    print(f"Summary - {data_name} set")
    print(f"Min: {describe_result.minmax[0]}, Max: {describe_result.minmax[1]}")
    print(f"Mean: {describe_result.mean}")
    print(f"Var: {describe_result.variance}")
