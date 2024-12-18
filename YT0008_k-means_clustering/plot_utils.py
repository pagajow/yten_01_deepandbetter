import matplotlib.pyplot as plt
import numpy as np

# Plot the clustering results
def plot_clusters(data, labels=None, centroids=None):
    """
    Visualizes the clustered data points and their centroids on a 2D scatter plot.

    Parameters:
    - data (ndarray): The dataset containing points as rows.
    - labels (ndarray): Cluster labels for each data point.
    - centroids (ndarray): The coordinates of the centroids for the clusters.
    """
    if labels is None:
        plt.scatter(data[:, 0], data[:, 1], c="#1f66e5", s=50)  # Plot data points
    else:
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50)  # Plot data points with colors based on cluster labels
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')  # Highlight centroids
    plt.title('K-means Clustering')  # Add a title to the plot
    plt.xlabel('X1')  # Label for the X-axis
    plt.ylabel('X2')  # Label for the Y-axis
    plt.legend()  # Add a legend for centroids
    plt.show()  # Display the plot


# Plot the WSS values to find the optimal number of clusters (Elbow Method)
def plot_optimal_k(history):
    """
    Plots the Within-Cluster Sum of Squares (WSS) for different values of K 
    to help identify the optimal number of clusters using the Elbow Method.

    Parameters:
    - history (list): A list of dictionaries containing 'k' (number of clusters) 
      and 'total_wss' (sum of WSS for all clusters).
    """
    k = [r["k"] for r in history]  # Extract the number of clusters
    wss = [r["total_wss"] for r in history]  # Extract the total WSS for each K
    plt.plot(k, wss, marker='o')  # Plot WSS vs. number of clusters with markers
    plt.xlabel("Number of Clusters (k)")  # Label for the X-axis
    plt.ylabel("Within-Cluster Sum of Squares (WSS)")  # Label for the Y-axis
    plt.grid()  # Add a grid for better readability
    plt.xticks(ticks=k)  # Set the ticks for the X-axis as the values of K

    # Adjust the Y-axis ticks to have a consistent step size
    step = 1000  # Step size for WSS
    limit = int(max(wss) / step) + 1  # Calculate the number of ticks based on the max WSS
    plt.yticks(ticks=[step * i for i in range(limit)])  # Set the ticks for the Y-axis
    plt.show()  # Display the plot


def plot_fit_history(data, history, max_cols=4):
    """
    Plots clustering results from each iteration stored in the history.
    Each subplot corresponds to one iteration, arranged dynamically in a grid.

    Parameters:
    - data (ndarray): The dataset containing points as rows.
    - history (list): A list of dictionaries where each entry represents 
      one iteration and contains 'predictions' (cluster labels) 
      and 'centroids' (coordinates of centroids).
    - max_cols (int): Maximum number of columns in the grid layout.
    """
    num_plots = len(history)  # Total number of plots
    num_cols = min(max_cols, num_plots)  # Determine the number of columns
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate the number of rows (ceil division)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    axes = np.array(axes).flatten()  # Flatten axes for easy iteration

    for idx, (ax, hist) in enumerate(zip(axes, history)):
        predictions = hist["predictions"]  # Extract cluster labels for this iteration
        centroids = hist["centroids"]  # Extract centroids for this iteration

        # Plot data points with cluster colors
        ax.scatter(data[:, 0], data[:, 1], c=predictions, cmap='viridis', s=50)
        # Plot centroids
        ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')
        ax.set_title(f'Iteration {idx + 1}')  # Title indicating iteration number
        ax.set_xlabel('X1')  # Label for the X-axis
        ax.set_ylabel('X2')  # Label for the Y-axis
        ax.legend()  # Add legend for centroids
        ax.grid()  # Add a grid for better readability

    # Hide any unused subplot axes
    for ax in axes[len(history):]:
        ax.axis('off')
    
    plt.tight_layout()  # Adjust layout to avoid overlapping text
    plt.show()  # Display the plot


def plot_multiple_run_history(history):
    """
    Plots a bar chart of the WSS values for each run in the multiple_run history.
    
    Parameters:
    - history (list): A list of dictionaries from the `multiple_run` function,
                      where each dictionary contains 'total_wss' and other details of a single run.
    """
    # Extract WSS values for each run
    wss_values = [run["total_wss"] for run in history]
    
    # Generate bar chart
    plt.figure(figsize=(10, 6))  # Set figure size
    plt.bar(range(len(wss_values)), wss_values, color='skyblue', edgecolor='black')
    plt.xlabel("Run Number")  # Label for the X-axis
    plt.ylabel("Total WSS")  # Label for the Y-axis
    plt.title("WSS for Each Run in Multiple K-Means Runs")  # Add a title
    plt.xticks(range(len(wss_values)), [f"Run {i+1}" for i in range(len(wss_values))], rotation=45)  # Label each bar
    
    # Add values on top of the bars
    for i, wss in enumerate(wss_values):
        plt.text(i, wss + max(wss_values) * 0.01, f"{wss:.2f}", ha='center', fontsize=9)
    
    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.show()  # Display the plot


