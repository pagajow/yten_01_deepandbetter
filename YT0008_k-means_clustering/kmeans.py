import numpy as np
import random

# Generate synthetic test data
def generate_data():
    """
    Generates synthetic 2D data points grouped into three clusters.
    Each cluster is drawn from a normal distribution with specified mean and standard deviation.
    The points are then shuffled randomly to remove ordering.
    """
    np.random.seed(42)  # Set the seed for reproducibility
    # Generate three clusters with different means
    cluster_1 = np.random.normal(loc=(2, 2), scale=0.5, size=(50, 2))
    cluster_2 = np.random.normal(loc=(8, 8), scale=0.5, size=(50, 2))
    cluster_3 = np.random.normal(loc=(5, 12), scale=0.5, size=(50, 2))
    # Stack all clusters into one dataset
    data = np.vstack([cluster_1, cluster_2, cluster_3])
    # Shuffle the rows to mix points from different clusters
    np.random.shuffle(data)
    return data


class KMeans:
    def __init__(self, k, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        
    def _create_clusters(self, data):
        """
        Creates clusters by assigning points to the nearest centroid.
        """
        # Calculate the difference between each point and each centroid
        differences = data[:, None] - self.centroids[None, :]
        
        # Compute Euclidean distances for each pair
        distances = np.linalg.norm(differences, axis=2)
        
        # Determine the closest centroid for each point
        cluster_labels = np.argmin(distances, axis=1)
        
        # Group points into clusters based on their labels
        return [np.where(cluster_labels == i)[0] for i in range(self.k)]
        
    def predict(self, data):
        """
        Assigns each data point to the nearest cluster based on Euclidean distance.
        """
        # Compute differences between all points and centroids
        # Calculate Euclidean distances
        # Return the index of the closest centroid for each point
        return np.argmin(np.linalg.norm(data[:, None] - self.centroids[None, :], axis=2), axis=1)

    def _calculate_wss(self, data, clusters):
        """
        Calculates the Within-Cluster Sum of Squares (WSS) for all clusters.
        """
        wss_list = []
        for i, cluster in enumerate(clusters):
            wss = 0
            if len(cluster) > 0:  # Ensure the cluster is not empty
                cluster_points = data[cluster]  # Extract points in the cluster
                wss = np.sum((cluster_points - self.centroids[i]) ** 2)  # Compute squared distances
            wss_list.append(wss)
        return wss_list

    def fit(self, data):
        """
        Fits the K-means clustering algorithm to the data.
        Randomly initializes centroids and iteratively updates them
        until convergence or reaching the maximum number of iterations.
        """
        history = []
        
        # Randomly select K points from the data as initial centroids
        self.centroids = data[random.sample(range(data.shape[0]), self.k)]
        
        # Iteratively update centroids
        for _ in range(self.max_iters):
            # Group points into clusters based on the closest centroid
            clusters = self._create_clusters(data)
            
            # Calculate Within-Cluster Sum of Squares (WSS) for the clusters
            wss = self._calculate_wss(data, clusters)
            
            # Append iteration data to the history
            history.append({
                "wss": wss,
                "total_wss": sum(wss),
                "predictions": self.predict(data),
                "centroids": self.centroids,
            })
            
            # Compute new centroids as the mean of points in each cluster
            new_centroids = np.array([data[cluster].mean(axis=0) for cluster in clusters])

            # Check for convergence: stop if centroids do not change
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids
 
        return history

    def multiple_run(self, data, n_runs=10):
        """
        Executes multiple runs of the K-means algorithm with different random initializations.
        """
        history = []
        best_total_wss = None
        best_centroids = None
        for _ in range(n_runs):
            self.centroids = None  # Reset centroids for each run
            run_history = self.fit(data)
            total_wss = run_history[-1]["total_wss"] if run_history else 0
            history.append({
                "centroids": self.centroids,
                "run_history": run_history,
                "total_wss": total_wss,
            })
            
            if best_total_wss is None or total_wss < best_total_wss:
                best_total_wss = total_wss
                best_centroids = self.centroids
                
        self.centroids = best_centroids if best_centroids is not None else self.centroids   
        
        return history

    def find_optimal_k(self, data, max_k=10):
        """
        Finds the optimal number of clusters (K) by evaluating the WSS for different values of K.
        """
        # Store the initial state of the model
        stored_k = self.k
        stored_centroids = self.centroids
        history = []  # Store history for each K

        for k in range(1, max_k + 1): # Test K from 1 to max_k
            self.k = k
            self.centroids = None  # Reset centroids before testing
            run_history = self.fit(data)  # Train the model
            total_wss = run_history[-1]["total_wss"] if run_history else 0
            history.append({
                "k": k,
                "total_wss": total_wss
            })

        # Restore the model state
        self.k = stored_k
        self.centroids = stored_centroids

        return history


