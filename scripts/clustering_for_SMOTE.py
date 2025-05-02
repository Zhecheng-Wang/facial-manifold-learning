from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from utils import load_dataset, load_blendshape



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
from mpl_toolkits.mplot3d import Axes3D
from yellowbrick.cluster import SilhouetteVisualizer, KElbowVisualizer, InterclusterDistance
import warnings
import polyscope as ps
import polyscope.imgui as psim
warnings.filterwarnings('ignore')

from scripts.plot_heads import *



class KMeansAnalyzer:
    def __init__(self, n_clusters=5, use_minibatch=True, batch_size=256, random_state=42):
        """
        Initialize KMeansAnalyzer with user-specified parameters
        
        Parameters:
        -----------
        n_clusters : int, default=5
            Number of clusters to find
        use_minibatch : bool, default=True
            Whether to use MiniBatchKMeans for partial fitting
        batch_size : int, default=256
            Batch size for MiniBatchKMeans
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.use_minibatch = use_minibatch
        self.batch_size = batch_size
        self.random_state = random_state
        
        if use_minibatch:
            self.kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=batch_size,
                random_state=random_state
            )
        else:
            self.kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=random_state
            )
        
        self.scaler = StandardScaler()
        self.data = None
        self.labels = None
        
    def fit_from_torch_loader(self, dataloader):
        """
        Fit KMeans model using a PyTorch DataLoader for incremental learning
        
        Parameters:
        -----------
        dataloader : torch.utils.data.DataLoader
            DataLoader containing the dataset batches
        """
        collected_data = []

        # If using MiniBatchKMeans, we'll fit incrementally
        if self.use_minibatch:
            print("Fitting MiniBatchKMeans incrementally...")
            batch_count = 0
            for batch_x, _, _ in dataloader:  # We ignore labels if any
                batch_numpy = batch_x.numpy()
                
                # If data is multi-dimensional, flatten it
                if len(batch_numpy.shape) > 2:
                    batch_numpy = batch_numpy.reshape(batch_numpy.shape[0], -1)
                
                # Collect data for later use
                collected_data.append(batch_numpy)
                
                # Partial fit
                self.kmeans.partial_fit(batch_numpy)
                batch_count += 1
                if batch_count % 10 == 0:
                    print(f"Processed {batch_count} batches...")
        
        # For standard KMeans, we'll collect all data first
        else:
            print("Collecting data for standard KMeans...")
            for batch_x, _ in dataloader:
                batch_numpy = batch_x.numpy()
                
                # If data is multi-dimensional, flatten it
                if len(batch_numpy.shape) > 2:
                    batch_numpy = batch_numpy.reshape(batch_numpy.shape[0], -1)
                
                collected_data.append(batch_numpy)
            
            # Concatenate all batches
            all_data = np.vstack(collected_data)
            print(f"Fitting KMeans on {all_data.shape[0]} samples...")
            self.kmeans.fit(all_data)
        
        # Store full dataset for later analysis
        self.data = np.vstack(collected_data)
        self.labels = self.kmeans.predict(self.data)
        return self
    
    def fit_from_numpy(self, X):
        """
        Fit KMeans model using a numpy array directly
        
        Parameters:
        -----------
        X : numpy.ndarray
            Data array with shape (n_samples, n_features)
        """
        # If data is multi-dimensional, flatten it
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        
        # Store data
        self.data = X
        
        # Fit KMeans
        if self.use_minibatch:
            # For MiniBatchKMeans, we'll split the data manually
            n_samples = X.shape[0]
            for i in range(0, n_samples, self.batch_size):
                batch = X[i:min(i+self.batch_size, n_samples)]
                self.kmeans.partial_fit(batch)
        else:
            # For standard KMeans, we'll fit all at once
            self.kmeans.fit(X)
        
        # Get cluster labels
        self.labels = self.kmeans.predict(X)
        return self
    
    def plot_cluster_distribution(self, figsize=(10, 6)):
        """
        Plot the distribution of data points across clusters
        """
        plt.figure(figsize=figsize)
        counts = np.bincount(self.labels)
        sns.barplot(x=np.arange(len(counts)), y=counts)
        plt.title('Distribution of Data Points Across Clusters')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Data Points')
        plt.xticks(np.arange(len(counts)))
        plt.tight_layout()
        plt.show()
    
    def plot_centroids_heatmap(self, figsize=(12, 8)):
        """
        Create a heatmap of the cluster centroids to visualize feature importance
        """
        centroids = self.kmeans.cluster_centers_
        plt.figure(figsize=figsize)
        sns.heatmap(centroids, cmap='viridis', annot=False)
        plt.title('Cluster Centroids Feature Values')
        plt.xlabel('Feature Index')
        plt.ylabel('Cluster')
        plt.tight_layout()
        plt.show()
    
    def plot_silhouette(self, figsize=(10, 8)):
        """
        Plot silhouette analysis to evaluate cluster quality
        """
        plt.figure(figsize=figsize)
        visualizer = SilhouetteVisualizer(self.kmeans, colors='yellowbrick')
        visualizer.fit(self.data)
        visualizer.show()
    
    def plot_inertia_vs_k(self, max_k=10, figsize=(10, 6)):
        """
        Plot the elbow method to find the optimal number of clusters
        
        Parameters:
        -----------
        max_k : int, default=10
            Maximum number of clusters to evaluate
        """
        plt.figure(figsize=figsize)
        visualizer = KElbowVisualizer(KMeans(random_state=self.random_state), k=(2, max_k))
        visualizer.fit(self.data)
        visualizer.show()
    
    def plot_intercluster_distance(self, figsize=(10, 8)):
        """
        Plot the distance between clusters
        """
        plt.figure(figsize=figsize)
        visualizer = InterclusterDistance(self.kmeans)
        visualizer.fit(self.data)
        visualizer.show()
    
    def plot_2d_clusters(self, figsize=(10, 8)):
        """
        Plot clusters in 2D using PCA for dimensionality reduction
        """
        # Apply PCA to reduce to 2 dimensions
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(self.data)
        
        plt.figure(figsize=figsize)
        
        # Plot points
        scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=self.labels, 
                              cmap='viridis', alpha=0.5, s=30)
        
        # Plot centroids
        centroids = pca.transform(self.kmeans.cluster_centers_)
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, 
                    linewidths=3, color='red', zorder=10)
        
        plt.colorbar(scatter, label='Cluster')
        plt.title('2D PCA Projection of Clusters')
        plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.tight_layout()
        plt.show()
    
    def plot_3d_clusters(self, figsize=(12, 10)):
        """
        Plot clusters in 3D using PCA for dimensionality reduction
        """
        # Apply PCA to reduce to 3 dimensions
        pca = PCA(n_components=3)
        reduced_data = pca.fit_transform(self.data)
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points
        scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], 
                             c=self.labels, cmap='viridis', alpha=0.5, s=30)
        
        # Plot centroids
        centroids = pca.transform(self.kmeans.cluster_centers_)
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], 
                   marker='x', s=200, linewidths=3, color='red', zorder=10)
        
        plt.colorbar(scatter, label='Cluster')
        ax.set_title('3D PCA Projection of Clusters')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})')
        plt.tight_layout()
        plt.show()
    
    def calculate_metrics(self):
        """
        Calculate and return clustering evaluation metrics
        """
        metrics = {
            'inertia': self.kmeans.inertia_,
            'silhouette_score': silhouette_score(self.data, self.labels),
            'calinski_harabasz_score': calinski_harabasz_score(self.data, self.labels),
            'davies_bouldin_score': davies_bouldin_score(self.data, self.labels)
        }
        
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
        
        return metrics
    
    def analyze_feature_importance(self, top_k=10, figsize=(12, 8)):
        """
        Analyze which features contribute most to cluster separation
        
        Parameters:
        -----------
        top_k : int, default=10
            Number of top features to display
        """
        # Calculate feature variance across cluster centroids
        centroids = self.kmeans.cluster_centers_
        feature_importance = np.var(centroids, axis=0)
        
        # Get indices of top k features
        top_indices = np.argsort(feature_importance)[::-1][:min(top_k, len(feature_importance))]
        
        plt.figure(figsize=figsize)
        plt.bar(range(len(top_indices)), feature_importance[top_indices])
        plt.xticks(range(len(top_indices)), [f"Feature {i}" for i in top_indices], rotation=45)
        plt.title(f'Top {len(top_indices)} Features by Importance in Cluster Separation')
        plt.xlabel('Feature')
        plt.ylabel('Variance Across Cluster Centroids')
        plt.tight_layout()
        plt.show()
        
        return top_indices, feature_importance[top_indices]
    
    def find_optimal_k(self, k_range=range(2, 11)):
        """
        Find optimal number of clusters using multiple metrics
        
        Parameters:
        -----------
        k_range : range, default=range(2, 11)
            Range of k values to evaluate
        """
        results = []
        
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state)
            km.fit(self.data)
            labels = km.labels_
            
            results.append({
                'k': k,
                'inertia': km.inertia_,
                'silhouette': silhouette_score(self.data, labels),
                'calinski_harabasz': calinski_harabasz_score(self.data, labels),
                'davies_bouldin': davies_bouldin_score(self.data, labels)
            })
        
        # Convert to DataFrame for easier visualization
        results_df = pd.DataFrame(results)
        
        # Plot metrics
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # Inertia (Elbow Method)
        axes[0, 0].plot(results_df['k'], results_df['inertia'], 'bo-')
        axes[0, 0].set_title('Elbow Method (Inertia)')
        axes[0, 0].set_xlabel('Number of Clusters (k)')
        axes[0, 0].set_ylabel('Inertia')
        axes[0, 0].grid(True)
        
        # Silhouette Score (higher is better)
        axes[0, 1].plot(results_df['k'], results_df['silhouette'], 'go-')
        axes[0, 1].set_title('Silhouette Score (higher is better)')
        axes[0, 1].set_xlabel('Number of Clusters (k)')
        axes[0, 1].set_ylabel('Silhouette Score')
        axes[0, 1].grid(True)
        
        # Calinski-Harabasz Index (higher is better)
        axes[1, 0].plot(results_df['k'], results_df['calinski_harabasz'], 'ro-')
        axes[1, 0].set_title('Calinski-Harabasz Index (higher is better)')
        axes[1, 0].set_xlabel('Number of Clusters (k)')
        axes[1, 0].set_ylabel('Calinski-Harabasz Score')
        axes[1, 0].grid(True)
        
        # Davies-Bouldin Index (lower is better)
        axes[1, 1].plot(results_df['k'], results_df['davies_bouldin'], 'mo-')
        axes[1, 1].set_title('Davies-Bouldin Index (lower is better)')
        axes[1, 1].set_xlabel('Number of Clusters (k)')
        axes[1, 1].set_ylabel('Davies-Bouldin Score')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return results_df
    
    def analyze_cluster_stability(self, n_runs=5, sample_frac=0.8):
        """
        Analyze cluster stability by running KMeans multiple times
        with different random initializations and data subsets
        
        Parameters:
        -----------
        n_runs : int, default=5
            Number of times to run KMeans
        sample_frac : float, default=0.8
            Fraction of data to sample in each run
        """
        n_samples = self.data.shape[0]
        sample_size = int(n_samples * sample_frac)
        
        centroid_distances = []
        
        reference_kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        reference_kmeans.fit(self.data)
        reference_centroids = reference_kmeans.cluster_centers_
        
        for i in range(n_runs):
            # Sample data
            indices = np.random.choice(n_samples, sample_size, replace=False)
            sampled_data = self.data[indices]
            
            # Run KMeans
            km = KMeans(n_clusters=self.n_clusters, random_state=self.random_state + i + 1)
            km.fit(sampled_data)
            
            # Calculate distances between this run's centroids and reference centroids
            # Note: This is simplified as we're not accounting for centroid label permutation
            distances = []
            for ref_centroid in reference_centroids:
                min_dist = np.min([np.linalg.norm(ref_centroid - c) for c in km.cluster_centers_])
                distances.append(min_dist)
            
            centroid_distances.append(np.mean(distances))
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(n_runs), centroid_distances)
        plt.axhline(y=np.mean(centroid_distances), color='r', linestyle='-', label='Mean Distance')
        plt.title('Cluster Stability Analysis')
        plt.xlabel('Run')
        plt.ylabel('Mean Distance to Reference Centroids')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        return np.mean(centroid_distances), np.std(centroid_distances)
    
    def identify_outliers(self, threshold=2.0, figsize=(12, 8)):
        """
        Identify outliers as points far from their cluster centroid
        
        Parameters:
        -----------
        threshold : float, default=2.0
            Number of standard deviations above the mean distance to consider a point an outlier
        """
        # Calculate distance of each point to its assigned centroid
        distances = []
        for i, label in enumerate(self.labels):
            centroid = self.kmeans.cluster_centers_[label]
            distance = np.linalg.norm(self.data[i] - centroid)
            distances.append(distance)
        
        distances = np.array(distances)
        
        # Calculate threshold for outliers
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        outlier_threshold = mean_dist + threshold * std_dist
        
        # Find outliers
        outlier_indices = np.where(distances > outlier_threshold)[0]
        
        print(f"Detected {len(outlier_indices)} outliers out of {len(self.data)} points ({len(outlier_indices)/len(self.data):.2%})")
        
        # If we have reduced the dimensionality for visualization, use that
        if hasattr(self, 'reduced_data') and self.reduced_data is not None:
            plt.figure(figsize=figsize)
            plt.scatter(self.reduced_data[:, 0], self.reduced_data[:, 1], 
                        c=self.labels, cmap='viridis', alpha=0.3, s=30)
            plt.scatter(self.reduced_data[outlier_indices, 0], self.reduced_data[outlier_indices, 1], 
                        color='red', s=50, label='Outliers')
            plt.title('Outlier Detection')
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            # Apply PCA to visualize
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(self.data)
            
            plt.figure(figsize=figsize)
            plt.scatter(reduced_data[:, 0], reduced_data[:, 1], 
                        c=self.labels, cmap='viridis', alpha=0.3, s=30)
            plt.scatter(reduced_data[outlier_indices, 0], reduced_data[outlier_indices, 1], 
                        color='red', s=50, label='Outliers')
            plt.title('Outlier Detection')
            plt.legend()
            plt.tight_layout()
            plt.show()
        
        return outlier_indices, distances[outlier_indices]
    
    def visualize_all(self):
        """
        Generate all visualizations and metrics at once
        """
        print("Generating all visualizations...\n")
        
        print("1. Cluster Distribution:")
        self.plot_cluster_distribution()
        print("\n")
        
        print("2. Cluster Centroids Heatmap:")
        self.plot_centroids_heatmap()
        print("\n")
        
        print("3. Silhouette Analysis:")
        self.plot_silhouette()
        print("\n")
        
        print("4. 2D Cluster Visualization:")
        self.plot_2d_clusters()
        print("\n")
        
        print("5. 3D Cluster Visualization:")
        self.plot_3d_clusters()
        print("\n")
        
        print("6. Feature Importance Analysis:")
        self.analyze_feature_importance()
        print("\n")
        
        print("7. Cluster Evaluation Metrics:")
        self.calculate_metrics()
        print("\n")
        
        print("8. Optimal K Analysis:")
        self.find_optimal_k()
        print("\n")
        
        print("9. Cluster Stability Analysis:")
        self.analyze_cluster_stability()
        print("\n")
        
        print("10. Outlier Detection:")
        self.identify_outliers()

# Main function to run the examples
if __name__ == "__main__":
    blendshapes = load_blendshape(model="SP")

    data_loader = load_dataset(
            batch_size=32,
            dataset="SP",
        )
    dataset = data_loader.dataset
    
    # Initialize and fit KMeans analyzer

    n_clusters = 10
    kmeans_analyzer = KMeansAnalyzer(n_clusters=n_clusters, use_minibatch=True)
    kmeans_analyzer.fit_from_torch_loader(data_loader)
    cluster_heads = kmeans_analyzer.kmeans.cluster_centers_
    # predict cluster assignment of each data point
    y = []
    for i in range(len(dataset)):
        W = dataset[i][0]
        W = W.numpy()
        W = np.reshape(W, (1, -1))
        y.append(kmeans_analyzer.kmeans.predict(W))
    y = np.concatenate(y, axis=0)
    len(y)
    # save y 
    np.save(f"/Users/evanpan/Documents/GitHub/ManifoldExploration/data/SP/k={n_clusters}_cluster_assignment.npy", y)


    # visualize cluster results in 3D 
    kmeans_analyzer.plot_3d_clusters(figsize=(12, 10))
    # Visualize cluster distribution
    kmeans_analyzer.plot_cluster_distribution(figsize=(10, 6))
    # Visualize cluster head faces
    ps.init()
    ps.remove_all_structures()
    rows = max(n_clusters//5, 1)
    cols = min(n_clusters, 5)
    plot_multiple_faces(blendshapes, cluster_heads, grid_size=[rows, cols], spacing=1)
    ps.show()

    
