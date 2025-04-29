#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified SMOTE implementation as a PyTorch Dataset object
with both oversampling and undersampling capabilities
"""
import torch
import numpy as np
from torch.utils.data import Dataset
from random import randint
import random
from scripts.SMOTE import *


class SMOTE(object):
    def __init__(self, distance='euclidian', dims=512, k=5):
        super(SMOTE, self).__init__()
        self.newindex = 0
        self.k = k
        self.dims = dims
        self.distance_measure = distance
        
    def populate(self, N, i, nnarray, min_samples, k):
        while N:
            nn = randint(0, k-2)
            
            diff = min_samples[nnarray[nn]] - min_samples[i]
            gap = random.uniform(0, 1)

            self.synthetic_arr[self.newindex, :] = min_samples[i] + gap * diff
            
            self.newindex += 1
            
            N -= 1
            
    def k_neighbors(self, euclid_distance, k):
        nearest_idx = torch.zeros((euclid_distance.shape[0], euclid_distance.shape[0]), dtype=torch.int64)
        
        idxs = torch.argsort(euclid_distance, dim=1)
        nearest_idx[:, :] = idxs
        
        return nearest_idx[:, 1:k]
    
    def find_k(self, X, k):
        euclid_distance = torch.zeros((X.shape[0], X.shape[0]), dtype=torch.float32)
        
        for i in range(len(X)):
            dif = (X - X[i])**2
            dist = torch.sqrt(dif.sum(axis=1))
            euclid_distance[i] = dist
            
        return self.k_neighbors(euclid_distance, k)
    
    def generate(self, min_samples, N, k):
        """
        Returns (N/100) * n_minority_samples synthetic minority samples.
        Parameters
        ----------
        min_samples : Numpy_array-like, shape = [n_minority_samples, n_features]
            Holds the minority samples
        N : percentage of new synthetic samples: 
            n_synthetic_samples = N/100 * n_minority_samples. Can be < 100.
        k : int. Number of nearest neighbours. 
        Returns
        -------
        S : Synthetic samples. array, 
            shape = [(N/100) * n_minority_samples, n_features]. 
        """
        T = min_samples.shape[0]
        self.synthetic_arr = torch.zeros(int(N/100)*T, self.dims)
        N = int(N/100)
        if self.distance_measure == 'euclidian':
            indices = self.find_k(min_samples, k)
        for i in range(indices.shape[0]):
            self.populate(N, i, indices[i], min_samples, k)
        self.newindex = 0
        return self.synthetic_arr


class BalancedSMOTEDataset(Dataset):
    def __init__(self, X, y, balance_strategy='both', target_samples_per_class=None, k=5, alpha_and_mask = False):
        """
        PyTorch Dataset that implements both SMOTE oversampling and undersampling
        to ensure balanced class distribution in each epoch.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Feature array of shape (num_of_data_points, dim)
        y : numpy.ndarray
            Class label array of shape (num_of_data_points,)
        balance_strategy : str, default='both'
            Strategy for balancing classes: 
            - 'smote': only use SMOTE to oversample minority classes
            - 'undersample': only undersample majority classes
            - 'both': use both techniques
        target_samples_per_class : int or None, default=None
            Target number of samples per class. If None, uses the median class count.
        k : int, default=5
            Number of nearest neighbors for SMOTE algorithm
        """
        self.X_orig = torch.from_numpy(X.astype(np.float32))
        self.y_orig = torch.from_numpy(y.astype(np.int64))
        self.dims = X.shape[1]
        self.num_classes = int(np.max(y) + 1)
        self.balance_strategy = balance_strategy
        self.k = k
        self.alpha_and_mask = alpha_and_mask
        # Initialize SMOTE object
        self.smote = SMOTE(dims=self.dims, k=self.k)
        
        # Balance dataset for the first time
        self._balance_dataset(target_samples_per_class)
    
    def _balance_dataset(self, target_samples_per_class=None):
        """
        Balance the dataset using the specified strategy
        """
        # Count samples per class
        class_counts = torch.zeros(self.num_classes)
        for i in range(self.num_classes):
            class_counts[i] = torch.sum(self.y_orig == i)
        
        # Determine target sample count per class
        if target_samples_per_class is None:
            # Use median count as default target
            target_count = int(torch.median(class_counts).item())
        else:
            target_count = target_samples_per_class
        
        X_balanced = []
        y_balanced = []
        
        # Process each class
        for i in range(self.num_classes):
            class_mask = self.y_orig == i
            X_class = self.X_orig[class_mask]
            y_class = self.y_orig[class_mask]
            count = len(X_class)
            
            if count < target_count and (self.balance_strategy in ['smote', 'both']):
                # Oversample minority class
                if count > self.k:  # Need at least k+1 samples for SMOTE
                    # Calculate the percentage to generate
                    N = (target_count - count) * 100 / count
                    synthetic_X = self.smote.generate(X_class, N, self.k)
                    synthetic_y = torch.ones(len(synthetic_X), dtype=torch.int64) * i
                    
                    # Combine original and synthetic samples
                    X_class = torch.cat((X_class, synthetic_X))
                    y_class = torch.cat((y_class, synthetic_y))
                else:
                    # If too few samples, duplicate existing ones
                    indices = torch.randint(0, count, (target_count - count,))
                    X_extra = X_class[indices]
                    y_extra = y_class[indices]
                    X_class = torch.cat((X_class, X_extra))
                    y_class = torch.cat((y_class, y_extra))
            
            elif count > target_count and (self.balance_strategy in ['undersample', 'both']):
                # Undersample majority class
                perm = torch.randperm(count)
                X_class = X_class[perm[:target_count]]
                y_class = y_class[perm[:target_count]]
            
            X_balanced.append(X_class)
            y_balanced.append(y_class)
        
        # Combine all classes
        self.X = torch.cat(X_balanced)
        self.y = torch.cat(y_balanced)
        
        # Create a random permutation for accessing the data
        self.perm = torch.randperm(len(self.X))
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Use the permutation to get random access
        real_idx = self.perm[idx]
        # return self.X[real_idx], self.y[real_idx]
        if self.alpha_and_mask:
            selected_id = torch.randint(0, self.dims, (1,), dtype=torch.long)
            # 2) pick cutoff alpha in [0,1]
            alpha = torch.rand(1)
            return self.X[real_idx], selected_id, alpha
        else:
            return self.X[real_idx]
    
    def on_epoch_end(self):
        """
        Call this method at the end of each epoch to rebalance the dataset
        """
        self._balance_dataset()
        
    def get_original_data(self):
        """
        Returns the original unbalanced data
        """
        return self.X_orig, self.y_orig


# Usage example:
if __name__ == "__main__":
    # Example data
    X = np.random.randn(1000, 512)
    y = np.random.randint(0, 5, 1000)
    
    # Create balanced dataset
    dataset = BalancedSMOTEDataset(X, y, balance_strategy='both', k=5)
    
    # Create DataLoader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # After each epoch, rebalance the dataset
    for epoch in range(3):
        for batch_X, batch_y in dataloader:
            # Training code would go here
            pass
        
        # Rebalance dataset for next epoch
        dataset.on_epoch_end()