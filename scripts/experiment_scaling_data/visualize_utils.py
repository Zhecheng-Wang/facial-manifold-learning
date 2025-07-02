import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_timeseries_bounds(data, log_scale=False, outlier_method='iqr', 
                          confidence_interval=0.95, title="Time Series with Bounds",
                          xlabel="Time Step", ylabel="Optimization Cost"):
    
    # Convert to numpy array for easier manipulation
    data = np.array(data)
    
    # Get dimensions
    n_series, n_timepoints = data.shape
    time_steps = np.arange(n_timepoints)
    
    # Remove outliers for each time point
    cleaned_data = np.zeros_like(data)
    
    for t in range(n_timepoints):
        values_at_t = data[:, t]
        
        if outlier_method == 'iqr':
            # Interquartile range method
            Q1 = np.percentile(values_at_t, 25)
            Q3 = np.percentile(values_at_t, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            mask = (values_at_t >= lower_bound) & (values_at_t <= upper_bound)
            
        elif outlier_method == 'zscore':
            # Z-score method
            z_scores = np.abs(stats.zscore(values_at_t))
            mask = z_scores < 3
            
        elif outlier_method == 'percentile':
            # Percentile method (remove top and bottom 5%)
            lower_percentile = (1 - confidence_interval) / 2 * 100
            upper_percentile = (1 + confidence_interval) / 2 * 100
            lower_bound = np.percentile(values_at_t, lower_percentile)
            upper_bound = np.percentile(values_at_t, upper_percentile)
            mask = (values_at_t >= lower_bound) & (values_at_t <= upper_bound)
        
        # Keep only non-outlier values
        cleaned_values = values_at_t[mask]
        
        # If too few values remain, use original data
        if len(cleaned_values) < 3:
            cleaned_values = values_at_t
        
        # Store cleaned data (pad with NaN if needed)
        cleaned_data[:len(cleaned_values), t] = cleaned_values
        cleaned_data[len(cleaned_values):, t] = np.nan
    
    # Calculate statistics for each time point
    means = []
    lower_bounds = []
    upper_bounds = []
    
    alpha = 1 - confidence_interval
    
    for t in range(n_timepoints):
        valid_values = cleaned_data[:, t]
        valid_values = valid_values[~np.isnan(valid_values)]
        
        if len(valid_values) > 0:
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values)
            
            # Calculate confidence interval
            if len(valid_values) > 1:
                # Use t-distribution for small samples
                t_stat = stats.t.ppf(1 - alpha/2, len(valid_values) - 1)
                margin_error = t_stat * std_val / np.sqrt(len(valid_values))
            else:
                margin_error = 0
            
            means.append(mean_val)
            lower_bounds.append(mean_val - margin_error)
            upper_bounds.append(mean_val + margin_error)
        else:
            # Fallback to original data if no valid values
            original_values = data[:, t]
            means.append(np.mean(original_values))
            lower_bounds.append(np.min(original_values))
            upper_bounds.append(np.max(original_values))
    
    means = np.array(means)
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot individual series (optional, with low alpha)
    for i in range(min(n_series, 20)):  # Limit to 20 series for clarity
        ax.plot(time_steps, data[i], alpha=0.1, color='gray', linewidth=0.5)
    
    # Plot mean line
    ax.plot(time_steps, means, color='blue', linewidth=2, label='Mean')
    
    # Plot confidence interval as shaded area
    ax.fill_between(time_steps, lower_bounds, upper_bounds, 
                    alpha=0.3, color='blue', label=f'{confidence_interval*100:.0f}% CI')
    
    # Set log scale if requested
    if log_scale:
        ax.set_yscale('log')
        ylabel += " (log scale)"
    
    # Customize plot
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Improve layout
    plt.tight_layout()
    
    return fig, ax