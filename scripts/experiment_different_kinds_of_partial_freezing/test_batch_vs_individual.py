import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam, SGD

def create_test_data(num_frames=4, num_params=2):
    """Create synthetic test data with different targets for each frame"""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create different target parameters for each frame
    targets = torch.randn(num_frames, num_params) * 2.0
    targets = targets.to(device)
    # Create some coupling between parameters to make the optimization interesting
    def loss_fn(params, target):
        # Quadratic loss with some cross-terms to make it more interesting
        diff = params - target
        loss = torch.sum(diff**2)
        return loss
    
    return targets, loss_fn

def optimize_individual(targets, loss_fn, optimizer_class, lr=0.1, num_iterations=50):
    """Optimize each frame individually"""
    num_frames, num_params = targets.shape
    
    # Store trajectory for each frame
    trajectories = []
    final_params = []
    
    for frame_idx in range(num_frames):
        # Initialize parameters for this frame
        params = torch.zeros(num_params, requires_grad=True)
        optimizer = optimizer_class([params], lr=lr)
        
        # Store trajectory for this frame
        trajectory = []
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            loss = loss_fn(params, targets[frame_idx])
            loss.backward()
            optimizer.step()
            
            # Store current parameters
            trajectory.append(params.detach().clone())
        
        trajectories.append(torch.stack(trajectory))
        final_params.append(params.detach().clone())
    
    return torch.stack(trajectories), torch.stack(final_params)

def optimize_batched(targets, loss_fn, optimizer_class, lr=0.1, num_iterations=50):
    """Optimize all frames in a batch simultaneously"""
    device = targets.device
    num_frames, num_params = targets.shape
    
    # Initialize parameters for all frames
    params = torch.zeros(num_frames, num_params, requires_grad=True).to(device)
    optimizer = optimizer_class([params], lr=lr)
    
    # Store trajectory
    trajectory = []
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        # Compute loss for all frames
        total_loss = 0
        total_loss = loss_fn(params, targets)
        
        total_loss.backward()
        optimizer.step()
        
        # Store current parameters
        trajectory.append(params.detach().cpu().clone())
    
    return torch.stack(trajectory), params.detach().cpu().clone()

def optimize_batched_separate_optimizers(targets, loss_fn, optimizer_class, lr=0.1, num_iterations=50):
    """Optimize with separate optimizers for each frame (should match individual)"""
    num_frames, num_params = targets.shape
    device = targets.device
    # Initialize parameters for all frames
    # params = torch.zeros(num_frames, num_params, requires_grad=True)
    params = [torch.zeros(num_params, requires_grad=True).to(device) for _ in range(num_frames)]
    # Create separate optimizers for each frame
    optimizers = []
    for frame_idx in range(num_frames):
        # Create optimizer for just this frame's parameters
        # frame_params = params[frame_idx:frame_idx+1]  # Keep batch dimension
        frame_params = params[frame_idx]  # Keep batch dimension
        optimizer = optimizer_class([frame_params], lr=lr)
        optimizers.append(optimizer)
    
    # Store trajectory
    trajectory = []
    
    for iteration in range(num_iterations):
        # Compute loss and update each frame separately
        for frame_idx in range(num_frames):
            optimizers[frame_idx].zero_grad()
            frame_loss = loss_fn(params[frame_idx], targets[frame_idx])
            frame_loss.backward()
            optimizers[frame_idx].step()
        
        # Store current parameters
        optimized_params = [p.detach().cpu().clone() for p in params]
        optimized_params = torch.stack(optimized_params)
        trajectory.append(optimized_params)
    
    return torch.stack(trajectory), optimized_params

def plot_optimization_comparison(targets, results, title_suffix=""):
    """Plot optimization trajectories for comparison"""
    num_frames, num_params = targets.shape
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Optimization Comparison - {title_suffix}', fontsize=16)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    for frame_idx in range(num_frames):
        color = colors[frame_idx % len(colors)]
        
        # Plot parameter 1 trajectory
        for method_name, (trajectory, final_params) in results.items():
            if len(trajectory.shape) == 3:  # Individual optimization
                traj = trajectory[frame_idx, :, 0]
            else:  # Batched optimization
                traj = trajectory[:, frame_idx, 0]
            
            axes[0, 0].plot(traj, label=f'{method_name} Frame {frame_idx}', 
                           color=color, linestyle='--' if 'Batched' in method_name else '-')
        
        # Plot parameter 2 trajectory
        for method_name, (trajectory, final_params) in results.items():
            if len(trajectory.shape) == 3:  # Individual optimization
                traj = trajectory[frame_idx, :, 1]
            else:  # Batched optimization
                traj = trajectory[:, frame_idx, 1]
            
            axes[0, 1].plot(traj, label=f'{method_name} Frame {frame_idx}', 
                           color=color, linestyle='--' if 'Batched' in method_name else '-')
        
        # Plot target points
        axes[0, 0].scatter(49, targets[frame_idx, 0], color=color, marker='*', s=100, 
                          label=f'Target Frame {frame_idx}')
        axes[0, 1].scatter(49, targets[frame_idx, 1], color=color, marker='*', s=100, 
                          label=f'Target Frame {frame_idx}')
    
    axes[0, 0].set_title('Parameter 1 Trajectory')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Parameter Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].set_title('Parameter 2 Trajectory')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Parameter Value')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 2D parameter space trajectory
    for frame_idx in range(num_frames):
        color = colors[frame_idx % len(colors)]
        
        for method_name, (trajectory, final_params) in results.items():
            if len(trajectory.shape) == 3:  # Individual optimization
                traj = trajectory[frame_idx, :, :]
            else:  # Batched optimization
                traj = trajectory[:, frame_idx, :]
            
            axes[1, 0].plot(traj[:, 0], traj[:, 1], 
                           label=f'{method_name} Frame {frame_idx}', 
                           color=color, linestyle='--' if 'Batched' in method_name else '-')
            axes[1, 0].scatter(traj[0, 0], traj[0, 1], color=color, marker='o', s=50)  # Start
            axes[1, 0].scatter(traj[-1, 0], traj[-1, 1], color=color, marker='x', s=50)  # End
        
        # Plot target
        axes[1, 0].scatter(targets[frame_idx, 0], targets[frame_idx, 1], 
                          color=color, marker='*', s=100, label=f'Target Frame {frame_idx}')
    
    axes[1, 0].set_title('2D Parameter Space Trajectory')
    axes[1, 0].set_xlabel('Parameter 1')
    axes[1, 0].set_ylabel('Parameter 2')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot final parameter differences
    method_names = list(results.keys())
    if len(method_names) >= 2:
        individual_final = results[method_names[0]][1]  # Assume first is individual
        batched_final = results[method_names[1]][1]    # Assume second is batched
        
        if len(individual_final.shape) == 2 and len(batched_final.shape) == 2:
            differences = torch.abs(individual_final - batched_final)
            
            frame_indices = range(num_frames)
            width = 0.35
            
            axes[1, 1].bar([i - width/2 for i in frame_indices], differences[:, 0], 
                          width, label='Parameter 1 Difference', alpha=0.7)
            axes[1, 1].bar([i + width/2 for i in frame_indices], differences[:, 1], 
                          width, label='Parameter 2 Difference', alpha=0.7)
            
            axes[1, 1].set_title('Final Parameter Differences\n(Individual vs Batched)')
            axes[1, 1].set_xlabel('Frame Index')
            axes[1, 1].set_ylabel('Absolute Difference')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
    
    plt.tight_layout()
    return fig

def run_comparison_test():
    """Run the complete comparison test"""
    # Create test data
    targets, loss_fn = create_test_data(num_frames=10, num_params=2)
    
    print("Targets for each frame:")
    for i, target in enumerate(targets):
        print(f"Frame {i}: {target.numpy()}")
    
    # Test with Adam optimizer
    print("\n=== Testing with Adam Optimizer ===")
    
    # Individual optimization
    print("Running individual optimization...")
    adam_individual = optimize_individual(targets, loss_fn, Adam, lr=0.1, num_iterations=50)
    
    # Batched optimization  
    print("Running batched optimization...")
    adam_batched = optimize_batched(targets, loss_fn, Adam, lr=0.1, num_iterations=50)
    adam_batched = [adam_batched[0].permute(1, 0, 2), adam_batched[1]]  # Keep only trajectory and final params

    # Batched with separate optimizers
    print("Running batched with separate optimizers...")
    adam_batched_separate = optimize_batched_separate_optimizers(targets, loss_fn, Adam, lr=0.1, num_iterations=50)
    adam_batched_separate = [adam_batched_separate[0].permute(1, 0, 2), adam_batched_separate[1]]  # Keep only trajectory and final params
    # Plot Adam results
    adam_results = {
        'Individual': adam_individual,
        'Batched': adam_batched,
        'Batched Separate': adam_batched_separate
    }
    
    fig1 = plot_optimization_comparison(targets, adam_results, "Adam Optimizer")
    plt.show()
    plt.savefig("adam_optimization_comparison.png")
    # Test with SGD optimizer
    print("\n=== Testing with SGD Optimizer ===")
    
    # Individual optimization
    print("Running individual optimization...")
    sgd_individual = optimize_individual(targets, loss_fn, SGD, lr=0.1, num_iterations=50)
    
    # Batched optimization
    print("Running batched optimization...")
    sgd_batched = optimize_batched(targets, loss_fn, SGD, lr=0.1, num_iterations=50) # shape = 50, 5, 2
    sgd_batched = [sgd_batched[0].permute(1, 0, 2), sgd_batched[1]]  # Keep only trajectory and final params
    # Plot SGD results
    sgd_results = {
        'Individual': sgd_individual,
        'Batched': sgd_batched
    }
    
    fig2 = plot_optimization_comparison(targets, sgd_results, "SGD Optimizer")
    plt.show()
    plt.savefig("sgd_optimization_comparison.png")
    
    # Print final results comparison
    print("\n=== Final Results Comparison ===")
    
    print("\nAdam - Individual vs Batched final parameters:")
    print("Individual final params:")
    print(adam_individual[1].numpy())
    print("Batched final params:")
    print(adam_batched[1].numpy())
    print("Batched Separate final params:")
    print(adam_batched_separate[1].numpy())
    
    print("\nSGD - Individual vs Batched final parameters:")
    print("Individual final params:")
    print(sgd_individual[1].numpy())
    print("Batched final params:")
    print(sgd_batched[1].numpy())
    
    # Calculate and print differences
    adam_diff = torch.abs(adam_individual[1] - adam_batched[1])
    sgd_diff = torch.abs(sgd_individual[1] - sgd_batched[1])
    
    print(f"\nAdam difference (Individual vs Batched): {torch.mean(adam_diff):.6f}")
    print(f"SGD difference (Individual vs Batched): {torch.mean(sgd_diff):.6f}")

if __name__ == "__main__":
    run_comparison_test()