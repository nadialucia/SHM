import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


def plot_optimization_progress(generations_data, target_E, target_modes, target_freqs):
    """
    Plot the optimization progress with three side-by-side plots
    """
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(15, 8))
    
    # Create a 2x3 grid of subplots
    gs = GridSpec(2, 3, figure=fig)
    
    # Create axes for each subplot
    ax1_top = fig.add_subplot(gs[0, 0])
    ax1_bottom = fig.add_subplot(gs[1, 0])
    ax2_top = fig.add_subplot(gs[0, 1])
    ax2_bottom = fig.add_subplot(gs[1, 1])
    ax3_top = fig.add_subplot(gs[0, 2])
    ax3_bottom = fig.add_subplot(gs[1, 2])
    
    best_solutions = []
    best_modes = []
    best_freqs = []
    
    # Group data by generation
    generation_groups = {}
    for data in generations_data:
        gen = data['generation']
        if gen not in generation_groups:
            generation_groups[gen] = []
        generation_groups[gen].append(data)
    
    # Get best solution for each generation
    for gen in sorted(generation_groups.keys()):
        gen_solutions = generation_groups[gen]
        # Find solution with best fitness in this generation
        best_solution = min(gen_solutions, key=lambda x: x['fitness'])
        
        best_solutions.append(best_solution['solution'])
        best_modes.append(best_solution['mode_shapes'])
        best_freqs.append(best_solution['frequencies'])
    
    best_solutions = np.array(best_solutions)
    best_modes = np.array(best_modes)
    best_freqs = np.array(best_freqs)
    
    # Rest of the plotting code remains the same
    # Plot 1: Solution Evolution
    ax1_top.plot(target_E, 'k-', label='Target E')
    ax1_top.set_title('Target Solution')
    ax1_top.legend()
    
    im1 = ax1_bottom.imshow(best_solutions, aspect='auto', cmap='viridis')
    ax1_bottom.set_ylabel('Generation')
    ax1_bottom.set_title('Solution Evolution')
    plt.colorbar(im1, ax=ax1_bottom)
    
    # Plot 2: Mode Shapes Evolution
    ax2_top.plot(target_modes.T)
    ax2_top.set_title('Target Mode Shapes')
    
    im2 = ax2_bottom.imshow(best_modes, aspect='auto', cmap='viridis')
    ax2_bottom.set_title('Mode Shapes Evolution')
    plt.colorbar(im2, ax=ax2_bottom)
    
    # Plot 3: Frequencies Evolution
    ax3_top.plot(target_freqs, 'k-', label='Target')
    ax3_top.set_title('Target Frequencies')
    ax3_top.legend()
    
    # Create a colormap for frequency differences
    freq_differences = np.abs(best_freqs - target_freqs)
    im3 = ax3_bottom.imshow(freq_differences, aspect='auto', cmap='viridis')
    ax3_bottom.set_title('Frequency Differences')
    plt.colorbar(im3, ax=ax3_bottom)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
# generations_data = [
#     {
#         'best_solution': solution_matrix,
#         'best_modes': mode_shapes,
#         'best_freqs': frequencies
#     },
#     # ... data for each generation
# ]
# plot_optimization_progress(generations_data, target_E, target_modes, target_freqs)