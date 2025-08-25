import matplotlib.pyplot as plt
import numpy as np


def plot_polar_single(radius, angle, draw_n_points=None, figsize=(10, 10),
                      label="Data Points", color="blue", point_size=5, point_alpha=0.6):
    """
    Plot a single set of polar coordinates on a unit circle.

    Args:
        radius: array-like, radius values (0-1)
        angle: array-like, angle values (0-1, representing 0 to 2π)
        draw_n_points: int or None, if provided randomly sample N points to draw, else draw all
        figsize: tuple, figure size
        label: str, label for the data points
        color: str, color for points
        point_size: int, size of scatter points
        point_alpha: float, alpha/transparency for points

    Returns:
        fig, ax: matplotlib figure and axis objects
        selected_indices: array of indices that were plotted (useful for debugging)
    """

    # Convert to numpy arrays for easier indexing
    radius = np.array(radius)
    angle = np.array(angle)

    # Determine which points to draw
    total_points = len(radius)
    if draw_n_points is not None and draw_n_points < total_points:
        # Randomly sample indices
        selected_indices = np.random.choice(total_points, size=draw_n_points, replace=False)
        selected_indices = np.sort(selected_indices)  # Sort for consistent ordering
    else:
        # Use all points
        selected_indices = np.arange(total_points)

    # Select the subset of data
    plot_radius = radius[selected_indices]
    plot_angle = angle[selected_indices]

    # Convert normalized angles (0-1) to radians (0-2π)
    plot_angle_rad = plot_angle * 2 * np.pi

    # Create figure and axis with white background
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'),
                           facecolor='white')
    ax.set_facecolor('white')

    # Plot points
    ax.scatter(plot_angle_rad, plot_radius,
               c=color, s=point_size, alpha=point_alpha, edgecolors='none',
               label=label, zorder=3)

    # Customize the plot
    ax.set_ylim(0, 1)
    ax.set_title("Polar Coordinates\nCenter of Mass Analysis",
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)

    # Add angle labels (in terms of timesteps)
    angle_positions = [step / 32 * 2 * np.pi for step in [0, 8, 16, 24]]
    angle_labels = [f'Step {step}' for step in [0, 8, 16, 24]]
    ax.set_thetagrids(np.degrees(angle_positions), angle_labels)

    # Remove radial labels
    ax.set_yticks([])
    ax.set_yticklabels([])

    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(0.1, 1.1))

    plt.tight_layout()
    return fig, ax, selected_indices


def plot_polar_density(radius, angle, bins=32, radial_bins=50, figsize=(10, 10),
                       cmap='Blues', density_alpha=0.7, nonlinear_scale='linear'):
    """
    Plot polar coordinates as a density/heatmap showing concentration areas.

    Args:
        radius: array-like, radius values (0-1)
        angle: array-like, angle values (0-1, representing 0 to 2π)
        bins: int, number of angular bins for density calculation
        radial_bins: int, number of radial bins for density calculation
        figsize: tuple, figure size
        cmap: str, colormap for density visualization
        density_alpha: float, alpha for density visualization
        nonlinear_scale: str, scaling for color mapping ('linear', 'sqrt', 'log', 'squared')

    Returns:
        fig, ax: matplotlib figure and axis objects
        density_hist: 2D histogram values
    """

    # Convert to numpy arrays
    radius = np.array(radius)
    angle = np.array(angle)

    # Convert normalized angles to radians
    angle_rad = angle * 2 * np.pi

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'),
                           facecolor='white')
    ax.set_facecolor('white')

    # Create 2D histogram for density
    # Fix the 0-degree gap by ensuring bins cover the full circle properly
    angle_bins = np.linspace(0, 2 * np.pi, bins + 1)
    radius_bins = np.linspace(0, 1, radial_bins + 1)

    # Calculate 2D histogram
    H, angle_edges, radius_edges = np.histogram2d(angle_rad, radius,
                                                  bins=[angle_bins, radius_bins])

    # Extend the histogram to close the gap at 0 degrees
    # Add the first angular bin to the end to make it continuous
    H_extended = np.concatenate([H, H[:1]], axis=0)

    # Apply non-linear scaling to the histogram values
    H_scaled = H_extended.copy().astype(float)
    H_scaled[H_scaled == 0] = np.nan  # Make empty bins transparent

    # Apply the chosen scaling to non-zero values
    mask = ~np.isnan(H_scaled)
    if nonlinear_scale == 'sqrt':
        H_scaled[mask] = np.sqrt(H_scaled[mask])
    elif nonlinear_scale == 'log':
        # Avoid log(0) by adding small value
        H_scaled[mask] = np.log(H_scaled[mask] + 1)
    elif nonlinear_scale == 'squared':
        H_scaled[mask] = H_scaled[mask] ** 2
    elif nonlinear_scale == 'linear':
        pass  # No scaling
    else:
        # Default to sqrt if unknown scale provided
        H_scaled[mask] = np.sqrt(H_scaled[mask])

    # Get bin centers
    angle_centers = (angle_edges[:-1] + angle_edges[1:]) / 2
    radius_centers = (radius_edges[:-1] + radius_edges[1:]) / 2

    # Extend angle centers to include the wraparound
    angle_centers_extended = np.concatenate([angle_centers, [angle_centers[0] + 2 * np.pi]])

    # Create meshgrid for plotting
    A, R = np.meshgrid(angle_centers_extended, radius_centers)

    # Plot density as filled contours
    im = ax.contourf(A, R, H_scaled.T, levels=20, cmap=cmap, alpha=density_alpha)

    # Customize the plot
    ax.set_ylim(0, 1)
    ax.set_title("Polar Coordinate Density\nCenter of Mass Distribution",
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)

    # Add angle labels
    angle_positions = [step / 32 * 2 * np.pi for step in [0, 8, 16, 24]]
    angle_labels = [f'Step {step}' for step in [0, 8, 16, 24]]
    ax.set_thetagrids(np.degrees(angle_positions), angle_labels)

    # Remove radial labels
    ax.set_yticks([])
    ax.set_yticklabels([])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.1)
    cbar.set_label('Point Density', rotation=270, labelpad=20)

    plt.tight_layout()
    return fig, ax, H_scaled


def plot_polar_with_stats(radius, angle, draw_n_points=None, figsize=(10, 10),
                          color="blue", show_center=True, show_spread=True):
    """
    Plot polar coordinates with statistical overlays (center of mass, spread indicators).

    Args:
        radius: array-like, radius values (0-1)
        angle: array-like, angle values (0-1, representing 0 to 2π)
        draw_n_points: int or None, if provided randomly sample N points to draw
        figsize: tuple, figure size
        color: str, color for points
        show_center: bool, if True show center of mass
        show_spread: bool, if True show spread indicators

    Returns:
        fig, ax: matplotlib figure and axis objects
        stats_dict: dictionary containing computed statistics
    """

    # Convert to numpy arrays
    radius = np.array(radius)
    angle = np.array(angle)

    # Sample points if requested
    total_points = len(radius)
    if draw_n_points is not None and draw_n_points < total_points:
        selected_indices = np.random.choice(total_points, size=draw_n_points, replace=False)
        selected_indices = np.sort(selected_indices)
        plot_radius = radius[selected_indices]
        plot_angle = angle[selected_indices]
    else:
        selected_indices = np.arange(total_points)
        plot_radius = radius
        plot_angle = angle

    # Convert to radians
    plot_angle_rad = plot_angle * 2 * np.pi

    # Calculate statistics
    # Center of mass in Cartesian coordinates
    x_coords = plot_radius * np.cos(plot_angle_rad)
    y_coords = plot_radius * np.sin(plot_angle_rad)
    center_x = np.mean(x_coords)
    center_y = np.mean(y_coords)
    center_radius = np.sqrt(center_x ** 2 + center_y ** 2)
    center_angle = np.arctan2(center_y, center_x)

    # Spread statistics
    mean_radius = np.mean(plot_radius)
    std_radius = np.std(plot_radius)

    stats_dict = {
        'center_radius': center_radius,
        'center_angle': center_angle,
        'mean_radius': mean_radius,
        'std_radius': std_radius,
        'total_points': len(selected_indices)
    }

    # Create plot
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'),
                           facecolor='white')
    ax.set_facecolor('white')

    # Plot points
    ax.scatter(plot_angle_rad, plot_radius,
               c=color, s=10, alpha=0.6, edgecolors='none', zorder=3)

    # Show center of mass
    if show_center and center_radius > 1e-6:
        ax.scatter(center_angle, center_radius,
                   c='red', s=100, marker='x', linewidth=3,
                   label='Center of Mass', zorder=5)

    # Show spread indicators
    if show_spread:
        # Draw circle at mean radius
        theta_circle = np.linspace(0, 2 * np.pi, 100)
        ax.plot(theta_circle, np.full_like(theta_circle, mean_radius),
                '--', color='green', alpha=0.5, linewidth=2,
                label=f'Mean Radius ({mean_radius:.2f})')

    # Customize plot
    ax.set_ylim(0, 1)
    ax.set_title("Polar Coordinates with Statistics\nCenter of Mass Analysis",
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)

    # Add angle labels
    angle_positions = [step / 32 * 2 * np.pi for step in [0, 8, 16, 24]]
    angle_labels = [f'Step {step}' for step in [0, 8, 16, 24]]
    ax.set_thetagrids(np.degrees(angle_positions), angle_labels)

    # Remove radial labels
    ax.set_yticks([])
    ax.set_yticklabels([])

    # Add legend
    if show_center or show_spread:
        ax.legend(loc='upper left', bbox_to_anchor=(0.1, 1.1))

    plt.tight_layout()
    return fig, ax, stats_dict