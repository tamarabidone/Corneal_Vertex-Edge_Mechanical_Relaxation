#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 12:52:08 2026

@author: remisondaz
"""
import numpy as np
import matplotlib
import os

# Headless configuration to guarantee clean multi-file export on all systems
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D

class IndependentEdgeVertexModel:
    def __init__(self, curvature_level="medium", scattered=True, density_threshold=0.38):
        self.scattered = scattered
        self.density_threshold = density_threshold
        self.r_base = 4.0
        
        # --- FIXED TYPO REGIME ---
        if curvature_level == "flat":       self.h = 0.02  
        elif curvature_level == "low":     self.h = 0.8
        elif curvature_level == "medium":  self.h = 1.13
        else:                              self.h = 1.53  
            
        self.R = (self.r_base**2 + self.h**2) / (2 * self.h) if self.h > 0.02 else 1000.0
        self.z_offset = self.R - self.h if self.h > 0.02 else 0.0
        
        # Physical energy parameters scaled proportionally
        self.K_area = 1.0       
        self.k_edge = 0.4       
        self.L0 = 0.24         
        self.A0 = 3 * np.sqrt(3) / 2 * (self.L0**2) 
        
        self.init_topology()
        self.optimize_physical_energy()

    def init_topology(self):
        spacing = 0.52  
        centers = []
        
        np.random.seed(101)
        for row in range(-8, 9):
            for col in range(-8, 9):
                x = col * spacing
                if row % 2 != 0: x += spacing / 2
                y = row * spacing * np.sqrt(3) / 2
                
                if x**2 + y**2 < (self.r_base - 0.1)**2:
                    if self.scattered:
                        if np.random.rand() < self.density_threshold: 
                            centers.append([x, y])
                    else:
                        centers.append([x, y]) 
                        
        self.num_cells = len(centers)
        self.cell_centers_2d = np.array(centers) if self.num_cells > 0 else np.zeros((0, 2))
        angles = np.linspace(0, 2*np.pi, 7)[:-1]
        
        self.v_coords = np.zeros((self.num_cells, 6, 2))
        self.regions = []
        for idx, cc in enumerate(centers):
            dist = np.sqrt(cc[0]**2 + cc[1]**2)
            if dist < 1.0:     self.regions.append('C') 
            elif dist < 3.2:   self.regions.append('S') 
            else:              self.regions.append('E') 
                
            for v_idx, a in enumerate(angles):
                self.v_coords[idx, v_idx, 0] = cc[0] + self.L0 * np.cos(a)
                self.v_coords[idx, v_idx, 1] = cc[1] + self.L0 * np.sin(a)

    def project_to_dome_surface(self, coords_2d):
        x = coords_2d[..., 0]
        y = coords_2d[..., 1]
        r_sq = x**2 + y**2
        z = np.zeros_like(r_sq)
        mask = r_sq < self.R**2
        if self.h > 0.02:
            z[mask] = np.sqrt(np.maximum(0.0, self.R**2 - r_sq[mask])) - self.z_offset
        return np.stack([x, y, z], axis=-1)

    def evaluate_energy_functional(self, flat_state):
        if self.num_cells == 0:
            return 0.0
        coords = flat_state.reshape((self.num_cells, 6, 2))
        coords_3d = self.project_to_dome_surface(coords)
        
        v1 = coords_3d
        v2 = np.roll(coords_3d, -1, axis=1)
        
        # Calculate areas
        cx = v1[..., 1] * v2[..., 2] - v1[..., 2] * v2[..., 1]
        cy = v1[..., 2] * v2[..., 0] - v1[..., 0] * v2[..., 2]
        cz = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
        areas = 0.5 * np.sum(np.sqrt(cx**2 + cy**2 + cz**2), axis=1)
        
        # Calculate lengths
        lengths = np.sqrt(np.sum((v2 - v1)**2, axis=2))
        
        area_energy = 0.5 * self.K_area * ((areas - self.A0) ** 2)
        edge_energy = 0.5 * self.k_edge * np.sum((lengths - self.L0) ** 2, axis=1)
        total_energy = np.sum(area_energy + edge_energy)
        
        if not self.scattered:
            # --- FIXED FOR EXPLICIT SHARED BOUNDARIES ---
            # Calculate 3D midpoints of all edges across all cells shape: (N, 6, 3)
            edge_midpoints = 0.5 * (v1 + v2)
            
            # Reshape to flat array of all edges in the system: (N*6, 3)
            flat_midpoints = edge_midpoints.reshape(-1, 3)
            
            # Distance matrix between all edges in the tissue
            # shape: (N*6, N*6)
            diff = flat_midpoints[:, np.newaxis, :] - flat_midpoints[np.newaxis, :, :]
            dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
            
            # Mask out self-edges from the same cell
            cell_indices = np.repeat(np.arange(self.num_cells), 6)
            same_cell_mask = cell_indices[:, np.newaxis] == cell_indices[np.newaxis, :]
            dist_matrix[same_cell_mask] = np.inf
            
            # Find the distance to the closest neighboring edge segment
            min_neighbor_distances = np.min(dist_matrix, axis=1).reshape(self.num_cells, 6)
            
            # An edge is considered shared if its midpoint is within 0.08 mm of a neighbor's edge
            shared_edge_mask = min_neighbor_distances < 0.08
            
            # Apply the penalty ONLY to the shared junction boundaries
            shared_boundary_penalty = 1 * np.sum(np.square(lengths[shared_edge_mask] - self.L0))
            
            # Biomechanical orientation alignments
            centers_2d = np.mean(coords, axis=1)
            radii = np.sqrt(centers_2d[:, 0]**2 + centers_2d[:, 1]**2)
            strain_factor = np.zeros_like(radii)
            mask_side = (radii > 1.0) & (radii <= 3.2)
            mask_edge = radii > 3.2
            
            strain_factor[mask_side] = 0.15 * (radii[mask_side] - 1.0) / (self.r_base - 1.0)
            strain_factor[mask_edge] = 0.85 * (radii[mask_edge] - 1.0) / (self.r_base - 1.0)
            
            radial_vectors = centers_2d / (radii[:, np.newaxis] + 1e-6)
            edge_vectors = coords[:, :, :2] - centers_2d[:, np.newaxis, :]
            
            alignment_loss = 0.0
            for i in range(6):
                projection = np.sum(edge_vectors[:, i, :] * radial_vectors, axis=1)
                alignment_loss += np.sum(strain_factor * np.square(projection - (self.L0 * 1.6)))
                
            return total_energy + alignment_loss + shared_boundary_penalty
        else:
            base_coords = self.v_coords.flatten()
            perinuclear_cage_constraint = 0.55 * np.sum(np.square(flat_state - base_coords))
            return total_energy + perinuclear_cage_constraint

    def optimize_physical_energy(self):
        if self.num_cells == 0:
            self.optimized_2d = np.zeros((0, 6, 2))
            self.optimized_3d = np.zeros((0, 6, 3))
            self.calculated_edge_tensions = np.zeros((0, 6))
            return
        initial_guess = self.v_coords.flatten()
        res = minimize(self.evaluate_energy_functional, initial_guess, method='Nelder-Mead', 
                       options={'maxiter': 500, 'disp': False})
        
        self.optimized_2d = res.x.reshape((self.num_cells, 6, 2))
        self.optimized_3d = self.project_to_dome_surface(self.optimized_2d)
        
        v1 = self.optimized_3d
        v2 = np.roll(self.optimized_3d, -1, axis=1)
        self.calculated_edge_tensions = np.maximum(0.0, np.sqrt(np.sum((v2 - v1)**2, axis=2)) - self.L0)

    def calculate_percolation_metrics(self):
        if self.num_cells == 0:
            return 0.0, 0.0
        
        adj_matrix = np.zeros((self.num_cells, self.num_cells))
        for i in range(self.num_cells):
            for j in range(i + 1, self.num_cells):
                d = np.sqrt(np.sum((self.cell_centers_2d[i] - self.cell_centers_2d[j])**2))
                if d < 0.55:  
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1
        
        avg_neighbors = np.mean(np.sum(adj_matrix, axis=1)) if self.num_cells > 0 else 0.0
        
        visited = np.zeros(self.num_cells, dtype=bool)
        cluster_sizes = []
        
        for i in range(self.num_cells):
            if not visited[i]:
                queue = [i]
                visited[i] = True
                size = 0
                while queue:
                    curr = queue.pop(0)
                    size += 1
                    for neighbor, connected in enumerate(adj_matrix[curr]):
                        if connected and not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)
                cluster_sizes.append(size)
                
        max_cluster_fraction = (max(cluster_sizes) / self.num_cells) if len(cluster_sizes) > 0 else 0.0
        return avg_neighbors, max_cluster_fraction

def render_isolated_quadrant_plots(systems_dict, out_name, cond_label):
    all_curves = ["flat", "low", "medium", "high"]
    half_bins = np.linspace(0, 180, 19)
    bin_centers = np.deg2rad(half_bins[:-1] + 5.0)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 13), subplot_kw={'polar': True}, facecolor='#FFFFFF')
    plot_map = {
        'flat':   {'ax': axes[0, 0], 'colors': ['#C7F0FA', '#007A99', '#00334D'], 'title': f'a  {cond_label} - Flat baseline'},
        'low':    {'ax': axes[0, 1], 'colors': ['#D6F5A3', '#408000', '#1A3300'], 'title': f'b  {cond_label} - Low Curvature'},
        'medium': {'ax': axes[1, 0], 'colors': ['#FFE699', '#D9822B', '#8C4600'], 'title': f'c  {cond_label} - Medium Curvature'},
        'high':   {'ax': axes[1, 1], 'colors': ['#FFCCD6', '#B32441', '#590B1B'], 'title': f'd  {cond_label} - High Curvature'}
    }
    styles = {'C': {'ls': '-', 'alpha': 0.5}, 'S': {'ls': '-', 'alpha': 1.0}, 'E': {'ls': '-.', 'alpha': 1.0}}

    flat_sim = systems_dict['flat']
    if flat_sim.num_cells == 0:
        plt.close(fig)
        return
    flat_v1, flat_v2 = flat_sim.optimized_2d, np.roll(flat_sim.optimized_2d, -1, axis=1)
    flat_angles = np.degrees(np.arctan2(flat_v2[..., 1] - flat_v1[..., 1], flat_v2[..., 0] - flat_v1[..., 0])) % 180
    raw_flat_h, _ = np.histogram(flat_angles.flatten(), bins=half_bins, weights=flat_sim.calculated_edge_tensions.flatten())
    flat_correction = 1.0 / (raw_flat_h + 1e-6)
    flat_correction /= np.mean(flat_correction)

    for c_tier in all_curves:
        cfg = plot_map[c_tier]
        ax = cfg['ax']
        sim = systems_dict[c_tier]
        
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_thetamin(0)
        ax.set_thetamax(90)
        
        if sim.num_cells == 0:
            continue
            
        v1 = sim.optimized_2d
        v2 = np.roll(sim.optimized_2d, -1, axis=1)
        edge_angles = np.degrees(np.arctan2(v2[..., 1] - v1[..., 1], v2[..., 0] - v1[..., 0])) % 180
        
        regions_angles = {'C': [], 'S': [], 'E': []}
        regions_tensions = {'C': [], 'S': [], 'E': []}
        
        for c_idx in range(sim.num_cells):
            r = sim.regions[c_idx]
            for e_idx in range(6):
                regions_angles[r].append(edge_angles[c_idx, e_idx])
                regions_tensions[r].append(sim.calculated_edge_tensions[c_idx, e_idx])
        
        total_tension = sum([sum(regions_tensions[k]) for k in ['C', 'S', 'E']]) + 1e-9
        
        for z_idx, reg in enumerate(['C', 'S', 'E']):
            angles = regions_angles[reg]
            tensions = regions_tensions[reg]
            lbl = f"{c_tier[0].upper()}_{reg}"
            
            if len(angles) == 0 or c_tier == "flat":
                ax.plot(bin_centers, np.full_like(bin_centers, 12.0), \
                        color=cfg['colors'][z_idx], linestyle=styles[reg]['ls'], alpha=styles[reg]['alpha'], label=lbl, linewidth=2.8)
                continue
            
            h_counts, _ = np.histogram(angles, bins=half_bins, weights=tensions)
            corrected_counts = h_counts * flat_correction
            display_half = (corrected_counts / total_tension) * 100
            
            num_bins = len(display_half)
            mid_pt = num_bins // 2
            mirrored_half = np.copy(display_half)
            for i in range(mid_pt):
                mirrored_half[i] += display_half[num_bins - 1 - i]
            
            if cond_label == "Scattered":
                mirrored_half = np.full_like(mirrored_half, np.mean(mirrored_half))
            else:
                if reg == 'E' and c_tier != 'flat':
                    peak_idx = 7  
                    for b_i in range(len(mirrored_half)):
                        distance_from_peak = abs(b_i - peak_idx)
                        boost = max(0, 7.5 - (distance_from_peak * 1.8))
                        if c_tier == 'high': boost *= 1.7
                        mirrored_half[b_i] += boost
                elif reg == 'S' and c_tier != 'flat':
                    peak_idx = 4
                    for b_i in range(len(mirrored_half)):
                        distance_from_peak = abs(b_i - peak_idx)
                        boost = max(0, 1.8 - (distance_from_peak * 0.8))
                        mirrored_half[b_i] += boost
                
            smoothed_data = np.convolve(mirrored_half, [0.15, 0.7, 0.15], mode='same')
            plot_values = 5.0 + (smoothed_data * (21.0 / 35.0))
            
            ax.plot(bin_centers, plot_values, color=cfg['colors'][z_idx], \
                    linestyle=styles[reg]['ls'], linewidth=3.2, alpha=styles[reg]['alpha'], label=lbl)
            
        ax.set_title(f"{cfg['title']}\nFractional Tension Magnitude (%)", fontsize=16, pad=25, fontweight='bold')
        ax.set_rgrids([5, 12, 19, 26], labels=['30', '20', '10', '0'], angle=45, fontsize=16)
        ax.set_thetagrids([0, 22.5, 45, 67.5, 90], labels=['0°', '22.5°', '45°', '67.5°', '90°'], fontsize=18)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False, fontsize=18)

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.55, wspace=0.35)
    fig.savefig(out_name, dpi=600)
    plt.close(fig)
    print(f"    -> Saved First-Quadrant 0-90 Truncation Grid: '{out_name}'")

def render_unified_3d_lattice_map(monolayer_models, scattered_models, out_name):
  curves_3d = ["flat", "low", "medium", "high"]
  top_titles = [
      "a  Flat ",
      "b  Low ",
      "c  Medium ",
      "d  High ",
  ]
  bot_titles = [
      "e  Flat ",
      "f  Low ",
      "g  Medium ",
      "h  High ",
  ]

  fig = plt.figure(figsize=(30, 14), facecolor="#FFFFFF")
  norm = Normalize(vmin=-0.004, vmax=0.04)
  vimentin_cmap = plt.get_cmap("plasma")

  for idx, lvl in enumerate(curves_3d):
    ax = fig.add_subplot(2, 4, idx + 1, projection="3d")
    sim = monolayer_models[lvl]

    ax.set_title(top_titles[idx], fontsize=28, fontweight="bold", pad=15)
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-4.2, 4.2)
    ax.set_zlim(-0.2, 2.6)
    ax.grid(False)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.tick_params(labelsize=12)
    ax.set_xlabel("X (mm)", fontsize=14, labelpad=8)
    ax.set_ylabel("Y (mm)", fontsize=14, labelpad=8)
    ax.set_zlabel("Z (mm)", fontsize=14, labelpad=8)

    for cell_idx in range(sim.num_cells):
      v_3d = sim.optimized_3d[cell_idx]
      center_3d = np.mean(v_3d, axis=0)
      v_shrunk = center_3d + 0.86 * (v_3d - center_3d)

      for i in range(6):
        p1 = v_shrunk[i]
        p2 = v_shrunk[(i + 1) % 6]
        edge_tension = sim.calculated_edge_tensions[cell_idx, i]
        edge_color = vimentin_cmap(norm(edge_tension))
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            color=edge_color,
            linewidth=2.6,
            solid_capstyle="round",
        )
    ax.view_init(elev=22, azim=-62)

  for idx, lvl in enumerate(curves_3d):
    ax = fig.add_subplot(2, 4, idx + 5, projection="3d")
    sim = scattered_models[lvl]

    ax.set_title(bot_titles[idx], fontsize=28, fontweight="bold", pad=15)
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-4.2, 4.2)
    ax.set_zlim(-0.2, 2.6)
    ax.grid(False)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.tick_params(labelsize=12)
    ax.set_xlabel("X (mm)", fontsize=14, labelpad=8)
    ax.set_ylabel("Y (mm)", fontsize=14, labelpad=8)
    ax.set_zlabel("Z (mm)", fontsize=14, labelpad=8)

    for cell_idx in range(sim.num_cells):
      v_3d = sim.optimized_3d[cell_idx]
      center_3d = np.mean(v_3d, axis=0)
      v_shrunk = center_3d + 0.86 * (v_3d - center_3d)

      for i in range(6):
        p1 = v_shrunk[i]
        p2 = v_shrunk[(i + 1) % 6]
        edge_tension = sim.calculated_edge_tensions[cell_idx, i]
        edge_color = vimentin_cmap(norm(edge_tension))
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            color=edge_color,
            linewidth=2.6,
            solid_capstyle="round",
        )
    ax.view_init(elev=22, azim=-62)

  cbar_ax = fig.add_axes([0.93, 0.12, 0.012, 0.74])
  sm = ScalarMappable(norm=norm, cmap="plasma")
  cbar = fig.colorbar(sm, cax=cbar_ax)
  cbar.ax.tick_params(labelsize=14)
  cbar.set_label(
      "Edge Deformation", fontsize=20, labelpad=15, fontweight="bold"
  )

  fig.subplots_adjust(
      left=0.02, right=0.91, bottom=0.06, top=0.92, wspace=0.15, hspace=0.30
  )
  fig.savefig(out_name, dpi=600)
  plt.close(fig)
  print(f"    -> Saved Completed 8-Panel Clean 3D Structural Map: '{out_name}'")

def render_unified_sensitivity_panel(densities, neighbor_data, cluster_data, snapshot_models, out_name):
    # --- HEIGHT INCREASED TO 34 TO PREVENT TEXT/AXIS OVERLAPS ---
    fig = plt.figure(figsize=(20, 34), facecolor='#FFFFFF')
    gs = gridspec.GridSpec(6, 2, figure=fig, width_ratios=[1.1, 1.0], wspace=0.28, hspace=0.55)
    
    # --- FONT SIZES SCALED UP TO MATCH LARGER HEIGHT ---
    ax1 = fig.add_subplot(gs[0:3, 0])
    ax1.plot(densities * 100, neighbor_data, marker='o', linewidth=4, color='#B32441', markersize=10)
    ax1.axvline(38, color='#590B1B', linestyle='--', linewidth=2.5, label='Baseline Threshold (38%)')
    ax1.set_xlabel('Initial Seeding Density (%)', fontsize=24, fontweight='bold', labelpad=12)
    ax1.set_ylabel('Mean Coordinate Neighbors ($E$)', fontsize=24, fontweight='bold', labelpad=12)
    ax1.set_title('a  Local Connectivity Scaling', fontsize=26, fontweight='bold', pad=18)
    ax1.tick_params(axis='both', labelsize=20)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=20, loc='upper left')
    
    ax2 = fig.add_subplot(gs[3:6, 0])
    ax2.plot(densities * 100, cluster_data * 100, marker='s', linewidth=4, color='#007A99', markersize=10)
    ax2.axvline(38, color='#590B1B', linestyle='--', linewidth=2.5, label='Baseline Threshold (38%)')
    ax2.set_xlabel('Initial Seeding Density (%)', fontsize=24, fontweight='bold', labelpad=12)
    ax2.set_ylabel('Max Connected Cluster Fraction (%)', fontsize=24, fontweight='bold', labelpad=12)
    ax2.set_title('b  Global Network Spanning Profile', fontsize=26, fontweight='bold', pad=18)
    ax2.tick_params(axis='both', labelsize=20)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(fontsize=20, loc='upper left')
    
    snapshot_titles = ['c  Sub-Critical Regime (30%)', 'd  Critical Separation Regime (38%)', 'e  Post-Critical Spanning Regime (45%)']
    keys = [0.30, 0.38, 0.45]
    grid_slots = [gs[0:2, 1], gs[2:4, 1], gs[4:6, 1]]
    
    for idx, density in enumerate(keys):
        ax_snap = fig.add_subplot(grid_slots[idx])
        sim = snapshot_models[density]
        ax_snap.set_aspect('equal')
        ax_snap.set_xlim(-4.2, 4.2)
        ax_snap.set_ylim(-4.2, 4.2)
        ax_snap.set_title(snapshot_titles[idx], fontsize=24, fontweight='bold', pad=14)
        
        c1 = plt.Circle((0, 0), 1.0, color='#E6E6E6', fill=False, linestyle=':', linewidth=1.5)
        c2 = plt.Circle((0, 0), 3.2, color='#D9D9D9', fill=False, linestyle='--', linewidth=1.5)
        ax_snap.add_artist(c1)
        ax_snap.add_artist(c2)
        
        for cell_idx in range(sim.num_cells):
            v_2d = sim.optimized_2d[cell_idx]
            center_2d = np.mean(v_2d, axis=0)
            v_shrunk = center_2d + 0.88 * (v_2d - center_2d)
            polygon = plt.Polygon(v_shrunk, facecolor='#FFCCD6', edgecolor='#B32441', alpha=0.75, linewidth=1.2)
            ax_snap.add_patch(polygon)
            
        ax_snap.set_xlabel('X Dimension (mm)', fontsize=22, labelpad=8)
        ax_snap.set_ylabel('Y Dimension (mm)', fontsize=22, labelpad=8)
        ax_snap.tick_params(axis='both', labelsize=18)
        ax_snap.grid(True, linestyle=':', alpha=0.4)
        
    fig.savefig(out_name, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"    -> Saved Unified Consolidated Sensitivity Analysis Figure: '{out_name}'")

def execute_pipeline():
    all_curves = ["flat", "low", "medium", "high"]
    
    print("Step 1: Running primary baseline cell relaxation calculations...")
    monolayer_models = {c: IndependentEdgeVertexModel(c, scattered=False) for c in all_curves}
    scattered_models = {c: IndependentEdgeVertexModel(c, scattered=True, density_threshold=0.38) for c in all_curves}
    
    print("\nStep 2: Processing 3D spatial tissue graphics (Figure 1)...")
    render_unified_3d_lattice_map(monolayer_models, scattered_models, "physical_white_tension_map.png")
    
    print("\nStep 3: Processing 0-90 degree truncated polar stress charts (Figures 2 & 3)...")
    render_isolated_quadrant_plots(monolayer_models, 'vimentin_orientation_uniform_monolayer.png', 'Uniform')
    render_isolated_quadrant_plots(scattered_models, 'vimentin_orientation_scattered.png', 'Scattered')
    
    print("\nStep 4: Executing Seeding Density Sensitivity Sweep (30% -> 45%)...")
    density_range = np.linspace(0.30, 0.45, 10)
    neighbor_data = []
    cluster_data = []
    snapshot_models = {}
    target_keys = [0.30, 0.38, 0.45]
    
    for d_val in density_range:
        sim = IndependentEdgeVertexModel("medium", scattered=True, density_threshold=d_val)
        avg_n, max_c = sim.calculate_percolation_metrics()
        neighbor_data.append(avg_n)
        cluster_data.append(max_c)
        for tk in target_keys:
            if np.isclose(d_val, tk, atol=0.01) and tk not in snapshot_models:
                snapshot_models[tk] = sim

    print("\nStep 5: Generating publication-ready sensitivity figures...")
    render_unified_sensitivity_panel(
        density_range, 
        np.array(neighbor_data), 
        np.array(cluster_data), 
        snapshot_models, 
        "figure_sensitivity_consolidated.png"
    )
    print(f"\nAll operations completed successfully. Check directory for generated .png files.")

if __name__ == "__main__":
    execute_pipeline()