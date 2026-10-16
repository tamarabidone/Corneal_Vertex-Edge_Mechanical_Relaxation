The simulation models both confluent monolayers and scattered patch conditions (simulating lack of cell-cell contact). 
It integrates area dilation constraints, edge tension, and optional intercellular coupling (cadherin/tight junction buffering) optimized via a Nelder-Mead simplex minimization framework. 
It also includes automated 3D lattice mapping and directional edge-tension anisotropy quantification (using Hanning-filtered polar distributions).

Geometric Substrate Modeling: Configurable spherical dome geometries parameterized by base radius and apex height (h), supporting flat, low, medium, and high curvature profiles.
Tissue Topologies: Supports both confluent monolayers and stochastic low-density "scattered patches" via randomized dropout filters.
Energy Minimization Engine: Computes stable physical configurations using area elasticity 
Anisotropy Analysis: Computes edge-orientation angles, quadrant folding symmetry, and Hanning-filtered polar frequency plots for directional tension analysis.

 Prerequisites & Dependencies
Python 3.8+ need to be installed along with Matplotlib:

pip install numpy scipy matplotlib
python corneal_endothelial_vertex_model.py
