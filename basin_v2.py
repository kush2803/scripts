import numpy as np
from ase.build import fcc111, add_adsorbate
from ase.optimize import BFGS
from ase.optimize.basin import BasinHopping
from mace.calculators import MACECalculator
from ase.io.trajectory import Trajectory
from ase.io import read
from ase.visualize import view
from ase.constraints import FixAtoms
import os

# Base Setup
slab = fcc111('Cu', size=(4, 4, 4), vacuum=10.0)


# Calc setup
model='medium.model'
calc = MACECalculator(model_paths=model, device='cpu')
slab.calc = calc

def resolve_overlaps(atoms, new_atom_index, cluster_indices, min_dist=1.3):
    """Nudge the new atom upwards if it is too close to any cluster atom."""
    overlap = True
    while overlap:
        overlap = False
        new_pos = atoms.positions[new_atom_index]
        
        for idx in cluster_indices:
            # Calculate distance between new atom and current cluster atom
            dist = atoms.get_distance(new_atom_index, idx)
            
            if dist < min_dist:
                # Nudge the new atom up along the y-axis
                atoms.positions[new_atom_index][1] += 0.2
                overlap = True
                print(f"Overlap detected (dist={dist:.2f}A). Nudging atom up by 0.2A...")
                break # Re-check all distances after the nudge
    return atoms


# Function to add an atom and find the global minimum
def grow_and_optimise(atoms, element, step_number):
    cluster_indices = [a.index for a in atoms if a.symbol == element]
   
    if len(cluster_indices) > 0:
        # Get CM of just the cluster atoms
        cluster_cm = atoms[cluster_indices].get_center_of_mass()
        target_pos = (cluster_cm[0]+2, cluster_cm[1]+2)
        height= 2.0
    else:
        # If it is the first atom, put it at a specific location
        target_pos = (atoms.get_cell()[0,0]/2, atoms.get_cell()[1,1]/2) # middle of slab for now
        height=2.0
        
    add_adsorbate(atoms, element, height=height, position=target_pos)
    #view(atoms)
    
    # The new atom is always the last one added
    new_atom_index = len(atoms) - 1
    
    
    # Run the overlap check before starting Basin Hopping
    if len(cluster_indices) > 0:
        atoms = resolve_overlaps(atoms, new_atom_index, cluster_indices, min_dist=1.3)
    
     
    # Define the Basin-Hopping scout
    bh = BasinHopping(atoms, 
                      temperature=500 * 0.00008617, # Temperature in eV
                      dr=0.5,                  # Distance of random jumps
                      optimizer=BFGS,                # Local relaxation method
                      trajectory = 'lowest.traj',
                      save_hop_trajectories=True,
                      cluster_symbol=element,
                      fmax=0.1,                     # maximal force for the optimizer
                      opt_maxstep=500)                       
    
    # Run the search for 20 'hops'
    bh.run(steps=20)
    print('Basin Hopping complete, going for manual polishing')
    _, opt = bh.get_minimum()
    return opt

# Systematic Growth Loop
# Define your starting directory
base_dir = os.getcwd()

# Systematic Growth Loop
for n in range(1, 4): 
    # Create and enter a new folder for this cluster size
    folder_name = f"size_{n}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Move into the folder
    os.chdir(folder_name)
    
    print(f"Finding global minimum for cluster size {n} in directory {folder_name}...")
    
    # Run the optimization
    # Note: Ensure 'model' path is absolute or correctly relative to this new folder
    slab = grow_and_optimise(slab, 'Pt', n)
    slab.calc = calc
    
    # Final local polish for the current cluster size
    print('Manual polishing starting now...')
    dyn = BFGS(slab, trajectory=f'optimise_{n}.traj')
    dyn.run(fmax=0.01)
    
    energy = slab.get_potential_energy()
    print(f"Energy for size {n}: {energy:.4f} eV")
    
    # Return to the base directory for the next iteration
    os.chdir(base_dir)