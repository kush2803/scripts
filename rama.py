import numpy as np
import csv
from ase import Atoms
from ase.io import read, write
from ase.optimize import BFGS
from mace.calculators import mace_mp
#from ase.calculators.aims import Aims  
#from ase.calculators.lj import LennardJones
import matplotlib.pyplot as plt
from ase.constraints import FixInternals
import os

#New method:
#from carmm.run.aims_path import set_aims_command
#set_aims_command(hpc="archer2", basis_set="light", defaults=2020)

# Rotate the dihedral
def rotate(atoms, phi_angle, psi_angle, phi_indices, psi_indices):
    atoms.rotate_dihedral(*phi_indices,phi_angle,indices=[22,23,24,48,49])
    atoms.rotate_dihedral(*psi_indices,psi_angle,indices=[22,23,24,48,49])
    print('Rotated angles', atoms.get_dihedral(*phi_indices), ' ' ,atoms.get_dihedral(*psi_indices))


# Contrain the dihedral
def fix_dihedral(atoms, phi_indexes, psi_indexes, phi_angle, psi_angle):
    c = FixInternals(dihedrals_deg=[
    [atoms.get_dihedral(*phi_indexes),phi_indexes],
    [atoms.get_dihedral(*psi_indexes),psi_indexes]
    ])
    atoms.set_constraint(c)
    print('Contstrained angles', atoms.get_dihedral(*phi_indexes), ' ', atoms.get_dihedral(*psi_indexes))

def calculate_rmsd(traj_file, base_file):
    # Read the base structure
    base = base_file
    
    # Read the trajectory file
    traj = traj_file
    
    # Calculate RMSD
    diff = traj.positions - base.positions
    squared_diff = np.sum(diff**2, axis=1)
    rmsd = np.sqrt(np.mean(squared_diff))
    print(rmsd)
    return rmsd


def heat_plot(phi_angles, psi_angles, energy_map, title = 'rama plot', ylabel = 'rel_energy'):
    plt.contourf(phi_angles, psi_angles, energy_map.T, levels=50, cmap='viridis')
    plt.colorbar(label=ylabel)
    plt.xlabel('Phi (degrees)')
    plt.ylabel('Psi (degrees)')
    plt.title(title)
    plt.savefig(title+'.png')
    
# Initialize molecule 
init= read('base.gjf')

model='~/virtual/mace/lib/python3.10/site-packages/mace/custom_model/large-0b2.model'
# Define the calculator
calc = mace_mp(model=model, default_dtype="float64", device='cpu')
#calc=LennardJones()
#from carmm.run.aims_calculator import get_aims_calculator
#calc = get_aims_calculator(dimensions=0)
#ai_calc = Aims(xc='pbe',
#           spin='none',
#           sc_iter_limit='600',                    
#           relativistic=('atomic_zora','scalar'),
#           compute_forces=True)

            
init.set_calculator(calc)

# Optimize geometry
optimizer = BFGS(init, trajectory = 'base_mace.traj')
optimizer.run(fmax=0.01)  # Convergence criterion


# Get energy 
atoms = read('base_mace.traj@-1')
atoms.set_calculator(calc)

base_energy = atoms.get_potential_energy()
print(f'The base energy is {base_energy}')

# Define dihedral indices 
phi_indices = [5, 4, 1, 48]   #  φ
psi_indices = [19,6,49,48]  #  ψ

# Arrays to store results
phi_angles = np.arange(-20, 24, 4)
psi_angles = np.arange(-20, 24, 4)

#variables for heat map and rmsd
energy_map = np.zeros((len(phi_angles), len(psi_angles)))
rmsd = np.zeros((len(phi_angles), len(psi_angles)))

base_path = os.getcwd()
# Open CSV file to write data
with open('ramachandran_data.csv', mode='w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Phi', 'Psi', 'Relative Energy (eV)', 'RMSD'])  # Header row
    
    # Loop over dihedral angles
    for i, phi in enumerate(phi_angles):
        for j, psi in enumerate(psi_angles):
            try:
                #create a dir
                folder_path = os.path.join(base_path, f'{phi}_{psi}')
                os.makedirs(folder_path, exist_ok=True)
                os.chdir(folder_path)
                
                #copying
                atoms_rot = atoms.copy()
                
                # Set dihedral angles
                rotate(atoms_rot, phi, psi, phi_indices, psi_indices)
                
                #set constraints
                fix_dihedral(atoms_rot, phi_indices, psi_indices, phi, psi)
                
                atoms_rot.set_calculator(calc)
                
                # Optimize geometry
                optimizer = BFGS(atoms_rot, trajectory='traj.traj')
                optimizer.run(fmax=0.01, steps = 700)  # Convergence criterion
                
                #calculate rmsd 
                traj=read('traj.traj@-1')
                rmsd_trial = calculate_rmsd(traj, atoms)
                rmsd[i,j] = rmsd_trial
                
                # Get energy and store in CSV and array
                energy = atoms_rot.get_potential_energy()
                rel_energy = energy - base_energy
                energy_map[i, j] = rel_energy
                csv_writer.writerow([phi, psi, rel_energy, rmsd_trial])  # Write row to CSV
                print(f'{phi}_{psi} done')
                
                os.chdir(base_path)
            
            except Exception as e:
                print(f"Error occurred for phi={phi}, psi={psi}: {str(e)}")
                # placeholder value to the CSV and energy_map
                csv_writer.writerow([phi, psi, "Error", "Error"])
                energy_map[i, j] = float('nan')
                rmsd[i,j] = float('nan')
                continue  


# Plot
heat_plot(phi_angles, psi_angles, energy_map, 'Ramachandran plot', 'Relative energy (eV)')

heat_plot(phi_angles, psi_angles, rmsd, 'RMSD plot', 'Deviation (Ang)')