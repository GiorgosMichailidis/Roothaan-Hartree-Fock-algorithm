This project focuses on the study of the Shannon entropy in atoms across the periodic table. The main objective is to analyze the behavior of the Shannon entropy as a 
function of the atomic number Z. To achieve this, I first computed the electronic wavefunctions of atoms, which form the basis for calculating the Shannon entropy. From the wavefunctions, 
I derived the expression of the electron density, both in the position space ρ(r) and in the momentum space n(k), which are closely related to the Shannon entropy.
For one to obtain the atom wavefunctions, the most common way is the Roothaan-Hartree-Fock method (Bunge et al., 1993). It is an iterative algorithm that approximates atomic orbitals along with the system’s energy.
The notebook was developed during the course Computational Quantum Physics at the M.Sc. in Computational Physics at the Aristotle University of Thessaloniki, Greece. In this repository, you will find:

1. The notebook with the Python implementation of the Roothaan-Hartree-Fock algorithm.
   
2. A PDF report that discusses the implementation of the Roothaan-Hartree-Fock method, presents the mathematical framework used to estimate the electron density
ρ(r) and the wavefunctions, and the results of the analysis that were compared with the findings of Chatzisavvas et al. (2005), upon which this work is based.

3. Related plots: 1) An example of an obtained electron and radial density distribution of the atom Ne, 2) The fitting of the Shannon entropy as a function of Z, 3) An example of an obtained radial orbital wavefunction of the atom Ne, 4) The Shannon entropies compared to the Chatzisavvas et al. (2005) results.
