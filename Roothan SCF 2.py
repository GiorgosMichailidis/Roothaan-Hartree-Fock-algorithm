from pyscf import gto, scf
from pyscf.gto import basis
import periodictable
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib
from scipy.special import spherical_jn
from scipy.stats import linregress

# To use LaTex in the plots
matplotlib.rcParams[  'text.usetex'  ]      =       True

# A function that computes and plots the radial wavefunctiond of the atoms
def get_orbitals(  Z, n, r  ):

    """
        Inputs:
                Z: The atomic number
                r: The radial distance to evaluate the wavefunction, in Bohr radii
                n: The principal quantum number
        
        Outputs:
                orb: The numerical wavefunctions, evaluated at the radial distance r
    
    """

    # A funtction to run Roothaan method
    def run_SCF(  r , basis  ):

        """
            Inputs:
                    r: The radial distance to evaluate the wavefunction, in Bohr radii
                    basis: The basis function such as Slater-type

            Outputs:
                    orb: The numerical wavefunctions, evaluated at the radial distance r
                    angular: An array that stores the angular quantum number l for each shell 
        
        """
        # Use it to get the element's symbol that corresponds to the Z value
        symbol          =      periodictable.elements[Z].symbol

        # For odd numbers of Z, spin must be 1 for PySCF to build the atom
        if Z  %  2      ==     0:

            spin        =      0

        else:

            spin        =      1

        # Buid the atom, mol is an object that will give us the basis functions
        mol             =      gto.Mole()

        mol.build(
                        atom        =       f'{symbol} 0.0, 0.0, 0.0',
                        basis       =       basis,
                        spin        =       spin

        )
    
        # Run the RHF and extract the coefficients for the wave functions
        mf          =        scf.RHF(  mol  ).run()

        # Coefficients of the molecular orbital 
        C           =        mf.mo_coeff

        # How many orbital the atom has
        shells      =       mol.nbas     

        # To store the quantum number l
        angular     =       []

        for i in range(  shells  ):

            angular.append(  mol.bas_angular(i)  )

        # An array of coordinates where the molecular orbitals are evaluated
        # Each row is a (x, y, z) point and we use it to evaluate the basis functions along the x-axis
        coords      =       np.array(  [  [  ri, 0, 0  ] for ri in r  ]  )
        #                                        ^ y and z will move us around in space away from the nucleus

        # This gives the spherical Gaussian-type orbital (GTO) basis functions at the coordinates
        basis       =       mol.eval_gto(  "GTOval_sph", coords  )
        #                                                 ^ mol.eval_gto requires a 2D array of 3D points
        
        # Radia orbital wavefunctions
        orb         =       np.dot(  basis, C  )
        #                       ^ This multiplies at each point in the r-space the C that corresponds to the orbital with the
        #                         values of the basis functions at these points.

        return orb, angular

    # Run the fuction to get the orbital wavefunctions evaluated at the radial distance r
    orb , angular   =       run_SCF(  r,  basis  =  "sto-3g"  )
    #                                                          ^ Despite the name, STO-3g for PySCF is a
    #                                                            GTO approximation of Slater-type basis
    #       

    fig, ax         =       plt.subplots(  figsize  =  (  10, 5  ), dpi  =  300  )

    # Just a function to ensure that all orbitals are plotted right
    def positive_orb(  p  ):
        
        for i in p:

            if i < 0: # In case an orbital like 2s is upside down

                return -1 * p
            
            else:

                return p

    
    # This is He, it has only 1s
    if  Z   ==  2:
        
        orb[:, 0]       =       positive_orb(  orb[:, 0]  ) / np.max(  np.abs(  orb[:, 0]  )  )

        ax.plot(  r, orb[:, 0], label  =  fr'{1}s'  )

    else: # Above 1s orbitals i.e.; 2s, 2p

        for i in range(  orb.shape[1]  -  2  ):  # Loop over MOs -columns of C
        #                                 ^ since we go until 2p then the rest columns with index 3, 4 are not needed
            
            # Plots for elements with s orbitals only
            if angular[i]       ==      0:
                
                orb[:, i]       =       positive_orb(  orb[:, i]  ) / np.max(  np.abs(  orb[:, i]  )  )
        
                ax.plot(  r, orb[:, i], label  =  fr'{i + 1}s'  )
            
            # After Z = 4, there are elemnts with p orbitals, plot them too
            elif angular[i]     ==      1       and     Z   >   4:

                orb[:, i]       =       positive_orb(  orb[:, i]  ) / np.max( np.abs(  orb[:, i]  )  )

                ax.plot(  r, orb[:, i], label  =  fr'{i}p'  )

    ax.axhline(  0, color = 'grey', ls ='--'  )

    ax.set_xlabel(  r"\textbf{r}" , fontsize  =  15  )

    ax.set_ylabel(  r"\textbf{Wavefunction}~[\textit{arbitrary~units}]", fontsize  =  15  )

    ax.set_title(  fr"\textbf{{Radial orbital wavefunctions of {periodictable.elements[Z].symbol}}}"  )

    plt.grid(  True  )

    plt.legend()

    return orb

# This a function to compute the Shannon entropy, both in radial space and momentum space
def func_to_get_entropies(  Z, n, r, k  ):

    """
    Inputs:
            Z: The atomic number
            r: The radial distance to evaluate the wavefunction, in Bohr radii
            n: The principal quantum number
            k: The momentum space to evaluate the wavefunctions
    
    Outputs:
            Sr: The Shannon entropy in the radial space
            Sk: The Shannon entropy in the momentum space
    
    """

    # These is a (500, 5) array of the MO evaluated at the radial distsnce r
    orbitals        =       get_orbitals(  Z, n, r  )

    # Fourier transform the φ(r) to φ(k), according to equation (29) in Chatzisavvas et al. 2005
    def phi_k( r, k, phi_r  ):

        """
        Inputs:
                phi_r: The evaluated wavefunctions in the radial space
                r: The radial distance to evaluate the wavefunction, in Bohr radii
                k: The momentum space to evaluate the wavefunctions
        
        Outputs:
                result: An array with the values of the wavefunction in the momentum space

        """
        result      =       []

        for k_val in k: # Compute the radial integral for every value of k

            if Z  ==  2: # This is for He
#                                                                           ^ The computation is more accurate for He, if we use the Jo Bessel function    
                I       =       4 * np.pi * r ** 2 * phi_r * spherical_jn(  0, k_val * r  ) 
        
                result.append(  np.trapz( I, r  )  )

            else: # For the rest of the atoms
#                                                                           ^ J2 Bessel function
                I       =       4 * np.pi * r ** 2 * phi_r * spherical_jn(  2, k_val * r  ) 
    
                result.append(  np.trapz( I, r  )  )

        return np.array(  result  )
    
    rho             =       np.sum(  orbitals ** 2, axis  =  1  )
    #                                   ^ Square the wavefunctions, because the orbitals are real and Σ φ*φ = Σ φ**2

    r_val           =       r.reshape(  -1, 1  ) # Reshape the radial distance in a column array

    # Make copies of the column with tile. This creates an array with the radial distance values in each column. The number of columns is the number of orbitals in the atom
    r_val           =       np.tile(  r_val, (  1, orbitals.shape[1]  )  ) 

    # Initialize an array to store the values from the φ(r) to φ(k) transform
    phik            =       np.zeros(  shape  = (  orbitals.shape[0], orbitals.shape[1]  ) )

    # Transform each orbital from r to k
    for i in range(  orbitals.shape[1]  ):
        
        phik[:, i]  =       phi_k(  r, k, orbitals[:, i]  ) # Each column in the array, is the values of the wavefunction in momentum space

    # Similar with before: φ(k) = Σ φ*φ = Σ φ**2
    nk              =       np.sum( phik ** 2,  axis = 1 )

    # Normalize rho so the 4π * int( r**2 * ρ  ) gives the number of electrons Z in the atom
    unorm_int       =       4 * np.pi * np.trapz(  r ** 2 * rho, r  )

    rho_norm        =       rho  /  unorm_int

    # Normalize nk as well
    unorm_int_nk    =       4 * np.pi * np.trapz(  k ** 2 * nk, k  )

    nk_norm         =       nk  /  unorm_int_nk

    # The radial distance function 4πr^2 * ρ
    radial_density  =       4 * np.pi * r**2 * rho_norm

    fig, axs        =       plt.subplots(  figsize  =  (  10, 5  ),  dpi  =  300  )

    axs.plot(  r, rho_norm, label  = r'$\rho(\mathbf{r})$')

    axs.plot(  r, radial_density, label  =  r'$4\cdot \pi \cdot r^2 \cdot \rho(\mathbf{r})$')

    axs.axhline(  0, color = 'grey', ls = '--'  )

    axs.set_xlabel(  r"\textbf{r}" , fontsize  =  15  )

    axs.set_title(  fr"\textbf{{Electon and radial density distribution of {periodictable.elements[Z].symbol}}}"  )

    axs.grid(True)

    axs.legend()

    # The Shannon entropy integral in radial space
    def shannon(  r, rho  ):
        return np.trapz(  -4 * np.pi * r**2 * rho * np.log(  rho  ), r  )

    Sr      =        shannon(  r, rho_norm  )

    # The Shannon entropy integral in momentum space
    def shannon_k(  k, n_k  ):

        return np.trapz(  -4 * np.pi * k**2 * n_k * np.log(  n_k  ), k  )

    Sk      =        shannon_k(  k, nk_norm  )

    return  Sr, Sk

# Radial range 
r      =       np.linspace(  1e-20, 5, 500 )
#                               ^ Very small value to avoid computation troubles with zero

# The momentum
k               =       np.linspace(  1e-20, 5, 500  )

# Define the range of Z
Z       =       np.arange(  2, 11, 1  )

shannon_r     =       []

shannon_k     =       []

# Run a loop for all Z to compute the entropies
for z in Z:

    if z  ==  2:  # This is He, n = 1

        n       =       1

        Sr, Sk  =       func_to_get_entropies(  z, n, r, k  )

        shannon_r.append(  Sr  )

        shannon_k.append(  Sk  )

    else:

        n       =       2

        Sr, Sk  =       func_to_get_entropies(  z, n, r, k  )

        shannon_r.append(  Sr  )

        shannon_k.append(  Sk  )

# Take the paper values, for comparison in the plots
shannon_r_paper       =       np.array( [   2.69851, 3.07144, 3.62386, 3.40545, 3.10602, 2.80169, 2.55054, 2.29883, 2.05514  ]  )

shannon_k_paper       =       np.array(  [  3.91342, 3.99682, 4.19019, 4.70590, 5.15658, 5.54934, 5.86737, 6.16333, 6.43707  ]  )

shannon_paper         =       np.array(  [  6.61193, 7.69826, 7.81405, 8.11135, 8.26260, 8.35103, 8.41791, 8.46215, 8.49221  ]  )

# The total entropy                                ^ Sr + Sk
shannon               =       np.array(  shannon_r  ) + np.array(  shannon_k  )

fig, ax               =       plt.subplots(  1, 3,  figsize = (  15, 5  ), dpi  =  300  )

ax[0].plot(  Z, shannon_r, '-o', color = 'black', label = 'Mine')

ax[0].plot(  Z, shannon_r_paper, '-o', color = 'red', label = 'Paper')

ax[0].set_xlabel( r"\textbf{Z}" , fontsize  =  15  )

ax[0].set_ylabel( r"$\mathbf{S_r}$" , fontsize  =  15 )

ax[0].set_title(  r"$\mathbf{Shannon~entropy (S_r)}$" )

ax[0].legend()

ax[1].plot(  Z, shannon_k, '-o', color = 'black', label = 'Mine'  )

ax[1].plot(  Z, shannon_k_paper, '-o', color = 'red', label = 'Paper'  )

ax[1].set_xlabel( r"\textbf{Z}" , fontsize  =  15  )

ax[1].set_ylabel( r"$\mathbf{S_k}$" , fontsize  =  15 )

ax[1].set_title(  r"$\mathbf{Shannon~entropy (S_k)}$" )

ax[1].legend()

ax[2].plot(  Z, shannon_paper, '-o', color = 'black', label = 'Mine')

ax[2].plot(  Z, shannon, '-o', color = 'red', label = 'Paper')

ax[2].set_xlabel( r"\textbf{Z}" , fontsize  =  15  )

ax[2].set_ylabel( r"$\mathbf{S}$" , fontsize  =  15 )

ax[2].set_title(  r"$\mathbf{Shannon~entropy (S)}$" )

ax[2].legend()

#fig.savefig(  "Shannon entropies.jpg", dpi = 300  )
plt.tight_layout()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                                                                                    #  
#                                                                                                    #
#                           Fit an exponential function to the entropy data                          #
#                                                                                                    #  
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# The atomic numbers
Z_tot       =       np.arange(  1, 55, 1  )

# The total entropy values from Chatzisavvas et al. 2005
S_tot       =       np.array(  [ 
                                  6.56659, 6.61193, 7.69826, 7.81405, 8.11135, 8.26260, 8.35103, 8.41791, 8.46215,
                                  8.49221, 8.81319, 8.91038, 9.06497, 9.15294, 9.20767, 9.24871, 9.27418, 9.28924,
                                  9.47419,9.54334, 9.60143, 9.64548, 9.68229, 9.70724, 9.73945, 9.76478, 9.78613, 
                                  9.71120, 9.81401, 9.83493, 9.89832, 9.93896, 9.96808, 9.99276, 10.01020, 10.02240, 
                                  10.12060, 10.16370, 10.20480, 10.23520, 10.25720, 10.27350, 10.29840, 10.30610, 
                                  10.31840, 10.29390, 10.33740, 10.35740, 10.40080, 10.43000, 10.45170, 10.47060, 
                                  10.48490, 10.49600
                                  ]  )

fig, ax     =       plt.subplots(  figsize = (  10, 8  ), dpi = 300  )

ax.plot(  Z_tot, S_tot, '-o', color = 'black'  )

ax.set_xlabel( r"\textbf{Z}" , fontsize  =  15  )

ax.set_ylabel( r"$\mathbf{S}$" , fontsize  =  15 )

ax.set_title(  r"$\mathbf{Shannon~entropy (S)}$" )

# Make the data in logarithmic scale so the relation is linear
logZ        =       np.log(  Z_tot  )

logS        =       np.log(  S_tot  )

# Use Scipy and fit the linear function
result      =       linregress(  logZ, logS  )

# Slope and intercept of the linear function
intercept   =       result.intercept
slope       =       result.slope

#                      ^ Intercept is logA                  ^ Fit the e^(β * logx )
ax.plot(  Z_tot, np.exp(  intercept  ) * np.exp( slope * np.log(  Z_tot  )  ), color = 'red', label = r'Fitting: $S = \beta\cdot e^{logx}$' )

ax.annotate(  fr'Slope: $0.119~\pm~ {result.stderr:.3f}$', xy = (  0, 10.4  )  )

ax.annotate(  fr'Intercept: $1.884~\pm~ {result.intercept_stderr:.3f}$', xy = (  0, 10.3  )  )
#                                          ^ To get the erros of the slope and intercept

ax.legend()

fig.savefig(  'Entropy fitting', dpi = 300  )