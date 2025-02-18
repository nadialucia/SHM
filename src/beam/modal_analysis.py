import numpy as np
from scipy.linalg import eigh

def calc_modal(elastic_modulus, length, width, height, density, poisson, n_elements, n_eigen, n_mp):
    """
    Calculate modal properties of a beam using FEM
    
    Parameters:
    -----------
    elastic_modulus : array_like
        Young's modulus for each segment (typically 10 segments)
    length : float
        Total length of beam
    width : float
        Width of beam cross-section
    height : float
        Height of beam cross-section
    density : float
        Material density
    poisson : float
        Poisson's ratio (not used in current Euler-Bernoulli formulation)
    n_elements : int
        Number of elements
    n_eigen : int
        Number of eigenvalues/modes to compute
        
    Returns:
    --------
    frequencies : ndarray
        Natural frequencies in Hz
    mode_shapes : ndarray
        Mode shapes (columns are modes, rows are nodal displacements)
    """
    # Derived parameters
    n_nodes = n_elements + 1
    n_dof = n_nodes * 2
    
    # Cross-section properties
    area = width * height
    inertia = width * height**3 / 12
    
    # Support conditions (pin-pin)
    restrained_dof = np.array([0, n_dof-2])  # First and last displacement DOFs
    free_dof = np.setdiff1d(np.arange(n_dof), restrained_dof)
    
    # Element properties
    elements_per_segment = n_elements / len(elastic_modulus)
    element_E = np.repeat(elastic_modulus, elements_per_segment)
    element_EI = element_E * inertia
    
    # Element matrices
    element_length = length / n_elements
    
    # Pre-compute common factors for element matrices
    L = element_length
    L2 = L * L
    L3 = L2 * L
    
    # Element stiffness matrix template
    k_template = np.array([
        [ 12,    6*L,   -12,    6*L],
        [ 6*L,  4*L2,   -6*L,  2*L2],
        [-12,   -6*L,    12,   -6*L],
        [ 6*L,  2*L2,   -6*L,  4*L2]
    ])
    
    # Element mass matrix template
    m_coef = density * area * L / 420
    m_template = np.array([
        [156,     22*L,    54,     -13*L],
        [22*L,    4*L2,    13*L,   -3*L2],
        [54,      13*L,    156,    -22*L],
        [-13*L,   -3*L2,   -22*L,  4*L2]
    ])
    
    # Global matrices
    K = np.zeros((n_dof, n_dof))
    M = np.zeros((n_dof, n_dof))
    
    # Assembly using vectorized operations
    for i in range(n_elements):
        dofs = np.array([2*i, 2*i+1, 2*i+2, 2*i+3])
        np.add.at(K, (dofs[:, None], dofs), k_template * element_EI[i] / L3)
        np.add.at(M, (dofs[:, None], dofs), m_template * m_coef)
    
    # Reduce matrices by removing constrained DOFs
    K_red = K[np.ix_(free_dof, free_dof)]
    M_red = M[np.ix_(free_dof, free_dof)]
    
    # Solve eigenvalue problem
    eigenvalues, eigenvectors = eigh(K_red, M_red)
    
    # Calculate frequencies
    frequencies = np.sqrt(eigenvalues) / (2 * np.pi)
    
    # Reconstruct full mode shapes
    mode_shapes = np.zeros((n_dof, n_eigen))
    mode_shapes[free_dof, :n_eigen] = eigenvectors[:, :n_eigen]
    
    # Check and correct mode shape orientation
    for i in range(n_eigen):
        check_node = n_elements//(2*(i+1))  # Node to check for each mode
        if mode_shapes[check_node, i] < 0:  # Check displacement DOF
            mode_shapes[:, i] *= -1  # Flip sign of entire mode shape
    

    return frequencies[:n_eigen], mode_shapes[::2, :n_eigen][::int(n_elements/(n_mp-1)), :].T
    