import numpy as np

# Reduced Plank Constant (in Joules)
RPC = 1.05457182e-34 #J

convert_to_ev = lambda x: x / 1.602e-19

def finite_square_well(m: float, L: float, V_0: float, n: int):
    """Simulates a finite square well using the Schrodinger equation.

    Args:
        m (float): The mass of the particle (g)
        L (float): The length of the box (m)
        V_0 (float): The potential energy outside the box (J)
        n (int): Number of steps (accuracy)

    Returns:
        A namedtuple with the following attributes:
        
        eigenvalues : (..., M) ndarray
            The eigenvalues in ascending order, each repeated according to
            its multiplicity.
        eigenvectors : {(..., M, M) ndarray, (..., M, M) matrix}
            The column ``eigenvectors[:, i]`` is the normalized eigenvector
            corresponding to the eigenvalue ``eigenvalues[i]``.  Will return a
            matrix object if `a` is a matrix object.
    """
    # Sets potential function to be 0 inside the box (from -L/2 <= x <= L/2) and V_0 everywhere else
    V = lambda x: V_0 if x < -L/2 or x > L/2 else 0
    
    # How many times L the total range runs through (ensures you get values beyond the box)
    RANGE_MULTIPLE = 5
    
    # Sets step size
    h = (RANGE_MULTIPLE*L) / n
    
    # Runs through x (-L/2 -> L/2 but extended by half the range multiple on either side, stepping up by step size)
    x = np.arange(-(RANGE_MULTIPLE/2)*L, (RANGE_MULTIPLE/2)*L + h, h)
    
    equations = []
    
    # i is the true index (starting at 1)
    for i in range(1, len(x)-1):
        # Sets list of 0s to length of the x-range minus endpoints
        eq = [0.0]*(len(x)-2)

        # Ignore left endpoint
        if i-1 != 0:
            # Index at i-2 because you need to subtract 1 due to 0-indexing and another 1 due to i-1
            eq[i-2] = -((RPC**2)/(2*m*h**2))
        
        eq[i-1] = ((RPC**2)/(m*h**2)) + V(x[i])

        # Ignore right endpoint
        if i != len(x)-2:
            eq[i] = -((RPC**2)/(2*m*h**2))
        
        
        equations.append(eq)
        
    equations = np.array(equations)
        
    solution = np.linalg.eigh(equations)
    
    print(solution.eigenvalues[:6])
    
    return solution
    
    
    
if __name__ == "__main__":
    # Example problem
    finite_square_well(m=9.109e-31, L=1e-9, V_0=1.602e-18, n=1000)