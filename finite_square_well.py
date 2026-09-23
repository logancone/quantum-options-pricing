from finite_difference_method import finite_difference_method_setup

import numpy as np

# Reduced Plank Constant
RPC = 1.05457182e-34
# m = partical mass, L = box length, V_0 = wall potential
def finite_square_well(m: float, L: float, V_0: float, n: int):
    V = lambda x: V_0 if x < -L/2 or x > L/2 else 0
    
    h = L / n
    
    x = np.arange(-2.5*L, 2.5*L + h, h)
    
    equations = []
    
    # i is the true index (starting at 1)
    for i in range(1, len(x)-1):
        eq = [0.0]*(len(x)-2)

        # If index is 0, just send constant to the answer side because it is known (left end condition)
        if i-1 != 0:
            # Index at i-2 because you need to subtract 1 due to 0-indexing and another 1 due to i-1
            eq[i-2] = -((RPC**2)/(2*m*h**2))
        
        eq[i-1] = ((RPC**2)/(m*h**2)) + V(x[i])
    
        if i != len(x)-2:
            eq[i] = -((RPC**2)/(2*m*h**2))
        
        # Substitue ending conditions
        
        equations.append(eq)
        
    equations = np.array(equations)
    
    # for eq in equations:
    #     print(eq)
    
    # print(equations)
        
    E, phi = np.linalg.eigh(equations)
    # print(E)
    
    # print()
    
    # print(phi)
    
    EeV = E / 1.602e-19
    
    print(EeV)
    return E, phi
    
    
    
if __name__ == "__main__":
    finite_square_well(m=9.109e-31, L=1e-9, V_0=1.602e-18, n=100)