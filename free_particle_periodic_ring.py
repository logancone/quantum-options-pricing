import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def free_particle(m: float, 
                       L: float,
                       n: int):
    """[replace]

    Args:
        m (float): The mass of the particle (kg)
        L (float): The length of the box (m)
        n (int): Number of steps (accuracy)
        visualize (bool): Pass True if you want to show a graph of the wave functions,
            False otherwise. Defaults to True.
        num_waves_to_show (int): How many waves to show in the visualization graph. Defaults
            to 8.
        show_energy_lines (bool): Pass True if you want to show a light gray line as the energy line,
            False otherwise. Defaults to False.

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
    # Sets potential function to be 0 everywhere
    
    # Sets step size
    h = L / n
    
    # Runs through x (-L/2 -> L/2 but extended by half the range multiple on either side, stepping up by step size)
    x = np.linspace(-10, 10, n, endpoint=False)
    
    equations = []
    
    # i is the true index (starting at 1)
    for i in range(0, len(x)):
        # Sets list of 0s to length of the x-range minus endpoints
        eq = [0.0]*(len(x))

        # Ignore left endpoint
        # if i-1 != 0:
            # Index at i-2 because you need to subtract 1 due to 0-indexing and another 1 due to i-1
        if i-1 != -1:
            eq[i-1] = -(1/(2*m*h**2))
        else:
            # Yes this is redundant, but it makes it more clear imo
            eq[-1] = -(1/(2*m*h**2))
        
        eq[i] = (1/(m*h**2))

        # Ignore right endpoint
        if i != len(x)-1:
            eq[i+1] = -(1/(2*m*h**2))
        else:
            eq[0] = -(1/(2*m*h**2))
        
        equations.append(eq)
        
    equations = np.array(equations)
        
    solution = np.linalg.eigh(equations)
    
    # Calculate initial wavefunction with Gaussian Packet
    k_0 = (2*np.pi*2)/L
    # k_0 = 0
    psi_x_0 = ((1/(2*np.pi*(1**2)))**0.25)*np.exp(-x**2/4*1**2)*np.exp(1j*k_0*x)
        
    # Calculate coefficients to construct new wavefunction
    c_n = []
    for i in range(n):
        c = sum(np.conj(solution.eigenvectors[:,i])*psi_x_0)
        c_n.append(c)
        
    c_n = np.array(c_n)
    
    # new_wavefunction = solution.eigenvectors @ c_n
    prob_dist = np.abs(solution.eigenvectors @ c_n)**2
    
    # Begin evolving through time and check against correct values
    t_n = 5
    dt = 0.1
    
    # t = 0
    # while t < t_n:
    #     # psi_t = solution.eigenvectors @ (c_n * np.exp(-1j*solution.eigenvalues*t))
    #     psi_t = new_wavefunction @ np.exp(-1j*solution.eigenvalues*t)
        
    #     prob_dist = np.abs(psi_t)**2
    #     average_particle_position = np.sum(prob_dist * x) / np.sum(prob_dist)
        
    #     experimental_width = np.sqrt(np.sum((x - average_particle_position)**2 * prob_dist) / np.sum(prob_dist))
        
    #     true_width = 1*np.sqrt(1+(t**2)/(4*1**4))
        
    #     print(f"t: {t:.1f} | Experimental Width: {experimental_width:.4f} | True Width: {true_width:.4f} | Error: {(np.abs((experimental_width-true_width)/true_width)*100):.5f}")
        
    #     t += dt
    fig, ax = plt.subplots()
    line, = ax.plot(x, prob_dist)
    
    def update(frame):
        t = frame * dt
        
        # psi_t = new_wavefunction @ np.exp(-1j*solution.eigenvalues*t)
        psi_t = solution.eigenvectors @ (c_n * np.exp(-1j*solution.eigenvalues*t))
        
        prob_dist = np.abs(psi_t)**2
        average_particle_position = np.sum(prob_dist * x) / np.sum(prob_dist)
        
        experimental_width = np.sqrt(np.sum((x - average_particle_position)**2 * prob_dist) / np.sum(prob_dist))
        
        true_width = 1*np.sqrt(1+(t**2)/(4*1**4))
        
        # print(f"t: {t:.1f} | Experimental Width: {experimental_width:.4f} | True Width: {true_width:.4f} | Error: {(np.abs((experimental_width-true_width)/true_width)*100):.5f}")
        if t <= t_n:
            print(f"t: {t:.1f} | Experimental Width: {experimental_width:.4f} | True Width: {true_width:.4f} | Error: {(np.abs((experimental_width-true_width)/true_width)*100):.5f}")
        
        line.set_ydata(prob_dist)
        
        return line,
    
    anim = FuncAnimation(fig=fig,
                        func=update,
                        frames=2000,
                        interval=10)
    
    plt.title("Probability Distribution")
    plt.xlabel("Position")
    plt.ylabel("Probability")
    plt.show()


if __name__ == "__main__":
    free_particle(m=1, L=20, n=64)