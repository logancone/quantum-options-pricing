import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import time

# Reduced Plank Constant (in Joules)
RPC = 1.05457182e-34 #J

j_to_ev = lambda x: x / 1.602e-19
ev_to_j = lambda x: x * 1.602e-19

def finite_square_well(m: float, 
                       L: float, 
                       V_0: float, 
                       n: int, 
                       visualize: bool = True, 
                       num_waves_to_show: int = 8, 
                       show_energy_lines: bool = False):
    """Simulates a finite square well using the Schrodinger equation.
    Visualizes using matplotlib

    Args:
        m (float): The mass of the particle (kg)
        L (float): The length of the box (m)
        V_0 (float): The potential energy outside the box (J)
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
    
    x = x[1:-1]
    
    # Graphing the waves
    if visualize:
        
        # Plot red line for 0 energy (floor)
        plt.axhline(0, color='red')
        
        # Plot the well in black (width of L, height of V_0)
        plt.plot([x[0], -L/2, -L/2, L/2, L/2, x[-1]], [V_0, V_0, 0, 0, V_0, V_0], 'k-', lw=3, label="Potential Well")

        # Wave functions are unitless, so multiply by the same scale as units to fit them into the visualization
        scale_factor = 1e-18
        
        for i in range(0, num_waves_to_show):
            if show_energy_lines:
                # Draw the energy level line
                plt.axhline(y=solution.eigenvalues[i], color='gray', linestyle='--', alpha=0.5)
            
            # Add the energy level to the wave function to seperate them visually
            plt.plot(x, solution.eigenvalues[i] + solution.eigenvectors[:, i] * scale_factor, label=f"E = {solution.eigenvalues[i]}")
            
        plt.show()
    
    return x, solution


def evolve_probability_distribution(energy_levels_to_superpose: list[int], 
                                    dt: float, 
                                    x: np.ndarray,
                                    eigenvalues: np.ndarray, 
                                    eigenvectors: np.ndarray):
    
    c = np.array([(1/np.sqrt(len(energy_levels_to_superpose)))]*len(energy_levels_to_superpose))
    
    fig, ax = plt.subplots()

    # calculate @ t=0
    psi_t = 0
    for i in range(len(energy_levels_to_superpose)):
        psi_t += c[i]*eigenvectors[energy_levels_to_superpose[i]]
        
    probability = np.abs(psi_t)**2
    line, = ax.plot(x, probability)
    
    # Highpoints
    # ax.axvline(x[296])
    # ax.axvline(x[701])
    increasing = True
    last_left_highpoint_val = probability[296]
    
    print(f"Estimated frequency: {(eigenvalues[1]-eigenvalues[0]) / (2*np.pi*RPC)}")
    
    start_frame = 0
    def update(frame):
        t = frame * dt
        
        psi_t = 0
        for i in range(len(energy_levels_to_superpose)):
            psi_t += c[i]*eigenvectors[energy_levels_to_superpose[i]]*np.exp(-1j*eigenvalues[energy_levels_to_superpose[i]]*t / 1.05e-34)
            
        probability = np.abs(psi_t)**2
        
        line.set_ydata(probability)
        
        # nonlocal last_left_highpoint_val, increasing, start_time
        # if increasing == True and probability[296] < last_left_highpoint_val:
        #     end_time = time.perf_counter()
        #     print(f"F: {end_time - start_time} IRL, {(end_time-start_time)*dt/10} True") #10 is the interval of anim
        #     start_time = end_time
        #     increasing = False
        # elif probability[296] > last_left_highpoint_val:
        #     increasing = True
        nonlocal start_frame, increasing, last_left_highpoint_val

        if increasing and probability[296] < last_left_highpoint_val:
            if start_frame is not None:
                period_frames = frame - start_frame
                period = period_frames * dt

                print(f"Freq: {1/period}")

            start_frame = frame
            increasing = False

        elif probability[296] > last_left_highpoint_val:
            increasing = True
        
        last_left_highpoint_val = probability[296]
        
        # print(probability.argmax())
        
        return line,
    
    anim = FuncAnimation(fig=fig,
                        func=update,
                        frames=2000,
                        interval=10)
    
    plt.show()
        
def plot_error_vs_grid_spacing():
    L = 5e-10 #m
    V_0 = ev_to_j(25) #25eV in J\

    m = 9.10938e-31 #electron mass in kg
    
    correct_eigenvalues = (1.12294, 4.46186, 9.90751, 17.16578, 24.78411)
    
    n_values = [25, 50, 100, 250, 500, 1000]
    
    h_values = []
    errored_eigenvalues = []
    eig_1_error = []
    
    for n in n_values:
        h_values.append((5*L)/n)
        x, (eigval, eigvec) = finite_square_well(m=m, L=L, V_0=V_0,n=n,visualize=False)
        errored_eigenvalues.append(tuple(j_to_ev(eigval[:5])))
        
        eig_1_error.append(np.abs(j_to_ev(eigval[0])-correct_eigenvalues[0])/correct_eigenvalues[0])
    
    print(h_values)
    print(eig_1_error)
    plt.loglog(h_values, eig_1_error)
    plt.show()
        
        
    
    
        
    
    
    
if __name__ == "__main__":
    # Example problem
    # x, (eigval, eigvec) = finite_square_well(m=9.109e-31, L=1e-9, V_0=1.602e-18, n=1000, visualize=False, num_waves_to_show=5)
    # evolve_probability_distribution([0,1],3e-17, x, eigval, eigvec)
    plot_error_vs_grid_spacing()

