from collections.abc import Callable

def runge_kutta_four(x_0: float, y_0: float, y_slope: Callable, x_n: float, h: float):
    """Runs a fourth degree Runge-Kutta method.

    Args:
        x_0 (float): The initial x value from the initial condition.
        y_0 (float): The initial y value from the initial condition.
        y_slope (Callable): Function representing the first derivative of y.
            should take in an x and a y value and output one slope value.
        x_n (float): The x value to calculate to.
        h (float): The step size for x.
    """
    
    # Initial conditions
    x = [x_0]
    y = [y_0]
    
    k_1 = []
    k_2 = []
    k_3 = []
    k_4 = []
    
    x_n = round(x_n,10)

    print(f"x_{len(x) - 1} = {x[-1]:.2f} | y_{len(y) - 1} = {y[-1]:.4f}")

    while x[-1] < x_n:
        k_1.append(y_slope(x[-1], y[-1]))
        k_2.append(y_slope(x[-1]+0.5*h,y[-1]+0.5*h*k_1[-1]))
        k_3.append(y_slope(x[-1]+0.5*h,y[-1]+0.5*h*k_2[-1]))
        k_4.append(y_slope(x[-1]+h,y[-1]+h*k_3[-1]))
        
        y.append(y[-1]+(h/6)*(k_1[-1]+2*k_2[-1]+2*k_3[-1]+k_4[-1]))
        x.append(x[-1]+h)
    
        print(f"x_{len(x) - 1} = {x[-1]:.2f} | y_{len(y) - 1} = {y[-1]:.4f}")
        
    return x, y
        

if __name__ == "__main__":
    # Example
    runge_kutta_four(1.0, 1.0, (lambda x, y : 2*x*y), 1.5, 0.05)
