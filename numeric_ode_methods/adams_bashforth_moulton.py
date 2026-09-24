from collections.abc import Callable
from numeric_ode_methods.runge_kutta_four import runge_kutta_four

def adams_bashforth_moulton(x_0: float, y_0: float, y_slope: Callable, x_n: float, h: float):
    """Runs the Adams Bashforth Moulton method.
    Uses Runge Kutta Four to get the inital y values

    Args:
        x_0 (float): The initial x value from the initial condition.
        y_0 (float): The initial y value from the initial condition.
        y_slope (Callable): Function representing the first derivative of y.
            should take in an x and a y value and output one slope value.
        x_n (float): The x value to calculate to.
        h (float): The step size for x.
    """
    # Use runge kutta to get the first four terms of y
    temp_x_n = x_0 + h*3
    print("Using RK4")
    x, y = runge_kutta_four(x_0, y_0, y_slope, temp_x_n, h)
    y_pred = y.copy()

    x_n = round(x_n, 10)
    print("Starting ABM")
    while x[-1] < x_n:
        print(x[-1], x_n)
        y_pred.append(y[-1] + (h/24) * 
                      (55*y_slope(x[-1],y[-1]) - 
                       59*y_slope(x[-2],y[-2]) + 
                       37*y_slope(x[-3],y[-3]) - 
                       9*y_slope(x[-4],y[-4])))
        
        x.append(x[-1] + h)
        
        y.append(y[-1] + (h/24) * 
                 (9*y_slope(x[-1],y_pred[-1]) + 
                  19*y_slope(x[-2],y[-1]) - 
                  5*y_slope(x[-3],y[-2]) + 
                  y_slope(x[-4],y[-3])))
        
        print(f"x_{len(x) - 1} = {x[-1]:.2f} | y_{len(y) - 1} = {y[-1]:.9f}")
    
    return x, y 
    

if __name__ == "__main__":
    adams_bashforth_moulton(0, 1, (lambda x, y: x+y-1), 0.8, 0.2)
    