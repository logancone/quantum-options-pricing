from collections.abc import Callable

def improved_eulers_method(x_0: float, y_0: float, y_slope: Callable, x_n: float, h: float):
    """Runs improved eulers method.

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
    y_pred = [y_0]
    y = [y_0]
    
    x_n = round(x_n,10)

    print(f"x_{len(x) - 1} = {x[-1]:.2f} | y_{len(y) - 1} = {y[-1]:.4f}")

    while x[-1] < x_n:
        
        y_pred.append(y[-1] + h * y_slope(x[-1], y[-1]))
        x.append(x[-1] + h)
        y.append(y[-1] + (h/2) * (y_slope(x[-2], y[-1]) + y_slope(x[-1], y_pred[-1])))
        
        print(f"x_{len(x) - 1} = {x[-1]:.2f} | y_{len(y) - 1} = {y[-1]:.4f}")
    
    return x, y 

if __name__ == "__main__":
    # Example
    improved_eulers_method(1.0, 1.0, (lambda x, y : 2*x*y), 1.5, 0.05)