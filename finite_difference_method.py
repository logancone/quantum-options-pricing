from collections.abc import Callable

import numpy as np

def finite_difference_method_setup(x_l: float, y_l: float, x_r: float, y_r: float, n: int, P: Callable, Q: Callable, f: Callable):
    """Runs the finite difference method.
    Expects form y'' + P(x)y' + Q(x)y = f(x)

    Args:
        x_l (float): The leftmost x value from the boundary condition.
        y_l (float): The leftmore y value from the boundary condition.
        x_r (float): The rightmost x value from the boundary condition.
        y_r (float): The rightmost y value from the boundary condition.
        n (int): Number of steps (higher = more details)
        P (Callable): The P(x) function from diff eq. Takes in one val and returns one val.
        Q (Callable): The Q(x) function from diff eq. Takes in one val and returns one val.
        f (Callable): The f(x) function from diff eq. Takes in one val and returns one val.
    """ 
    h = (x_r - x_l) / n
    
    x = []
    for i in range(n+1):
        x.append(x_l + (i*h))
    assert x[-1] == x_r
    
    equations = []
    answers = []
    
    # i is the true index (starting at 1)
    for i in range(1, n):
        answer = ((h**2)*f(i))
        
        eq = [0.0]*(n-1)

        # If index is 0, just send constant to the answer side because it is known (left end condition)
        if i-1 == 0:
            answer += (-1) * (1-(h/2)*P(x[i])) * y_l
        else:
            # Index at i-2 because you need to subtract 1 due to 0-indexing and another 1 due to i-1
            eq[i-2] = (1-(h/2)*P(x[i]))
        
        
        eq[i-1] = (-2 + (h**2)*Q(i))
    
        if i == n-1:
            answer += (-1) * (1 + (h/2)*P(i)) * y_r
        else:
            eq[i] = (1 + (h/2)*P(i))
        
        # Substitue ending conditions
        
        equations.append(eq)
        answers.append(answer)
        
    equations = np.array(equations)
    answers = np.array(answers)
    
    for id, eq in enumerate(equations):
        print(eq, answers[id])
        
    return equations, answers
    # solution = np.linalg.solve(equations, answers)
    
    # return solution
    
if __name__ == "__main__":
    eq, ans = finite_difference_method_setup(0, 0, 1, 5, 4, lambda x: 0, lambda x: -4, lambda x: 0)
    
    solution = np.linalg.solve(eq, ans)
    print(solution)
    