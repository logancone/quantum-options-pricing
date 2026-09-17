# Initial conditions
x = [1.0]
y = [1.0]
f = lambda x, y : 2*x*y

# Adjustable parameters
x_n = 1.5
h = 0.05

print(f"x_{len(x) - 1} = {x[-1]:.2f} | y_{len(y) - 1} = {y[-1]:.4f}")

while x[-1] <= x_n:
    y.append(y[-1] + h * f(x[-1], y[-1]))
    x.append(x[-1] + h)
    
    print(f"x_{len(x) - 1} = {x[-1]:.2f} | y_{len(y) - 1} = {y[-1]:.4f}")
