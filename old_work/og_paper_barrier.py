import matplotlib.pyplot as plt

import numpy as np

from qiskit import QuantumCircuit
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit.circuit.library import LinearAmplitudeFunction
from qiskit.primitives import StatevectorSampler
from qiskit_finance.circuit.library.probability_distributions.lognormal import LogNormalDistribution
from qiskit.circuit.library import StatePreparation
from qiskit import QuantumRegister

from scipy.stats import multivariate_normal

# number of qubits to represent the uncertainty
num_uncertainty_qubits = 3

# parameters for considered random distribution
S = 2.0  # initial spot price
vol = 0.4  # volatility of 40%
r = 0.05  # annual interest rate of 4%
T = 40 / 365  # 40 days to maturity

t1 = T/2
t2 = T

mu1 = (r - 0.5 * vol**2) * t1 + np.log(S)
mu2 = (r - 0.5 * vol**2) * t2 + np.log(S)

B = 2.0

cov = vol**2 * np.array([
    [t1, t1],
    [t1, t2]
])


num_price_values = 4

std1 = np.sqrt(cov[0,0])
std2 = np.sqrt(cov[1,1])

low1 = mu1 - 3 * std1
high1 = mu1 + 3 * std1

low2 = mu2 - 3 * std2
high2 = mu2 + 3 * std2

x1 = np.linspace(low1, high1, num_price_values)
x2 = np.linspace(low2, high2, num_price_values)

# Convert to stock prices
S1_values = np.exp(x1)
S2_values = np.exp(x2)

mean_vector = np.array([mu1, mu2])

joint_distribution = multivariate_normal(
    mean=mean_vector,
    cov=cov
)

# Evaluate probability density at each grid point
probabilities = np.zeros((num_price_values, num_price_values))

for i, xi in enumerate(x1):
    for j, xj in enumerate(x2):
        probabilities[i, j] = joint_distribution.pdf([xi, xj])

# Normalize so probabilities sum to 1
probabilities /= np.sum(probabilities)

print("Joint probabilities:")
print(probabilities)

print("\nSum:", np.sum(probabilities))

amplitudes = np.sqrt(probabilities.flatten())

print("Number of amplitudes:", len(amplitudes))
print("Amplitude norm:", np.linalg.norm(amplitudes))


num_uncertainty_qubits_per_time = 2
total_uncertainty_qubits = 4

barrier_call = QuantumCircuit(7)

state_prep = StatePreparation(amplitudes)

barrier_call.append(
    state_prep,
    range(total_uncertainty_qubits)
)

# Check barrier at T/2
# q1 = 1 means S(T/2) >= 2.0
barrier_call.cx(1, 4)

# Check barrier at T
# q3 = 1 means S(T) >= 2.0
barrier_call.cx(3, 5)

# q6 = q4 OR q5
barrier_call.cx(4, 6)
barrier_call.cx(5, 6)
barrier_call.ccx(4, 5, 6)

K = 1.9

low = S2_values[0]
high = S2_values[-1]

payoff_function = LinearAmplitudeFunction(
    num_state_qubits=2,
    slope=1,
    offset=-K,
    domain=(low, high),
    image=(0, high - K),
    breakpoints=[K],
    rescaling_factor=0.25,
)
# barrier_call.add_register(QuantumRegister(1, "barrier"))


# print(qc)

# sigma = vol * np.sqrt(T)
# mean = np.exp(mu + sigma**2 / 2)
# variance = (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)
# stddev = np.sqrt(variance)

# # lowest and highest value considered for the spot price; in between, an equidistant discretization is considered.
# low = np.maximum(0, mean - 3 * stddev)
# high = mean + 3 * stddev

# # construct A operator for QAE for the payoff function by
# # composing the uncertainty model and the objective
# uncertainty_model = LogNormalDistribution(
#     num_uncertainty_qubits, mu=mu, sigma=sigma**2, bounds=(low, high)
# )

# # # plot probability distribution
# # x = uncertainty_model.values
# # y = uncertainty_model.probabilities
# # plt.bar(x, y, width=0.2)
# # plt.xticks(x, size=15, rotation=90)
# # plt.yticks(size=15)
# # plt.grid()
# # plt.xlabel("Spot Price at Maturity $S_T$", size=15)
# # plt.ylabel("Probability", size=15)
# # plt.show()

# # set the strike price (should be within the low and the high value of the uncertainty)
# strike_price = 1.896

# # set the approximation scaling for the payoff function
# c_approx = 0.25

# # setup piecewise linear objective fcuntion
# breakpoints = [low, strike_price]
# slopes = [0.0, 1.0]
# offsets = [0.0, 0.0]
# f_min = 0
# f_max = high - strike_price
# european_call_objective = LinearAmplitudeFunction(
#     num_uncertainty_qubits,
#     slopes,
#     offsets,
#     domain=(low, high),
#     image=(f_min, f_max),
#     breakpoints=breakpoints,
#     rescaling_factor=c_approx,
# )

# # construct A operator for QAE for the payoff function by
# # composing the uncertainty model and the objective
# num_qubits = european_call_objective.num_qubits
# european_call = QuantumCircuit(num_qubits)
# european_call.append(uncertainty_model, range(num_uncertainty_qubits))
# european_call.append(european_call_objective, range(num_qubits))

# # draw the circuit
# print(european_call.draw())

# # plot exact payoff function (evaluated on the grid of the uncertainty model)
# x = uncertainty_model.values
# y = np.maximum(0, x - strike_price)
# plt.plot(x, y, "ro-")
# plt.grid()
# plt.title("Payoff Function", size=15)
# plt.xlabel("Spot Price", size=15)
# plt.ylabel("Payoff", size=15)
# plt.xticks(x, size=15, rotation=90)
# plt.yticks(size=15)
# # plt.show()

# # evaluate exact expected value (normalized to the [0, 1] interval)
# exact_value = np.dot(uncertainty_model.probabilities, y)
# exact_delta = sum(uncertainty_model.probabilities[x >= strike_price])
# print("exact expected value:\t%.4f" % exact_value)
# print("exact delta value:   \t%.4f" % exact_delta)

# print(european_call.draw())

# # set target precision and confidence level
# epsilon = 0.01
# alpha = 0.05

# problem = EstimationProblem(
#     state_preparation=european_call,
#     objective_qubits=[3],
#     post_processing=european_call_objective.post_processing,
# )
# # construct amplitude estimation
# ae = IterativeAmplitudeEstimation(
#     epsilon_target=epsilon, alpha=alpha, sampler=StatevectorSampler(default_shots = 100, seed= 75)
# )

# result = ae.estimate(problem)

# conf_int = np.array(result.confidence_interval_processed)
# print("Exact value:        \t%.4f" % exact_value)
# print("Estimated value:    \t%.4f" % (result.estimation_processed))
# print("Confidence interval:\t[%.4f, %.4f]" % tuple(conf_int))
