import matplotlib.pyplot as plt

import numpy as np

from qiskit_algorithms import IterativeAmplitudeEstimation
from qiskit_algorithms import MaximumLikelihoodAmplitudeEstimation
from qiskit.primitives import StatevectorSampler
from qiskit_finance.circuit.library.probability_distributions.lognormal import LogNormalDistribution

from qiskit_finance.applications.estimation import EuropeanCallPricing

# number of qubits to represent the uncertainty
num_uncertainty_qubits = 3

# parameters for considered random distribution
S = 2.0  # initial spot price
vol = 0.4  # volatility of 40%
r = 0.05  # annual interest rate of 5% **typo??**
T = 40 / 365  # 40 days to maturity

# resulting parameters for log-normal distribution
mu = (r - 0.5 * vol**2) * T + np.log(S)
sigma = vol * np.sqrt(T)
mean = np.exp(mu + sigma**2 / 2)
variance = (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)
stddev = np.sqrt(variance)

# lowest and highest value considered for the spot price; in between, an equidistant discretization is considered.
low = np.maximum(0, mean - 3 * stddev)
high = mean + 3 * stddev

uncertainty_model = LogNormalDistribution(
    num_uncertainty_qubits, mu=mu, sigma=sigma**2, bounds=(low, high)
)

# set the strike price (should be within the low and the high value of the uncertainty)
strike_price = 1.896

exact_value = np.dot(uncertainty_model.probabilities, np.maximum(0, uncertainty_model.values - strike_price))

# set the approximation scaling for the payoff function
c_approx = 0.25

european_call_pricing = EuropeanCallPricing(
    num_state_qubits=num_uncertainty_qubits,
    strike_price=strike_price,
    rescaling_factor=c_approx,
    bounds=(low, high),
    uncertainty_model=uncertainty_model,
)

# set target precision and confidence level
epsilon = 0.01
alpha = 0.05

problem = european_call_pricing.to_estimation_problem()


# construct amplitude estimation (MLAE and IQAE)
mlae = MaximumLikelihoodAmplitudeEstimation(
    evaluation_schedule=[0,1,2,4,8], sampler=StatevectorSampler(default_shots=100, seed=75)
)
mlae_result = mlae.estimate(problem)
mlae_conf_int = mlae.compute_confidence_interval(mlae_result, alpha=alpha)


iqae = IterativeAmplitudeEstimation(
    epsilon_target=epsilon, alpha=alpha, sampler=StatevectorSampler(default_shots = 100, seed= 75)
)
iqae_result = iqae.estimate(problem)

iqae_conf_int = np.array(iqae_result.confidence_interval_processed)

# print results
print("Exact value:        \t%.4f" % exact_value)

print("MLAE Estimated value:    \t%.4f" % (european_call_pricing.interpret(mlae_result)))
print("MLAE Confidence interval:\t[%.4f, %.4f]" % tuple(mlae_conf_int))

print("IQAE Estimated value:    \t%.4f" % (european_call_pricing.interpret(iqae_result)))
print("IQAE Confidence interval:\t[%.4f, %.4f]" % tuple(iqae_conf_int))