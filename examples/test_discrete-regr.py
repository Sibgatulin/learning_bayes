import matplotlib.pyplot as plt
from jax import numpy as np
from jax import random
from numpyro import handlers, util
from numpyro.infer import MCMC, NUTS, Predictive

from learning_bayes import discrete

rng = random.PRNGKey(0)

# |%%--%%| <yn4p5NSpYy|nBSmKlJ8Go>
"""
# Simulate
"""
# |%%--%%|
theta = np.linspace(0, 1, 15)
with handlers.seed(rng_seed=1):
    trace = handlers.trace(discrete.model_regression_gen).get_trace(theta)
print(util.format_shapes(trace))
# |%%--%%|
prob_class_true = np.array([0.3, 0.2, 0.5])
# loc_true = np.array([0, 5, 10])
loc_true = np.array([[0, 0], [0.5, 2], [0.0, 4]])
w_true = np.array([[0.0, 0.0], [0.8, 0.2], [0.1, 0.9]])
model_generative = handlers.condition(
    discrete.model_regression_gen,
    {"prob_class": prob_class_true, "loc": loc_true, "weights": w_true},
)
# |%%--%%|
rng, rng_simulate, rng_infer, rng_infer_from_one = random.split(rng, 4)
predictive = Predictive(model_generative, num_samples=100)
samples_prior = predictive(rng_simulate, theta=theta)
# |%%--%%|
obs = samples_prior["likelihood"]
# |%%--%%|
fig, axes = plt.subplots(ncols=2, sharey=True)
for ax, arr in zip(axes, [samples_prior["z"][:1], obs[0]]):
    im = ax.imshow(arr.T)
    plt.colorbar(im, ax=ax, orientation="horizontal")
axes[0].set_ylabel("spatial dim")
axes[0].set_xlabel("samples")
axes[1].set_xlabel("theta")
axes[0].set_title("z")
axes[1].set_title("signal")
plt.show()
# |%%--%%|
"""
# Infer
"""
# |%%--%%|
with handlers.seed(rng_seed=1):
    trace = handlers.trace(discrete.model_regression_inf).get_trace(theta, obs)
    # trace = handlers.trace(model_conditioned).get_trace(theta)
print(util.format_shapes(trace))
# |%%--%%|
kernel = NUTS(discrete.model_regression_inf)
# kernel = NUTS(model_conditioned)
mcmc = MCMC(kernel, num_warmup=500, num_samples=500, num_chains=1)
mcmc.run(rng_infer, theta=theta, y=obs)
mcmc.print_summary()
# |%%--%%|
"""
Looks good
## Test: infer using only one obs (and the generative model without the obs plate)
"""
# |%%--%%|
model_conditioned = handlers.condition(
    discrete.model_regression_gen, {"likelihood": obs[0]}
)
# |%%--%%|
kernel = NUTS(model_conditioned)
mcmc = MCMC(kernel, num_warmup=500, num_samples=500, num_chains=1)
mcmc.run(rng_infer_from_one, theta=theta)
mcmc.print_summary()
# |%%--%%|
"""
Gets harder
"""
# |%%--%%|
