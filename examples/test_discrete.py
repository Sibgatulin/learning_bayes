import matplotlib.pyplot as plt
from jax import numpy as np
from jax import random
from numpyro import handlers, util
from numpyro.infer import MCMC, NUTS, Predictive

from learning_bayes import discrete

rng = random.PRNGKey(0)

# |%%--%%|
"""
# Simulate
"""
# |%%--%%|
with handlers.seed(rng_seed=1):
    trace = handlers.trace(discrete.model_gen).get_trace()
print(util.format_shapes(trace))
# |%%--%%|
prob_class_true = np.array([0.3, 0.2, 0.5])
loc_true = np.array([0, 5, 10])
model_generative = handlers.condition(
    discrete.model_gen, {"prob_class": prob_class_true, "loc": loc_true}
)
# |%%--%%|
rng, rng_simulate, rng_infer = random.split(rng, 3)
predictive = Predictive(model_generative, num_samples=100)
samples_prior = predictive(rng_simulate)
# |%%--%%|
obs = samples_prior["likelihood"]
bin_edges = np.linspace(obs.min(), obs.max(), 20)
hist = np.array([np.histogram(a, bins=bin_edges)[0] for a in obs.T])
# |%%--%%|
plt.imshow(hist, extent=tuple(bin_edges[np.array([0, -1])]) + (0, 9))
plt.xlabel("observed value")
plt.ylabel("spatial dim")
plt.title("histogram")
plt.show()
# |%%--%%|
"""
# Infer
"""
# |%%--%%|
with handlers.seed(rng_seed=1):
    trace = handlers.trace(discrete.model_inf).get_trace(obs)
print(util.format_shapes(trace))
# |%%--%%|
kernel = NUTS(discrete.model_inf)
mcmc = MCMC(kernel, num_warmup=500, num_samples=500, num_chains=1)
mcmc.run(rng_infer, y=obs)
mcmc.print_summary()
