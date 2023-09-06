import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro import handlers, util
from numpyro.infer import SVI, Predictive, TraceEnum_ELBO
from numpyro.infer.autoguide import AutoDelta
from numpyro.optim import ClippedAdam

# |%%--%%| <QQQbpnt30T|5EsgyC9nVr>
r"""°°°
# Start with independent sites
°°°"""
# |%%--%%| <5EsgyC9nVr|poJvsDs4hv>


def random_field(grid_shape: tuple[int, ...] = (), n_state=2):
    logit = jnp.zeros(())
    with numpyro.plate("class", n_state):
        mu = numpyro.sample("mu", dist.Normal(jnp.zeros(n_state), jnp.ones(n_state)))
    sigma = numpyro.sample("sigma", dist.HalfCauchy(1.0))

    with numpyro.plate_stack("spatial_dim", grid_shape):
        # with numpyro.plate("spatial_dim_1", grid_shape[-2], dim=-2):
        #     with numpyro.plate("spatial_dim_0", grid_shape[-1], dim=-1):
        # z = numpyro.sample("z", dist.Categorical(prob), infer={"enumerate": "parallel"})
        z = numpyro.sample(
            "z", dist.Bernoulli(logits=logit), infer={"enumerate": "parallel"}
        )
        obs = numpyro.sample("obs", dist.Normal(mu[z], sigma))
    return obs


# |%%--%%| <poJvsDs4hv|nustZeQzGJ>

grid_shape = (10, 10)
n_sample = 4
mu_true = jnp.array([-1, 1])
random_field_conditioned = handlers.condition(random_field, {"mu": mu_true})
samples = Predictive(random_field_conditioned, num_samples=n_sample)(
    random.PRNGKey(0), grid_shape
)
fig, axes = plt.subplots(ncols=2, nrows=n_sample, sharex=True, sharey=True)
for idx, ax_row in enumerate(axes):
    for name, ax in zip(["z", "obs"], ax_row):
        im = ax.imshow(samples[name][idx])
        ax.set_title(f"sigma={samples['sigma'][idx]:.2}")
    plt.colorbar(im, ax=ax_row)
plt.show()

data = samples["obs"][0]
random_field_observed = handlers.condition(random_field, {"obs": data})

# |%%--%%| <nustZeQzGJ|Cj2NZ4aHsn>
r"""°°°
## Inference
°°°"""
# |%%--%%| <Cj2NZ4aHsn|QHXSnaXHPn>

# guide_global = AutoDelta(
#     handlers.block(handlers.seed(random_field_observed, 0), hide=["z"])
# )
#


def guide_local(grid_shape: tuple[int, ...] = (), n_state=2):
    # prob = jnp.ones(grid_shape + (n_state,)) / n_state
    p_mu_loc = numpyro.param("mu_auto_loc", jnp.array([-2.0, 1.4]))
    p_mu_scale = numpyro.param(
        "mu_auto_scale", jnp.ones(n_state) * 0.5, constrain=dist.constraints.positive
    )
    p_sigma = numpyro.param(
        "sigma_auto_scale", jnp.ones(()) * 1e-3, constrain=dist.constraints.positive
    )
    with numpyro.plate("class", n_state):
        numpyro.sample("mu", dist.Normal(p_mu_loc, p_mu_scale))
    numpyro.sample("sigma", dist.Delta(p_sigma))

    with numpyro.plate_stack("spatial_dim", grid_shape):
        p_logit = numpyro.param("prob", jnp.zeros(grid_shape))
        numpyro.sample(
            "z", dist.Bernoulli(logits=p_logit), infer={"enumerate": "parallel"}
        )


# |%%--%%| <QHXSnaXHPn|RuqgXX0iyF>

params = {}
guides = {"local": guide_local}  # , "global": guide_global}
for label, guide in guides.items():
    svi = SVI(
        random_field_observed,
        guide,
        ClippedAdam(5e-5),
        TraceEnum_ELBO(max_plate_nesting=2),
    )
    params[label] = svi.run(
        random.PRNGKey(1), num_steps=100_000, grid_shape=grid_shape, stable_update=True
    )
# |%%--%%| <RuqgXX0iyF|jBP8Q50v7y>
print(f"{params['local'].params['sigma_auto_scale']=}")
print(f"{params['local'].params['mu_auto_loc']=}")
print(f"{params['local'].params['mu_auto_scale']=}")
# print(f"{params['global'].params['mu_auto_loc']=}")
# print(f"{params['global'].params['sigma_auto_loc']=}")
# |%%--%%| <jBP8Q50v7y|hbjOQSPGb7>
plt.plot(params["local"].losses)
plt.show()
# |%%--%%| <hbjOQSPGb7|PIObp2tThO>
fig, axes = plt.subplots(nrows=3, sharex=True, sharey=True)
for ax, arr in zip(axes, [data, samples["z"][0], params["local"].params["prob"]]):
    im = ax.imshow(arr)
    plt.colorbar(im, ax=ax)
plt.show()
