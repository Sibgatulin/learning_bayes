import jax.numpy as np
import numpyro
import numpyro.distributions as dist

NCLASS = 3
SIGMA = 1.0
NSPATIAL = 10


def model_gen():
    plate_class = numpyro.plate("class", size=NCLASS, dim=-1)
    plate_spatial = numpyro.plate("spatial", size=NSPATIAL, dim=-1)

    prob_class = numpyro.sample("prob_class", dist.Dirichlet(np.ones(NCLASS)))

    with plate_class:
        loc = numpyro.sample("loc", dist.Normal(0, 1))

    with plate_spatial:
        z = numpyro.sample(
            "z", dist.Categorical(prob_class), infer={"enumerate": "parallel"}
        )
        numpyro.sample("likelihood", dist.Normal(loc[z], SIGMA))


def model_inf(y):
    nobs, nspatial = y.shape

    plate_obs = numpyro.plate("obs", size=nobs, dim=-2)
    plate_class = numpyro.plate("class", size=NCLASS, dim=-1)
    plate_spatial = numpyro.plate("spatial", size=nspatial, dim=-1)

    prob_class = numpyro.sample("prob_class", dist.Dirichlet(np.ones(NCLASS)))

    # loc = numpyro.sample("loc", dist.Normal(np.zeros(NCLASS)).to_event(1))
    with plate_class:
        loc = numpyro.sample("loc", dist.Normal(0, 1))

    with plate_obs, plate_spatial:
        z = numpyro.sample(
            "z", dist.Categorical(prob_class), infer={"enumerate": "parallel"}
        )
        numpyro.sample("likelihood", dist.Normal(loc[z], SIGMA), obs=y)
