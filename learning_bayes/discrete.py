import jax.numpy as np
import numpyro
import numpyro.distributions as dist

NCLASS = 3
SIGMA = 1.0


def model_gen():
    plate_class = numpyro.plate("class", size=NCLASS, dim=-1)

    prob_class = numpyro.sample("prob_class", dist.Dirichlet(np.ones(NCLASS)))

    with plate_class:
        loc = numpyro.sample("loc", dist.Normal(0, 1))

    z = numpyro.sample(
        "z", dist.Categorical(prob_class), infer={"enumerate": "parallel"}
    )
    numpyro.sample("likelihood", dist.Normal(loc[z], SIGMA))


def model_inf(y):
    nobs = y.size

    plate_obs = numpyro.plate("obs", size=nobs, dim=-1)
    plate_class = numpyro.plate("class", size=NCLASS, dim=-1)

    prob_class = numpyro.sample("prob_class", dist.Dirichlet(np.ones(NCLASS)))

    # loc = numpyro.sample("loc", dist.Normal(np.zeros(NCLASS)).to_event(1))
    with plate_class:
        loc = numpyro.sample("loc", dist.Normal(0, 1))

    with plate_obs:
        z = numpyro.sample(
            "z", dist.Categorical(prob_class), infer={"enumerate": "parallel"}
        )
        numpyro.sample("likelihood", dist.Normal(loc[z], SIGMA), obs=y)
