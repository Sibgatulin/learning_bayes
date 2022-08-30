import jax.numpy as np
import numpyro
import numpyro.distributions as dist

NCLASS = 3
SIGMA = 1.0
NSPATIAL = 10
NTHETA = 5


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


def model_regression_gen(theta):
    plate_class = numpyro.plate("class", size=NCLASS, dim=-1)
    plate_spatial = numpyro.plate("spatial", size=NSPATIAL, dim=-1)
    plate_theta = numpyro.plate("theta", size=NTHETA, dim=-2)

    prob_class = numpyro.sample("prob_class", dist.Dirichlet(np.ones(NCLASS)))

    with plate_class:
        loc = numpyro.sample("loc", dist.Normal(0, 1))

    with plate_spatial:
        z = numpyro.sample(
            "z", dist.Categorical(prob_class), infer={"enumerate": "parallel"}
        )
    # broadcast manually (as opposed to using vectorized)
    # because the result must be consistent with plates anyway
    pred = numpyro.deterministic("pred", theta[:, None] * loc[z])
    with plate_spatial, plate_theta:
        numpyro.sample("likelihood", dist.Normal(pred, SIGMA))


def model_inf(y):
    nobs, nspatial = y.shape

    plate_obs = numpyro.plate("obs", size=nobs, dim=-2)
    plate_class = numpyro.plate("class", size=NCLASS, dim=-1)
    plate_spatial = numpyro.plate("spatial", size=nspatial, dim=-1)

    prob_class = numpyro.sample("prob_class", dist.Dirichlet(np.ones(NCLASS)))

    with plate_class:
        loc = numpyro.sample("loc", dist.Normal(0, 1))

    with plate_obs, plate_spatial:
        z = numpyro.sample(
            "z", dist.Categorical(prob_class), infer={"enumerate": "parallel"}
        )
        numpyro.sample("likelihood", dist.Normal(loc[z], SIGMA), obs=y)


def model_regression_inf(theta, y):
    nobs, ntheta, nspatial = y.shape

    plate_class = numpyro.plate("class", size=NCLASS, dim=-1)
    plate_spatial = numpyro.plate("spatial", size=nspatial, dim=-1)
    plate_theta = numpyro.plate("theta", size=ntheta, dim=-2)
    plate_obs = numpyro.plate("obs", size=nobs, dim=-3)

    prob_class = numpyro.sample("prob_class", dist.Dirichlet(np.ones(NCLASS)))

    with plate_class:
        loc = numpyro.sample("loc", dist.Normal(0, 5))

    with plate_obs, plate_spatial:
        z = numpyro.sample(
            "z", dist.Categorical(prob_class), infer={"enumerate": "parallel"}
        )
    # broadcast manually (as opposed to using vectorized)
    # because the result must be consistent with plates anyway
    pred = numpyro.deterministic("pred", theta[:, None] * loc[z])
    with plate_obs, plate_theta, plate_spatial:
        numpyro.sample("likelihood", dist.Normal(pred, SIGMA), obs=y)
