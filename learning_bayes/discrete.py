import jax.numpy as np
import numpyro
import numpyro.distributions as dist

NCLASS = 3
SIGMA = 1.0
NSPATIAL = 10
NCOMPONENT = 2


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


def forward(theta, p, w):
    arg = theta * p
    return np.sum(np.exp(arg) * w, axis=-1)


def model_regression_gen(theta):
    ntheta = len(theta)
    plate_component = numpyro.plate("component", size=NCOMPONENT, dim=-1)
    plate_class = numpyro.plate("class", size=NCLASS, dim=-2)
    plate_spatial = numpyro.plate("spatial", size=NSPATIAL, dim=-1)
    plate_theta = numpyro.plate("theta", size=ntheta, dim=-2)

    prob_class = numpyro.sample("prob_class", dist.Dirichlet(np.ones(NCLASS)))

    with plate_class, plate_component:
        loc = numpyro.sample("loc", dist.Normal(0, 1))
    with numpyro.plate("class_", size=NCLASS, dim=-1):
        weights = numpyro.sample("weights", dist.Dirichlet(np.ones(NCOMPONENT)))

    with plate_spatial:
        z = numpyro.sample(
            "z", dist.Categorical(prob_class), infer={"enumerate": "parallel"}
        )

    # the last dim of loc, being a part of the event shape, does not seem to count as
    # loc[z, :] is not allowed
    print(f"{loc[z].shape=}")
    print(f"{weights[z].shape=}")
    print(f"{theta.shape}")
    pred = numpyro.deterministic(
        "pred", forward(theta[:, None, None], loc[z], weights[z])
    )
    with plate_theta, plate_spatial:
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

    plate_component = numpyro.plate("component", size=NCOMPONENT, dim=-1)
    plate_class = numpyro.plate("class", size=NCLASS, dim=-2)
    plate_spatial = numpyro.plate("spatial", size=nspatial, dim=-1)
    plate_theta = numpyro.plate("theta", size=ntheta, dim=-2)
    plate_obs = numpyro.plate("obs", size=nobs, dim=-3)

    prob_class = numpyro.sample("prob_class", dist.Dirichlet(np.ones(NCLASS)))

    with plate_class, plate_component:
        loc = numpyro.sample("loc", dist.Normal(0, 1))
    with numpyro.plate("class_", size=NCLASS, dim=-1):
        weights = numpyro.sample("weights", dist.Dirichlet(np.ones(NCOMPONENT)))

    with plate_obs, plate_spatial:
        z = numpyro.sample(
            "z", dist.Categorical(prob_class), infer={"enumerate": "parallel"}
        )
    pred = numpyro.deterministic(
        "pred", forward(theta[:, None, None], loc[z], weights[z])
    )
    with plate_obs, plate_theta, plate_spatial:
        numpyro.sample("likelihood", dist.Normal(pred, SIGMA), obs=y)
