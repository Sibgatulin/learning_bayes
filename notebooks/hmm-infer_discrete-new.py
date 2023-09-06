from pprint import pprint

import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
from jax import numpy as jnp
from jax import random
from numpyro import handlers
from numpyro.contrib.control_flow import scan
from numpyro.contrib.funsor import config_enumerate, infer_discrete, markov
from numpyro.infer import MCMC, NUTS, SVI, Predictive, TraceEnum_ELBO, autoguide
from numpyro.optim import Adam

rng_key = random.PRNGKey(0)

#|%%--%%| <81iIncQUk9|wPMnjPNZDH>

nt = 20
nstate = 2

# |%%--%%| <wPMnjPNZDH|XlxXYzbwPo>

prior = {
    "mu_loc": jnp.array([-1.0, 1.0]),
    "mu_scale": jnp.ones(nstate),
    "alpha": 0.6 * jnp.eye(2) + 0.2,
    "conc": 10.0,
    "sigma_loc": 1.0,
    "init": jnp.ones(nstate) / nstate,
}


def define_global_sites(prior):
    with numpyro.plate("state", nstate, dim=-1):
        mu = numpyro.sample("mu", dist.Normal(prior["mu_loc"], prior["mu_scale"]))

    sigma = numpyro.sample("sigma", dist.Exponential(1.0 / prior["sigma_loc"]))
    s_m1 = numpyro.sample(
        "s_-1",
        dist.Categorical(prior["init"]),
    )
    trans_prob = numpyro.sample(
        "trans", dist.Dirichlet(prior["alpha"] * prior["conc"]).to_event(1)
    )
    return mu, sigma, s_m1, trans_prob


def model_scan(prior, obs=None, n=None):
    if obs is None:
        assert n is not None
    else:
        if n is None:
            n = len(obs)
        else:
            assert n == len(obs)

    mu, sigma, s_m1, trans_prob = define_global_sites(prior)

    def transition(s_prev, y_obs):
        s = numpyro.sample("s", dist.Categorical(trans_prob[s_prev]))
        y = numpyro.sample("y", dist.Normal(mu[s], sigma), obs=y_obs)
        return s, (s, y)

    _, (s, y) = scan(transition, s_m1, obs, length=n)
    return (s, y)


def model_markov(prior, obs=None, n=None):
    if obs is None:
        assert n is not None
        obs = [None] * n
    else:
        if n is None:
            n = len(obs)
        else:
            assert n == len(obs)

    mu, sigma, s_m1, trans_prob = define_global_sites(prior)

    states = [s_m1]
    for t in markov(range(n)):
        s = numpyro.sample(f"s_{t}", dist.Categorical(trans_prob[states[-1]]))
        y = numpyro.sample(f"y_{t}", dist.Normal(mu[s], sigma), obs=obs[t])

        states.append(s)
    return states


# |%%--%%| <XlxXYzbwPo|tIicWgRmUa>
r"""°°°
# Generate data
Simulate 10 chains and chose one with some action
°°°"""
# |%%--%%| <tIicWgRmUa|4QweeS2WDX>

samples = Predictive(model_scan, num_samples=10, parallel=True)(
    rng_key, prior | {"mu_loc": jnp.array([-3, 2])}, n=nt
)
fig, axes = plt.subplots(nrows=10, figsize=(10, 20), sharex=True, sharey=True)
for idx, ax in enumerate(axes):
    ax.twinx().plot(samples["s"][idx], label="s", c="k", lw=2)
    ax.plot(samples["y"][idx], label="s", c="C0")
axes[-1].set_xlabel("t")
plt.show()

# |%%--%%| <4QweeS2WDX|xX0Xo1Qajk>

idx = 1
gt = {k: v[idx] for k, v in samples.items()}

# |%%--%%| <xX0Xo1Qajk|g9bTbaYt6m>

plt.plot(gt["y"])
plt.xlabel("t")
plt.ylabel("y")
plt.show()

# |%%--%%| <g9bTbaYt6m|HtnoeIiDIH>
r"""°°°
# Learn model's continuous parameters with NUTS + enumeration
## Scan
°°°"""
# |%%--%%| <HtnoeIiDIH|zM1QaB8ISu>

kernel = NUTS(config_enumerate(model_scan))
mcmc = MCMC(
    kernel,
    num_warmup=500,
    num_samples=1_000,
    num_chains=2,
)
rng_key, rng_mcmc, rng_inf, rng_svi = random.split(rng_key, 4)
mcmc.run(rng_mcmc, prior=prior, obs=gt["y"])
mcmc.print_summary()
posterior = mcmc.get_samples(group_by_chain=True)
print(posterior["mu"].mean(1))

# |%%--%%| <zM1QaB8ISu|XKXf0ngS1v>
r"""°°°
## Markov
°°°"""
# |%%--%%| <XKXf0ngS1v|6sZU2UHTSE>

kernel = NUTS(config_enumerate(model_markov))
mcmc = MCMC(
    kernel,
    num_warmup=500,
    num_samples=1_000,
    num_chains=2,
)
rng_key, rng_mcmc, rng_inf, rng_svi = random.split(rng_key, 4)
mcmc.run(rng_mcmc, prior=prior, obs=gt["y"])
mcmc.print_summary()
posterior = mcmc.get_samples(group_by_chain=True)
print(posterior["mu"].mean(1))

# |%%--%%| <6sZU2UHTSE|yEp1E8Zc9Z>

posterior_mean = {k: v.mean((0, 1)) for k, v in posterior.items()}
pprint(posterior_mean)
if posterior_mean["mu"][1] < posterior_mean["mu"][0]:
    posterior_mean["mu"] = posterior_mean["mu"][::-1]
    posterior_mean["trans"] = posterior_mean["trans"][::-1, ::-1]
pprint(posterior_mean)

# |%%--%%| <yEp1E8Zc9Z|1kYMh5ViXg>

# following fails due to the stalling PR: https://github.com/pyro-ppl/numpyro/pull/991
# model_scan_conditioned = handlers.condition(model_scan, posterior_mean)
# infer_discrete(
#     config_enumerate(model_scan_conditioned), first_available_dim=-2, temperature=0
# )(prior=prior, obs=gt["y"])

# |%%--%%| <1kYMh5ViXg|g1va3YSor5>

model_markov_conditioned = handlers.condition(model_markov, posterior_mean)
states = infer_discrete(
    config_enumerate(model_markov_conditioned), first_available_dim=-2, temperature=0
)(prior=prior, obs=gt["y"])
states = jnp.stack(states)

plt.plot(gt["s"], label="gt")
plt.plot(states[1:], label="inferred")
plt.show()

# |%%--%%| <g1va3YSor5|zH4ofvP9dN>
r"""°°°
# Can I use TraceEnum_ELBO?
## With scan
°°°"""
#|%%--%%| <diQ1QLQTuN|ykSBBpRp2x>

guide_global = autoguide.AutoDelta(
    handlers.block(
        handlers.seed(
            lambda prior, obs=None, n=None: define_global_sites(prior), rng_seed=0
        ),
        hide=["s_-1"],
    )
)

# |%%--%%| <diQ1QLQTuN|diQ1QLQTuN>

svi = SVI(
    config_enumerate(model_scan),
    guide_global,
    Adam(1e-3),
    TraceEnum_ELBO(max_plate_nesting=1),
)
result = svi.run(rng_svi, num_steps=10_000, prior=prior, obs=gt["y"])
result.params
# Not terribly robust at all

# |%%--%%| <diQ1QLQTuN|juSpEfpgZ3>
r"""°°°
## With Markov
°°°"""
# |%%--%%| <juSpEfpgZ3|AeiflIk2TK>

svi = SVI(
    config_enumerate(model_markov),
    guide_global,
    Adam(5e-3),
    TraceEnum_ELBO(max_plate_nesting=1),
)
result = svi.run(rng_svi, num_steps=10_000, prior=prior, obs=gt["y"])
result.params
# works

# |%%--%%| <AeiflIk2TK|8LT1cHr7LC>

params = {k.replace("_auto_loc", ""): v for k, v in result.params.items()}
pprint(params)
if params["mu"][1] < params["mu"][0]:
    params["mu"] = params["mu"][::-1]
    params["trans"] = params["trans"][::-1, ::-1]
pprint(params)

# |%%--%%| <8LT1cHr7LC|UUeBQtC3FU>

model_markov_conditioned = handlers.condition(model_markov, params)
states = infer_discrete(
    config_enumerate(model_markov_conditioned), first_available_dim=-2, temperature=0
)(prior=prior, obs=gt["y"])
states = jnp.stack(states)

plt.plot(gt["s"], label="gt")
plt.plot(states[1:], label="inferred")
plt.show()

# |%%--%%| <UUeBQtC3FU|btA0no9vSs>
r"""°°°
# Can I use guide-side enumeration?
°°°"""
# |%%--%%| <btA0no9vSs|amHSe6cCwl>


def guide_markov(prior, obs=None, n=None):
    if obs is None:
        assert n is not None
        obs = [None] * n
    else:
        if n is None:
            n = len(obs)
        else:
            assert n == len(obs)

    guide_global(prior)
    s_logits = numpyro.param("s_logits", jnp.zeros((nt + 1, nstate)))

    for t in markov(range(n + 1)):
        numpyro.sample(f"s_{t-1}", dist.Categorical(logits=s_logits[t]))
    return []


svi = SVI(
    config_enumerate(model_markov),
    guide_markov,
    Adam(5e-3),
    TraceEnum_ELBO(max_plate_nesting=1),
)
result = svi.run(rng_svi, num_steps=10_000, prior=prior, obs=gt["y"])
result.params
# works

# |%%--%%| <amHSe6cCwl|aAczRApG7E>

params = {k.replace("_auto_loc", ""): v for k, v in result.params.items()}
pprint(params)
if params["mu"][1] < params["mu"][0]:
    params["mu"] = params["mu"][::-1]
    params["trans"] = params["trans"][::-1, ::-1]
    params["s_logits"] = params["s_logits"][..., ::-1]
pprint(params)

# |%%--%%| <aAczRApG7E|3GzZpvNFiV>

s_prob = dist.transforms.SigmoidTransform()(params["s_logits"])

plt.plot(gt["s"], label="gt")
plt.plot(s_prob[1:, 1], label="inferred")
plt.show()

# |%%--%%| <3GzZpvNFiV|ukJNSJ7w9q>
r"""°°°
## With scan...
°°°"""
# |%%--%%| <ukJNSJ7w9q|8yU9XxrQJR>


def guide_scan(prior, obs=None, n=None):
    if obs is None:
        assert n is not None
    else:
        if n is None:
            n = len(obs)
        else:
            assert n == len(obs)

    guide_global(prior)
    s_logits = numpyro.param("s_logits", jnp.zeros((n + 1, nstate)))
    numpyro.sample(
        "s_-1",
        dist.Categorical(logits=s_logits[0]),
    )

    def transition(carry, s_logit):
        s = numpyro.sample("s", dist.Categorical(logits=s_logit))
        return carry, s

    _, s = scan(transition, None, s_logits[1:], length=n)
    return s, obs


svi = SVI(
    config_enumerate(model_scan),
    guide_scan,
    Adam(5e-3),
    TraceEnum_ELBO(max_plate_nesting=1),
)
result = svi.run(rng_svi, num_steps=10_000, prior=prior, obs=gt["y"])
result.params
# works

# |%%--%%| <8yU9XxrQJR|a88gOFnXJZ>

params = {k.replace("_auto_loc", ""): v for k, v in result.params.items()}
pprint(params)
if params["mu"][1] < params["mu"][0]:
    params["mu"] = params["mu"][::-1]
    params["trans"] = params["trans"][::-1, ::-1]
    params["s_logits"] = params["s_logits"][..., ::-1]
pprint(params)

# |%%--%%| <a88gOFnXJZ|syJDyRpCge>

s_prob = dist.transforms.SigmoidTransform()(params["s_logits"])

plt.plot(gt["s"], label="gt")
plt.plot(s_prob[1:, 1], label="inferred")
plt.show()
