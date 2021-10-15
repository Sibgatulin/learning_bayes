import argparse
import torch
import pyro.contrib.funsor
from pyro.infer.autoguide import AutoDelta
from pyroapi import distributions as dist
from pyroapi import handlers, infer, pyro, pyro_backend

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default="contrib.funsor", type=str)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--print-shapes", action="store_true")

X0 = 5.0
SIGMA = 1.0
BETA = (10.0, 1.0)
DATA = torch.tensor([20.0, 40.0, 50.0, 60.0, 60.0])


def model(nstate=3, data=DATA):
    """Simple Switching Linear Dynamical System:

    dx/dt = β s, modelled as

    β ~ N(μ_β, σ_β)
    s[0] ~ Cat(w0)
    x[0] ~ N(µ_x, σ_x)
    s[t] ~ Cat(T s[t-1])
    x[t] ~ N(x[t-1] + β s[t-1], σ_x)
    """
    nt = len(data)

    probs_trans = pyro.sample(
        "transition",
        dist.Dirichlet(torch.ones((nstate, nstate))).to_event(1),
    )

    s = pyro.sample(
        "s_-1",
        dist.Categorical(torch.ones(nstate) / nstate),
        infer={"enumerate": "parallel"},
    )
    x = pyro.sample(
        "x_-1",
        dist.Normal(X0, SIGMA),
    )

    beta = pyro.sample("beta", dist.Normal(*BETA))

    try:
        time_plate = pyro.vectorized_markov(name="time", size=nt, dim=-1)
    except NotImplementedError:
        time_plate = pyro.markov(range(nt))

    for t in time_plate:
        x = pyro.sample(
            f"x_{t}",
            dist.Normal(x + s * beta, SIGMA),
            obs=data[t],
        )
        s = pyro.sample(
            f"s_{t}",
            dist.Categorical(probs_trans[s]),
            infer={"enumerate": "parallel"},
        )


def print_shapes(model, guide, first_available_dim):
    guide_trace = handlers.trace(guide).get_trace()
    model_trace = handlers.trace(
        handlers.replay(handlers.enum(model, first_available_dim), guide_trace)
    ).get_trace()
    print(model_trace.format_shapes())


def main(args, max_plate_nesting=1):

    pyro.set_rng_seed(args.seed)
    pyro.clear_param_store()

    guide = AutoDelta(handlers.block(model, expose=["beta", "transition", "x_-1"]))

    if args.print_shapes:
        print_shapes(model, guide, -1 - max_plate_nesting)

    try:
        Elbo = infer.TraceMarkovEnum_ELBO
        max_plate_nesting += 1
    except NotImplementedError:
        Elbo = infer.TraceEnum_ELBO

    print("Evaluate enumerated ELBO")
    elbo = Elbo(max_plate_nesting=max_plate_nesting)
    elbo.loss(model, guide)


if __name__ == "__main__":
    args = parser.parse_args()
    with pyro_backend(args.backend):
        main(args)
