r"""°°°
This is a modification of [an example](https://github.com/lanl/scico/blob/main/examples/scripts/deconv_tv_admm.py) from the SCICO package
°°°"""
# |%%--%%| <Fu25XYBiSn|njINuqK7VH>

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
import scico.numpy as snp
import scico.random
from jax import random
from matplotlib.colors import TwoSlopeNorm
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from scico import functional, linop, loss, metric
from scico.optimize.admm import ADMM, CircularConvolveSolver
from xdesign import SiemensStar, discrete_phantom

# |%%--%%| <njINuqK7VH|YGL2O834CF>
r"""°°°
# Create a ground truth image.
°°°"""
# |%%--%%| <YGL2O834CF|orkFzBGsKU>

phantom = SiemensStar(32)
N = 256  # image size
x_gt = snp.pad(discrete_phantom(phantom, N - 16), 8)
x_gt = jax.device_put(x_gt)  # convert to jax type, push to GPU

# |%%--%%| <orkFzBGsKU|nGiOxCLlGr>
r"""°°°
# Simulate the signal

Set up the forward operator and create a test signal consisting of a
blurred signal with additive Gaussian noise.
°°°"""
# |%%--%%| <nGiOxCLlGr|M8lsVYW2Pf>

n = 5  # convolution kernel size
σ = 20.0 / 255  # noise level

psf = snp.ones((n, n)) / (n * n)
A = linop.CircularConvolve(h=psf, input_shape=x_gt.shape)

Ax = A(x_gt)  # blurred image
noise, key = scico.random.randn(Ax.shape, seed=0)
y = Ax + σ * noise

# |%%--%%| <M8lsVYW2Pf|cBjVr9A8Xx>
r"""°°°
# Solve using ADMM
°°°"""
# |%%--%%| <cBjVr9A8Xx|kzrfY8f05c>

λ = 2e-2  # L21 norm regularization parameter
ρ = 5e-1  # ADMM penalty parameter
maxiter = 50  # number of ADMM iterations

f = loss.SquaredL2Loss(y=y, A=A)
# Penalty parameters must be accounted for in the gi functions, not as
# additional inputs.
g = λ * functional.L21Norm()  # regularization functionals gi
C = linop.FiniteDifference(x_gt.shape, circular=True)
solver = ADMM(
    f=f,
    g_list=[g],
    C_list=[C],
    rho_list=[ρ],
    x0=A.adj(y),
    maxiter=maxiter,
    subproblem_solver=CircularConvolveSolver(),
    itstat_options={"display": True, "period": 10},
)
solution = {"admm": solver.solve()}

# |%%--%%| <kzrfY8f05c|ve5qNROAw1>
r"""°°°
# Now try the simplest model with numpyro

Neither attempt to match `sigma_dx` to `λ` used above, nor to learn it with a hyperprior.
°°°"""
# |%%--%%| <ve5qNROAw1|b3GFFaBPVX>


def deconvolution_tv(y, sigma_x=0.5, sigma_dx=0.1, sigma_y=0.1):
    """Probabilistic model of deconvolution with TV regularisation

    Parameters
    ----------

    y: jnp.ndarray, float
        Observed blurred and noisy 2D image with values between -1 and 1
    h_fft: jnp.ndarray, complex
        Convolution kernel in the fourier domain, DC component at [0,0]
    sigma_x: float
        Scale parameter for the Normal prior on the desired x
    sigma_dx: float
        Scale parameter for the Laplace prior on the gradient of the desired x
    sigma_y: float
        Standard deviation of the observation noise in y
    """

    with numpyro.plate_stack("ij", y.shape):
        x = numpyro.sample("x", dist.Normal(0.5, sigma_x))

    grad_x = jnp.stack(jnp.gradient(x), axis=0)
    with numpyro.plate_stack("ij", y.shape):
        with numpyro.plate("grad_component", y.ndim):
            numpyro.sample("laplace_prior_x", dist.Laplace(0.0, sigma_dx), obs=grad_x)

    y_pred = A(x)

    with numpyro.plate_stack("ij", y.shape):
        numpyro.sample("likelihood", dist.Normal(y_pred, sigma_y), obs=y)


# |%%--%%| <b3GFFaBPVX|1QAi68fYoU>

guide = AutoDelta(deconvolution_tv)
optimizer = numpyro.optim.Adam(step_size=1e-2)
svi = SVI(deconvolution_tv, guide, optimizer, loss=Trace_ELBO())
res = svi.run(random.PRNGKey(0), num_steps=1_000, y=y, sigma_dx=0.75)
solution["svi"] = res.params["x_auto_loc"]
for k, v in res.params.items():
    if k != "x_auto_loc":
        print(k, v)

# |%%--%%| <1QAi68fYoU|4ReG7Xue1M>
r"""°°°
# Visualise the results
°°°"""
# |%%--%%| <4ReG7Xue1M|djVdeRroP7>

vmin, vmax = x_gt.min(), x_gt.max()

fig, axes = plt.subplots(
    nrows=2, ncols=4, figsize=(15, 10), gridspec_kw={"width_ratios": [10] * 3 + [1]}
)

im = axes[0, 0].imshow(x_gt, vmin=vmin, vmax=vmax, interpolation="none")
axes[0, 0].set_title("ground truth")

im = axes[1, 0].imshow(y, vmin=vmin, vmax=vmax, interpolation="none")
axes[1, 0].set_title("blurred, noisy image: %.2f (dB)" % metric.psnr(x_gt, y))

for ax_col, (method, arr) in zip(axes[:, 1:].T, solution.items()):
    ax_col[0].imshow(arr, vmin=vmin, vmax=vmax, interpolation="none")
    ax_col[0].set_title(f"{method}: {metric.psnr(x_gt, arr):.2f} (dB)")

    im_r = ax_col[1].imshow(
        arr - x_gt, interpolation="none", norm=TwoSlopeNorm(0, -0.5, 0.5), cmap="RdBu_r"
    )
    ax_col[1].set_title(f"{method} - gt")


for ax in axes[:, :-1].flat:
    ax.axis("off")

plt.colorbar(im, cax=axes[0, -1])
plt.colorbar(im_r, cax=axes[1, -1])
plt.show()
