from setuptools import find_packages, setup


def readme():
    with open("README.md") as f:
        f.read()


setup(
    name="learing_bayes",
    version="0.1",
    description="Collection of snippets on questions in Bayesian inference",
    long_description=readme(),
    url="https://github.com/Sibgatulin/learing_bayes/",
    author="Renat Sibgatulin",
    packages=find_packages(),
    install_requires=[
        "funsor",
        "matplotlib",
    ],
    extras_require={
        "torch": ["torch", "pyro-ppl"],
        "jax": ["jax", "numpyro"],
        "jupyter": ["jupyter"],
    },
)
