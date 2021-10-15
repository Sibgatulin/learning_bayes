from setuptools import setup, find_packages


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
        "jax",
        "torch",
        "pyro-ppl",
        "numpyro",
        "funsor",
        "jupyter",
        "matplotlib",
    ],
)
