
from setuptools import setup, find_packages


install_requires = [
    "torch==1.5.0",
]


setup(
    name="gqnlib",
    version="0.1",
    description="Generative Query Network by PyTorch",
    packages=find_packages(),
    install_requires=install_requires,
)
