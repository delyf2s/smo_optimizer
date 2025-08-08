from setuptools import setup, find_packages

setup(
    name="smo_optimizer",
    version="0.1.0",
    description="Selective Momentum Optimizer for PyTorch",
    author="Delyfss",
    url="https://github.com/delyf2s/smo_optimizer",
    packages=find_packages(),
    install_requires=["torch"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
