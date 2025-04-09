from setuptools import setup, find_packages

setup(
    name="FinanceToolkit",
    version="0.0.1",
    description="Toolkit financier",
    author="Matteo Bernard",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "typing",
        "tqdm",
        "scipy",
        "matplotlib"
    ],
    include_package_data=True,
)
