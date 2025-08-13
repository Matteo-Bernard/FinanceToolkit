from setuptools import setup, find_packages

setup(
    name="FinanceToolkit",
    version="0.1.8",
    description="Toolkit for financial data analysis and modeling",
    author="Matteo Bernard",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "typing",
        "tqdm",
        "scipy",
    ],
    include_package_data=True,
)
