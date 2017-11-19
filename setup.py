from setuptools import setup, find_packages

setup(
    name='trading_env',
    version='0.0.1dev0',
    author='Yvictor',
    author_email='410175015@gms.ndhu.edu.tw',
    packages=find_packages(),
    install_requires=["pandas",
                      "numpy",
                      "matplotlib",
                      "colour"],
)