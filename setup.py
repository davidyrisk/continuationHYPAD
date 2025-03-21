from setuptools import setup, find_packages

setup(
    name="continuationHYPAD",
    version="1.0.0",
    author="David Y. Risk-Mora",
    author_email="david.risk@my.utsa.edu",
    description="Package for sensitivity analysis of continuation-based postbuckling using HYPAD-FEM.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/...",  # UPDATE
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pyoti", # IDK
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
        # "License :: ...",
    ],
    python_requires=">=3.7",  # Python 3.9.21
)