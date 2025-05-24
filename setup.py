from setuptools import setup, find_packages

_ = setup(
    name="facemodel",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages("src"),
)
