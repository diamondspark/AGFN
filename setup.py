from setuptools import setup, find_packages

with open("requirements_jn.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="atomicGFN",
    version="0.1.0",
    author="Mohit Pandey, Emmanuel Bengio, Gopeshh Subbaraj",
    description="Pretraining and Finetuning GFN",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=requirements,
    include_package_data=True,
)