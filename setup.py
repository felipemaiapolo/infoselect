
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="infosel",
    version="1.0.0",
    author="Felipe Maia Polo & Felipe Leno da Silva",
    author_email="felipemaiapolo@gmail.com, f.leno@usp.br",
    description="A package that makes feature/variable selection for supervised learning using Mutual Information.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['scipy','numpy','pandas','sklearn','matplotlib'],
) 
