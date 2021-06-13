import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="biot_savart",  # Replace with your own username
    version="5.0",
    author="Mingde Yin",
    author_email="mdsuper@hotmail.com",
    description="Magnetic Field Computation Library.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vuthalab/biot_savart",
    license="BSD-3-Clause",
    install_requires=[
        "numpy",
        "matplotlib"
    ],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
