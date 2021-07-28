import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="advchain",
    version="0.18",
    author="Chen (Cherise) Chen",
    author_email="work.cherise@gmail.com",
    description="adversarial data augmentation with chained  transformations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cherise215/advchain",
    project_urls={
        "Bug Tracker": "https://github.com/cherise215/advchain/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
