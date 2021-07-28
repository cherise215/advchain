from distutils.core import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="advchain",
    version="0.19.2",
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
    install_requires=[            # I get to this in a second
        'torch>=1.6',
        'numpy',
        'SimpleITK',
        'scikit-image',
    ],
    package_dir={"": "src"},
    packages=find_packages(where="advchain"),
    python_requires=">=3.6",
)
