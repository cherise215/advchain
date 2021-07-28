from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="advchain",
    version="0.19.7",
    author="Chen (Cherise) Chen",
    author_email="work.cherise@gmail.com",
    description="adversarial data augmentation with chained transformations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cherise215/advchain",
    project_urls={
        "Bug Tracker": "https://github.com/cherise215/advchain/issues",
    },
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate you support Python 3. These classifiers are *not*
        # checked by 'pip install'. See instead 'python_requires' below.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
    ],
    install_requires=[            # I get to this in a second
        'torch>=1.6',
        'numpy',
        'SimpleITK',
        'scikit-image',
    ],
    extras_require={
        'plot': ['matplotlib'],
    },
    python_requires=">=3.6",
    keywords='advchain',
    packages=find_packages(include=['advchain', 'advchain.*']),
)
