from distutils.core import setup
setup(
    name='advchain',
    packages=['advchain'],
    version='0.1',
    # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    license='MIT',
    description='a plug-in module for adversarial data augmentor with chained transformations',
    author='Chen (Cherise) Chen',
    author_email='work.cherise@gmail.com',
    url='https://github.com/cherise215/',
    download_url='https://github.com/cherise215/advchain/archive/refs/tags/v_01.zip',
    keywords=['data augmentation', 'pytorch', 'segmentation'],
    install_requires=[
        'matplotlib',
        'seaborn',
        'numpy',
        'SimpleITK',
        'skimage',
        'torch'
    ],
    classifiers=[
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',   # Again, pick a license
        'Programming Language :: Python :: 3.6.9',
    ],
)
