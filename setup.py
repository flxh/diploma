from setuptools import setup

setup(
    name='diploma',
    version='',
    packages=['ml', 'simulation', 'timeseriesprediction'],
    package_dir={'': 'src'},
    url='',
    license='',
    author='Florus Härtel',
    author_email='',
    description='',
    install_requires=['tensorflow==1.15', 'numpy', 'scipy', 'scikit-learn', 'matplotlib', 'pandas', 'tensorflow-probability==0.7']
)
