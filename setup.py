from setuptools import setup, find_packages

setup(
    name='CAETomo',
    version='0.1.0',
    description='1D-CAE embedded EELS tomography',
    author='Jinseok Ryu',
    author_email='jinseuk56@gmail.com',
    url='https://github.com/jinseuk56',
    packages=find_packages(include=['CAETomo']),
    setup_requires=['pytest-runner', 'flake8'],
    tests_require=['pytest']
)