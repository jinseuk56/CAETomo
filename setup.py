from setuptools import setup, find_packages

setup(
    name='CAETomo',
    version='1.0.0',
    description='1D-CAE embedded EELS tomography',
    author='Jinseok Ryu',
    author_email='jinseuk56@gmail.com',
    url='https://github.com/jinseuk56',
    packages=find_packages(include=['CAETomo']),
    install_requires=[
        'drca',
        'streamlit', # Required for the new GUI
        # Make sure to include torch, tifffile, opencv-python, torchkbnufft, etc. here if needed!
    ],
    entry_points={
        'console_scripts': [
            'caetomo-gui=CAETomo.cli:run_gui',  # Creates the terminal launcher
        ],
    },
    setup_requires=['pytest-runner', 'flake8'],
    tests_require=['pytest']
)
