from setuptools import setup

setup(
    name='my_torch_wrapper',
    version='0.1',
    description='Auxiliary framework to handle common Deep Learning tasks with PyTorch.',
    author='Alejandro Fernández',
    author_email='alejandro.fernandez07@estudiant.upf.edu',
    packages=['my_torch_wrapper'],
    python_requires='>=3.9',
    install_requires=[
        'numpy>=1.24.3',
        'torch>=2.0.1',
        'torchinfo>=1.8.0',
    ],
)