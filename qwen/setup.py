from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name='qwen-thermodynamic',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.24.0',
        'tqdm>=4.65.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'matplotlib>=3.7.0',
            'pandas>=2.0.0',
        ]
    },
    python_requires='>=3.8',
    author='Research Team',
    description='Qwen-Thermodynamic: Entropy-Regularized Transformers for Coherent and Creative Text Generation',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/your-org/qwen-thermodynamic',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
