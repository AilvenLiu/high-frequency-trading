from setuptools import setup, find_packages

setup(
    name="crypto_trading",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'requests>=2.31.0',
        'pyyaml>=6.0.1',
        'websocket-client>=1.6.4',
        'pandas>=2.1.1',
        'redis>=5.0.1',
        'psutil>=5.9.5',
        'prometheus-client>=0.17.1',
        'torch>=2.1.0',
        'numpy>=1.24.3',
        'scikit-learn>=1.3.0',
        'websockets>=11.0.3',
        'python-dateutil>=2.8.2',
        'aiohttp>=3.8.5',
        'ujson>=5.8.0',
        'aiofiles>=23.1.0',
        'ccxt>=3.0.0'
    ],
    extras_require={
        'dev': [
            'pytest>=7.4.2',
            'pytest-asyncio>=0.21.1',
            'pytest-cov>=4.1.0',
            'pytest-mock>=3.11.1',
            'black>=23.9.1',
            'isort>=5.12.0',
            'mypy>=1.5.1',
            'flake8>=6.1.0'
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)