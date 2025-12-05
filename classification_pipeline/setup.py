from setuptools import setup, find_packages

setup(
    name='ccp',
    version='0.2',
    packages=['ccp'],
    install_requires=[
        'numpy',
        'matplotlib',
        'scikit-learn',
        'seaborn',
        'pandas',
        'cmcrameri',
        'openTSNE',
        'tqdm',
        'umap-learn',
        'pacmap'
    ],
)
