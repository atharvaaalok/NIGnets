from setuptools import find_packages, setup


# Read contents of readme.md to use as long description later
with open('README.md', 'r', encoding = 'utf-8') as f:
    readme = f.read()


setup(
    name = 'NIGnets',
    author = 'Atharva Aalok',
    author_email = 'atharvaaalok@gmail.com',
    version = '0.0.0',
    url = 'https://github.com/atharvaaalok/NIGnets',
    license = 'MIT',
    description = 'Neural Injective Geometry networks (NIGnets) for non-self-intersecting geometry.',
    long_description = readme,
    long_description_content_type = 'text/markdown',
    packages = find_packages(),
    install_requires = [
        'matplotlib==3.10.0',
        'numpy==2.2.3',
        'setuptools==57.4.0',
        'svg.path==6.3',
        'torch==2.5.1'
    ],
    keywords = ['NIGnets', 'geometry', 'neural networks', 'machine learning', 'deep learning',
                'optimization', 'shape optimization', 'geometric deep learning', 'shape matching',
                'neural injective geometry', 'geosimilarity', 'pytorch', 'autograd', 'curves',
                'surfaces', 'curve fitting', 'surface fitting']
)