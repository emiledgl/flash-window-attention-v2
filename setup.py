from setuptools import setup, find_packages

setup(
    name='flash_win_attn_v2',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'triton',
    ],
    entry_points={
        'console_scripts': [
            # Add command line scripts here
        ],
    },
    author='',
    author_email='',
    description='',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='',
    classifiers=[
        # to be added
    ],
    python_requires='>=3.6',
)