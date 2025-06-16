from setuptools import setup, find_packages

setup(
    name='flash_win_attn_v2',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch>=2.6.0',
        'triton>=3.2.0',
    ],
    author='emiledgl',
    author_email='',
    description='Triton implementation of Flash Window Attention for Swin Transformer V2',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/emiledgl/flash-window-attention-v2',
    python_requires='>=3.10',
)