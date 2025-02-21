from setuptools import setup, find_packages

setup(
    name='mlvtrans',
    version='__version__',
    author='yugotakada',
    author_email='yugo.takada1@gmail.com',
    description='mlvtrans: construct a compatible symplectic basis and transversal phase-type gates for self-dual CSS codes',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yugotakada/mlvtrans.git',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'stim',
    ],
    entry_points={
        'console_scripts': [
            'mlvtrans=mlvtrans.main:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
     license=LICENSE,
)
