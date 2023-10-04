from setuptools import setup, find_packages

setup(
    name='curiosity',
    version='0.0.1',
    url='https://github.com/MarkHaoxiang/curiosity.git',
    author='Hao Xiang Li',
    author_email='mark.haoxiang@gmail.com',
    description='Implementation of reinforcement reinforcement learning algorithms',
    packages=find_packages('curiosity', 'curiosity.*'),    
    install_requires=['torch', 'gymnasium']
)