from setuptools import setup

setup(
    name='Military_Vehicles_Tracking',
    version='1.0.0',
    packages=['Military_Vehicles_Tracking', 'Military_Vehicles_Tracking.deep_sort_pytorch',
              'Military_Vehicles_Tracking.deep_sort_pytorch.utils',
              'Military_Vehicles_Tracking.deep_sort_pytorch.deep_sort',
              'Military_Vehicles_Tracking.deep_sort_pytorch.deep_sort.deep',
              'Military_Vehicles_Tracking.deep_sort_pytorch.deep_sort.sort'],
    url='https://github.com/Lin-Sinorodin/Military_Vehicles_Tracking',
    license='MIT',
    author='lin',
    author_email='linsino123@gmail.com',
    description='MOT for military vehicles'
)
