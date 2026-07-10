import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'hybrid_fdd'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Trained models loaded by the node at runtime
        (os.path.join('share', package_name, 'models'),
            glob('models/*.pkl')),
        # Launch files
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        # Config (if any)
        (os.path.join('share', package_name, 'config'),
            glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tiffa',
    maintainer_email='tiffani.k.ng@gmail.com',
    description='Hybrid FDD system for lunar construction robots',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'hybrid_fdd_node = hybrid_fdd.hybrid_fdd_node:main',
        ],
    },
)
