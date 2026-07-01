from setuptools import setup
import os
from glob import glob

package_name = 'performance_monitor'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'),
            glob('config/*')),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tiffa',
    maintainer_email='tiffani.ng@ufl.edu',
    description='Performance monitoring for lunar FDD system',
    license='MIT',
    entry_points={
        'console_scripts': [
            'monitor_node = performance_monitor.monitor_node:main',
        ],
    },
)
