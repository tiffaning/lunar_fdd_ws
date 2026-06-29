from setuptools import find_packages, setup

package_name = 'construction_robot'
from setuptools import setup
import os
from glob import glob

package_name = 'construction_robot'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install launch files
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        # Install urdf files
        (os.path.join('share', package_name, 'urdf'),
            glob('urdf/*')),
        # Install config files
        (os.path.join('share', package_name, 'config'),
            glob('config/*')),
	#scripts directory
        (os.path.join('share', package_name, 'scripts'),
            glob('scripts/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tiffa',
    maintainer_email='tiffani.ng@ufl.edu',
    description='Lunar construction robot model',
    license='MIT',
    entry_points={
        'console_scripts': [
            'safe_joint_test = construction_robot.safe_joint_test:main',
        ],
    },
)
