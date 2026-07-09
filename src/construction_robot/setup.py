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
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'urdf'),
            glob('urdf/*')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*')),
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
            'repeated_motion = construction_robot.repeated_motion:main',
        ],
    },
)
