from setuptools import setup

package_name = 'data_logger'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tiffa',
    maintainer_email='tiffani.k.ng@gmail.com',
    description='Shared data logging utility',
    license='MIT',
)
