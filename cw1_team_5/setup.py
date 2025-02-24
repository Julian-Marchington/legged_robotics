from setuptools import setup
import os
from glob import glob

package_name = 'cw1_team_5'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), 
         glob(os.path.join('launch', '*.launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='team_5',
    maintainer_email='dhyan.shyam.24@ucl.ac.uk',
    description='Coursework 1 solution for team 5',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'obstacle_follower_node = cw1_team_5.obstacle_follower_node:main',
            'bug0_node = cw1_team_5.bug0_node:main',
            'bug1_node = cw1_team_5.bug1_node:main',
        ],
    },
) 
