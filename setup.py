import glob
import os

from setuptools import find_packages, setup

package_name = 'demo2'


def package_files(relative_dir):
    return [
        path
        for path in glob.glob(os.path.join(relative_dir, '*'))
        if os.path.isfile(path)
    ]


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(where='src', exclude=['test']),
    package_dir={'': 'src'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), package_files('launch')),
        (os.path.join('share', package_name, 'config'), package_files('config')),
        (os.path.join('share', package_name, 'urdf'), package_files('urdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yang',
    maintainer_email='yang@todo.todo',
    description='YOLO pose to mirrored dual Nero arm control.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'arm_yolo26s_v2 = demo2.arm_yolo26s_v2:main',
            'position_to_angle_v2 = demo2.position_to_angle_v2:main',
        ],
    },
)
