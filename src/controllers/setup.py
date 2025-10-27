from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'controllers'

setup(
    name=package_name,
    version='1.0.0',
    # Automatically discover packages including 'controllers' and 'controllers.lib'
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'resource'), [
            'resource/controllers',
        ]),
        # (os.path.join('share', package_name, 'resource', 'map_conditioned_cuniform_models_v2.5'), [
        #     'resource/map_conditioned_cuniform_models_v2.5/best_feature_extractor.pth',
        #     'resource/map_conditioned_cuniform_models_v2.5/best_model.pth',
        # ]),
        # (os.path.join('share', package_name, 'resource', 'unsupervised_cuniform_model_v2.5'), [
        #     'resource/unsupervised_cuniform_model_v2.5/MapAct_best_model_single_env.pt',
        # ]),
    ],
    install_requires=['setuptools', 'rclpy', 'numpy', 'torch', 'scipy', 'pyyaml', 'matplotlib', 'numba'],
    zip_safe=True,
    maintainer='yukang',
    maintainer_email='mikasa.cyk@gmail.com',
    description='PyTorch-based trajectory planners (MPPI, CU-MPPI, UGE-MPC)',
    license='MIT',
    entry_points={
        'console_scripts': [
            'local_planner_node = controllers.local_planner_node:main',
            
            # perception preprocessing for controller input
            'costmap_processor_node = controllers.costmap_processor_node:main',
            'dummy_local_costmap_publisher = controllers.dummy_local_costmap_publisher:main',
            'dummy_odom_publisher = controllers.dummy_odom_publisher:main',
            'dummy_goal_publisher = controllers.dummy_goal_publisher:main',
        ],
    },
)
