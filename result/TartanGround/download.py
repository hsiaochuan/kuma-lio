import tartanair as ta

ta.init('/home/hsiaochuan/Downloads/tartanground')

# env = [
#     "Downtown",
#     "ModernCityDowntown",
#     "ModularNeighborhood",
#     "NordicHarbor",
#     "OldTownFall",
#     "OldTownSummer",
# ]
#
# ta.download_ground(
#     env = env,
#     version = ['anymal'],
#     traj = ["P2000"],
#     modality = [
#         'image',
#         'meta',
#         # 'depth',
#         # 'seg',
#         'lidar',
#         # 'imu',
#         # 'rosbag',
#         # 'sem_pcd',
#         # 'seg_labels',
#         # 'rgb_pcd'
#     ],
#     camera_name = [
#         'lcam_front',
#         "rcam_front",
#         # 'lcam_right',
#         # 'lcam_left',
#         # 'lcam_back'
#     ],
#     unzip = True
# )

env = [
    "AbandonedCable",
    # "CarWelding",
    # "GothicIsland",
    # "JapaneseAlley",
    # "Supermarket",
]

ta.download_ground(
    env = env,
    version = ['diff'],
    traj = ["P1000"],
    modality = [
        'image',
        'meta',
        # 'depth',
        # 'seg',
        'lidar',
        'imu',
        # 'rosbag',
        # 'sem_pcd',
        # 'seg_labels',
        'rgb_pcd'
    ],
    camera_name = [
        'lcam_front',
        # "rcam_front",
        # 'lcam_right',
        # 'lcam_left',
        # 'lcam_back'
    ],
    unzip = True
)