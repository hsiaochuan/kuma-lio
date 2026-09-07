import os
import subprocess
data_dirs = ["/media/hsiaochuan/hsiaochuan-data/MARS-LVIG/HKisland_GNSS03"]
GNSS_FGO = "../build/global_mapping_offline"
for data_dir in data_dirs:
    obs_fname = os.path.join(data_dir, "HKisland_GNSS03.o")
    rnx_fname = os.path.join(data_dir, "BRDC00IGS_R_20232970000_01D_MN.rnx")
    lio_fname = os.path.join(data_dir, "gnss_log.txt")
    subprocess.run([GNSS_FGO,
                    "--obs_fname", obs_fname,
                    "--nav_fname", rnx_fname,
                    "--lio_fname", lio_fname,
                    "--out_dir", data_dir,
                    ],
                   check=True)