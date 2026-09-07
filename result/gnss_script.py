import os
import subprocess

DATA_DIR = "/media/hsiaochuan/hsiaochuan-data/MARS-LVIG"
GNSS_FGO = "../build/global_mapping_offline"

# obs/nav for this sequence live in the per-sequence subdirectory, not the MARS-LVIG root
datas = [
    "HKisland_GNSS03",
]
for data in datas:
    GNSS_DIR = os.path.join(DATA_DIR, data)
    obs_fname = os.path.join(GNSS_DIR, "obs.o")
    rnx_fname = os.path.join(GNSS_DIR, "nav.rnx")
    lio_fname = os.path.join(DATA_DIR, data + "_faster_lio_result", "gnss_log.txt")

    subprocess.run([GNSS_FGO,
                    "--obs_fname", obs_fname,
                    "--nav_fname", rnx_fname,
                    "--lio_fname", lio_fname,
                    "--out_dir", GNSS_DIR,
                    ],
                   check=True)
