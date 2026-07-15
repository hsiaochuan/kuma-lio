raw_dir="/mnt/data/home/hsiaochuan/data/MCD_VIRAL/raw/"
merge_dir="/mnt/data/home/hsiaochuan/data/MCD_VIRAL/bag/"
python3 rosbag_merge.py -i ${raw_dir}/ntu_day_01_mid70.bag ${raw_dir}/ntu_day_01_vn100.bag -o ${merge_dir}/ntu_day_01.bag
python3 rosbag_merge.py -i ${raw_dir}/ntu_day_02_mid70.bag ${raw_dir}/ntu_day_02_vn100.bag -o ${merge_dir}/ntu_day_02.bag
python3 rosbag_merge.py -i ${raw_dir}/ntu_day_10_mid70.bag ${raw_dir}/ntu_day_10_vn100.bag -o ${merge_dir}/ntu_day_10.bag
python3 rosbag_merge.py -i ${raw_dir}/ntu_night_04_mid70.bag ${raw_dir}/ntu_night_04_vn100.bag -o ${merge_dir}/ntu_night_04.bag
python3 rosbag_merge.py -i ${raw_dir}/ntu_night_08_mid70.bag ${raw_dir}/ntu_night_08_vn100.bag -o ${merge_dir}/ntu_night_08.bag
python3 rosbag_merge.py -i ${raw_dir}/ntu_night_13_mid70.bag ${raw_dir}/ntu_night_13_vn100.bag -o ${merge_dir}/ntu_night_13.bag

python3 rosbag_merge.py -i ${raw_dir}/kth_day_06_mid70.bag ${raw_dir}/kth_day_06_vn200.bag -o ${merge_dir}/kth_day_06.bag
python3 rosbag_merge.py -i ${raw_dir}/kth_day_09_mid70.bag ${raw_dir}/kth_day_09_vn200.bag -o ${merge_dir}/kth_day_09.bag
python3 rosbag_merge.py -i ${raw_dir}/kth_day_10_mid70.bag ${raw_dir}/kth_day_10_vn200.bag -o ${merge_dir}/kth_day_10.bag
python3 rosbag_merge.py -i ${raw_dir}/kth_night_01_mid70.bag ${raw_dir}/kth_night_01_vn200.bag -o ${merge_dir}/kth_night_01.bag
python3 rosbag_merge.py -i ${raw_dir}/kth_night_04_mid70.bag ${raw_dir}/kth_night_04_vn200.bag -o ${merge_dir}/kth_night_04.bag
python3 rosbag_merge.py -i ${raw_dir}/kth_night_05_mid70.bag ${raw_dir}/kth_night_05_vn200.bag -o ${merge_dir}/kth_night_05.bag

python3 rosbag_merge.py -i ${raw_dir}/tuhh_day_02_mid70.bag ${raw_dir}/tuhh_day_02_vn200.bag -o ${merge_dir}/tuhh_day_02.bag
python3 rosbag_merge.py -i ${raw_dir}/tuhh_day_03_mid70.bag ${raw_dir}/tuhh_day_03_vn200.bag -o ${merge_dir}/tuhh_day_03.bag
python3 rosbag_merge.py -i ${raw_dir}/tuhh_day_04_mid70.bag ${raw_dir}/tuhh_day_04_vn200.bag -o ${merge_dir}/tuhh_day_04.bag
python3 rosbag_merge.py -i ${raw_dir}/tuhh_night_07_mid70.bag ${raw_dir}/tuhh_night_07_vn200.bag -o ${merge_dir}/tuhh_night_07.bag
python3 rosbag_merge.py -i ${raw_dir}/tuhh_night_08_mid70.bag ${raw_dir}/tuhh_night_08_vn200.bag -o ${merge_dir}/tuhh_night_08.bag
python3 rosbag_merge.py -i ${raw_dir}/tuhh_night_09_mid70.bag ${raw_dir}/tuhh_night_09_vn200.bag -o ${merge_dir}/tuhh_night_09.bag
