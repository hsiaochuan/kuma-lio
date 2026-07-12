import csv


def euroc_csv_to_tum(input_csv, output_tum):
    """
    Convert EuRoC ground-truth CSV to TUM trajectory format.

    Input CSV (state_groundtruth_estimate0/data.csv):
        #timestamp [ns], p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m],
        q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z [], ...

    Output TUM:
        timestamp[s] tx ty tz qx qy qz qw
    """

    with open(input_csv, "r", newline="") as fin, \
            open(output_tum, "w") as fout:

        reader = csv.reader(fin)
        next(reader)  # skip header

        for row in reader:
            timestamp_ns, x, y, z, qw, qx, qy, qz = row[:8]
            timestamp_s = float(timestamp_ns) * 1e-9

            fout.write(
                f"{timestamp_s:.9f} "
                f"{x} {y} {z} "
                f"{qx} {qy} {qz} {qw}\n"
            )


if __name__ == "__main__":
    input_csv = "/mnt/data/home/hsiaochuan/data/euroc/vicon_room2/V2_01_easy/V2_01_easy/mav0/state_groundtruth_estimate0/data.csv"
    output_tum = "/mnt/data/home/hsiaochuan/data/euroc/vicon_room2/V2_01_easy/V2_01_easy/mav0/state_groundtruth_estimate0/data.txt"

    euroc_csv_to_tum(input_csv, output_tum)

    print(f"Saved to {output_tum}")