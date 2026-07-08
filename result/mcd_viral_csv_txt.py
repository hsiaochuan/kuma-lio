import csv


def csv_to_tum(input_csv, output_tum):
    """
    Convert trajectory CSV to TUM trajectory format.

    Input CSV:
        num,t,x,y,z,qx,qy,qz,qw

    Output TUM:
        timestamp tx ty tz qx qy qz qw
    """

    with open(input_csv, "r", newline="") as fin, \
            open(output_tum, "w") as fout:

        reader = csv.DictReader(fin)

        for row in reader:
            fout.write(
                f"{row['t']} "
                f"{row['x']} {row['y']} {row['z']} "
                f"{row['qx']} {row['qy']} {row['qz']} {row['qw']}\n"
            )


if __name__ == "__main__":
    input_csv = "/mnt/data/home/hsiaochuan/data/MCD_VIRAL/bag/ground_truth/tuhh_day_02/pose_inW.csv"
    output_tum = "/mnt/data/home/hsiaochuan/data/MCD_VIRAL/bag/ground_truth/tuhh_day_02/pose_inW.txt"

    csv_to_tum(input_csv, output_tum)

    print(f"Saved to {output_tum}")