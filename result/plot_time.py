import os
import argparse
import pandas as pd
from matplotlib import pyplot as plt


def remove_outlier(x):
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    m = (x <= q3 + (q3 - q1) * 1.5).all(axis=1)
    return x[m]


def read_time_log(log_file):
    df = pd.read_csv(log_file, engine="python", skipinitialspace=True)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c != ""]]
    return df


def time_plot(log_file, display=False):
    # read time log
    df = read_time_log(log_file)
    print("Read " + str(df.shape[0]) + " rows from " + log_file)
    # compute average
    for c in df:
        x = pd.to_numeric(df[c], errors="coerce")
        x = x[x.notna()]  # remove nan
        fmt = "%-35s: num=%d, ave=%f, std=%f, max=%f, min=%f"
        print(fmt % (c, len(x), x.mean(), x.std(), x.max(), x.min()))
    # plot
    preferred_cols = ["Laser Mapping Single Run", "IEKF Solve and Update"]
    available_cols = [c for c in preferred_cols if c in df.columns]
    if available_cols:
        plot_df = df[available_cols].apply(pd.to_numeric, errors="coerce")
    else:
        plot_df = df.apply(pd.to_numeric, errors="coerce")
        plot_df = plot_df.dropna(axis=1, how="all")
        if plot_df.shape[1] == 0:
            print("No numeric columns available for plotting.")
            return
    plot_df = plot_df[plot_df.notna().all(axis=1)]  # remove nan rows
    plot_df = remove_outlier(plot_df)
    y = plot_df.rolling(7, center=True, min_periods=1).mean()  # sliding average, window size = 7
    fig = plt.figure(num=log_file)
    _ = plt.plot(y)
    plt.legend(y.columns)
    plt.ylabel("time/ms")
    plt.xlabel("laser scan")
    out_file = os.path.splitext(log_file)[0] + ".png"
    plt.savefig(out_file)
    if display:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot timing log and save image.")
    parser.add_argument("log_file", help="Path to time log file (e.g., time_log.txt)")
    parser.add_argument("--display", action="store_true", help="Show the plot window")
    args = parser.parse_args()
    time_plot(args.log_file, display=args.display)
