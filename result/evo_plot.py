import matplotlib.pyplot as plt
from evo.core import sync, metrics
from evo.tools import file_interface, plot

# /tmp/all_odom_260711020024.tum
traj_ref = file_interface.read_tum_trajectory_file("/tmp/all_odom_260711020024.tum")
traj_est = file_interface.read_tum_trajectory_file("/mnt/data/home/hsiaochuan/data/euroc/vicon_room2/V2_01_easy/V2_01_easy/mav0/state_groundtruth_estimate0/data.txt")

traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff=0.01)

r_a, t_a, s = traj_est.align(traj_ref, correct_scale=False)

ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
ape_metric.process_data((traj_ref, traj_est))
ape_stats = ape_metric.get_all_statistics()

print("APE (translation part) statistics:")
for stat_name, value in ape_stats.items():
    print(f"  {stat_name}: {value:.4f}")

fig = plt.figure(figsize=(6, 6))
ax = plot.prepare_axis(fig, plot.PlotMode.xy)
plot.traj(ax, plot.PlotMode.xy, traj_ref, style="--", color="gray",
          label="reference", alpha=0.7)
plot.traj(ax, plot.PlotMode.xy, traj_est, style="-", color="blue",
          label="estimate (aligned)", alpha=0.9)

fig.patch.set_facecolor("#ffffff")
ax.set_facecolor("#ffffff")
ax.grid(True, color="#b0b0b0")
for spine in ax.spines.values():
    spine.set_color("black")
ax.set_title(f"Aligned Trajectory (APE RMSE: {ape_stats['rmse']:.4f} m)")
leg = ax.get_legend()
leg.get_frame().set_facecolor("white")
leg.get_frame().set_alpha(0.5)
fig.savefig("aligned_plot.png")