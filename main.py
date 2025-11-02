import turbie_mod as tm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Read Turbie parameters
params = tm.read_turbie_parameters()

# 2. Build system matrices
M, C, K = tm.build_system_matrices(params)

# 3. Define wind speeds and TI categories
U_array = np.arange(4, 26, 1)  # 4 m/s to 25 m/s
TI_array = [0.05, 0.1, 0.15]  # turbulence intensity categories

# 4. User-specified case for time series plot
U_plt = 10
TI_plt = 0.1

# 5. Simulation settings
# Note that in my file all wind files are in the same folder
folder = Path("C:/Users/IngCa/Project2_46W38/inputs/wind_files/")
file_name = f"wind_{U_plt}_ms_TI_{TI_plt:.1f}.txt"
file_path = folder / file_name
if not file_path.exists():
    raise FileNotFoundError(f"File not found: {file_path}")

wind_data = np.loadtxt(file_path, skiprows=1)
t_wind = wind_data[:, 0]
t_span = (t_wind[0], t_wind[-1])

y0 = [0.0, 0.0, 0.0, 0.0]

# 6. Output folder (tutti i file qui)
output_folder = Path("outputs")
output_folder.mkdir(exist_ok=True)

# 7. Loop over all TIs and wind speeds
for TI in TI_array:
    means_x1, stds_x1 = [], []
    means_x2, stds_x2 = [], []

    for U in U_array:
        # Read wind file and build wind_func
        try:
            wind_data = tm.read_wind_file(U, TI)
            t_u = wind_data[:, 0]
            u_t = wind_data[:, 1]
            wind_func = tm.build_wind_func(t_u, u_t, kind="linear")
            U_mean = np.mean(u_t)
        except FileNotFoundError:
            wind_func = None
            U_mean = U
            print(f"Wind file missing for U={U}, TI={TI}, using constant wind.")

        Ct_value = tm.get_Ct_value(U_mean, TI, plot=False)

        sol = tm.simulate_turbie(
            M, C, K, params, U_mean, Ct_value, t_span, y0=y0, wind_func=wind_func
        )

        # Save full time series directly in outputs/
        # Note that in my file all outputs go to the folder "outputs"
        filename = output_folder / f"Turbie_U{U:.1f}_TI{TI:.2f}.txt"
        header = "time[s] x1[m] x2[m] x1_dot[m/s] x2_dot[m/s]"
        np.savetxt(
            filename, np.column_stack([sol.t, sol.y.T]), header=header, comments=""
        )
        print(f"Saved simulation results for U={U}, TI={TI} to {filename}")

        # Compute mean and std
        means_x1.append(np.mean(sol.y[0]))
        stds_x1.append(np.std(sol.y[0]))
        means_x2.append(np.mean(sol.y[1]))
        stds_x2.append(np.std(sol.y[1]))

    # Save summary file in outputs/
    summary_file = output_folder / f"Turbie_summary_TI{TI:.2f}.txt"
    summary_data = np.column_stack([U_array, means_x1, stds_x1, means_x2, stds_x2])
    header = "U[m/s] mean_x1[m] std_x1[m] mean_x2[m] std_x2[m]"

# 8. Plot all curves
plt.figure(figsize=(12, 7))
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
markers = ["o", "s", "^"]
line_styles = ["-", "--", "-."]
line_widths = [2.5, 3.5, 4.5]

for i, TI in enumerate(TI_array):
    summary_file = output_folder / f"Turbie_summary_TI{TI:.2f}.txt"
    data = np.loadtxt(summary_file, skiprows=1)
    U_vals = data[:, 0]
    mean_x1, std_x1 = data[:, 1], data[:, 2]
    mean_x2, std_x2 = data[:, 3], data[:, 4]

    plt.errorbar(
        U_vals,
        mean_x1,
        yerr=std_x1,
        fmt=markers[i],
        color=colors[i],
        linestyle=line_styles[0],
        linewidth=line_widths[i],
        markersize=9,
        alpha=0.9,
        label=f"Blade x1, TI={TI}",
        capsize=4,
        capthick=2,
    )

    plt.errorbar(
        U_vals,
        mean_x2,
        yerr=std_x2,
        fmt=markers[i],
        color=colors[i],
        linestyle=line_styles[1],
        linewidth=line_widths[i],
        markersize=9,
        alpha=0.9,
        label=f"Tower x2, TI={TI}",
        capsize=4,
        capthick=2,
    )

plt.xlabel("Wind speed U [m/s]", fontsize=14)
plt.ylabel("Displacement [m]", fontsize=14)
plt.title(
    "Mean ± std of Turbie displacements vs Wind Speed (all TI categories)", fontsize=16
)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(fontsize=12, loc="upper left", ncol=2)
plt.tight_layout()
plt.savefig("Figure_1.jpeg", dpi=200)
plt.show()

# 9. Plot wind speed + displacements in subplots
t_u = wind_data[:, 0]
u_t = wind_data[:, 1]
wind_func_plt = tm.build_wind_func(t_u, u_t)
U_mean_plt = np.mean(u_t)
Ct_plt = tm.get_Ct_value(U_mean_plt, TI_plt, plot=False)
sol_plt = tm.simulate_turbie(
    M, C, K, params, U_mean_plt, Ct_plt, t_span, y0=y0, wind_func=wind_func_plt
)

fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
axs[0].plot(t_wind, u_t, label="Wind speed U(t) [m/s]", color="k", linewidth=2)
axs[0].set_ylabel("U [m/s]", fontsize=12)
axs[0].set_title(f"Wind and Turbie Response at U={U_plt} m/s, TI={TI_plt}", fontsize=14)
axs[0].grid(True, linestyle="--", alpha=0.6)
axs[0].legend(fontsize=10)

axs[1].plot(
    sol_plt.t, sol_plt.y[0], label="Blade x1(t) [m]", color="tab:blue", linewidth=2
)
axs[1].plot(
    sol_plt.t, sol_plt.y[1], label="Tower x2(t) [m]", color="tab:orange", linewidth=2
)
axs[1].set_xlabel("Time [s]", fontsize=12)
axs[1].set_ylabel("Displacement [m]", fontsize=12)
axs[1].grid(True, linestyle="--", alpha=0.6)
axs[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig("Figure_2.jpeg", dpi=200)
plt.show()
