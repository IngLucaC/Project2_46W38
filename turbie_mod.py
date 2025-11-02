import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# 1. Read Turbie parameters
def read_turbie_parameters():
    """
    Reads the parameter file in its fixed order:
    m_b, m_n, m_h, m_t, c1, c2, k1, k2, fb, ft, drb, drt, Dr, rho
    """
    folder = Path("C:/Users/IngCa/Project2_46W38/inputs/turbie_inputs/")
    file_name = "turbie_parameters.txt"
    file_path = folder / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    values = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip() and not line.strip().startswith("#"):
                try:
                    val = float(line.split()[0])
                    values.append(val)
                except ValueError:
                    continue

    if len(values) < 14:
        raise ValueError(f"Expected at least 14 parameters, found {len(values)}")

    keys = [
        "m_b",
        "m_n",
        "m_h",
        "m_t",
        "c1",
        "c2",
        "k1",
        "k2",
        "fb",
        "ft",
        "drb",
        "drt",
        "Dr",
        "rho",
    ]
    return dict(zip(keys, values))


# 2. Build system matrices
def build_system_matrices(params=None):
    if params is None:
        params = read_turbie_parameters()

    m_b = params["m_b"]
    m_n = params["m_n"]
    m_t = params["m_t"]
    m_h = params["m_h"]
    c1 = params["c1"]
    c2 = params["c2"]
    k1 = params["k1"]
    k2 = params["k2"]

    m1 = 3 * m_b
    m2 = m_n + m_t + m_h

    M = np.array([[m1, 0], [0, m2]])
    C = np.array([[c1, -c1], [-c1, c1 + c2]])
    K = np.array([[k1, -k1], [-k1, k1 + k2]])

    return M, C, K


# 3. Thrust coefficient Ct(U)
def Ct_lookup(kind="linear", plot=False):
    folder = Path("C:/Users/IngCa/Project2_46W38/inputs/turbie_inputs/")
    file_name = "CT.txt"
    file_path = folder / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    Ct_tab = np.loadtxt(file_path, comments="#", skiprows=1)
    U, Ct = Ct_tab[:, 0], Ct_tab[:, 1]
    get_Ct = interp1d(U, Ct, kind=kind, bounds_error=False, fill_value=(Ct[0], Ct[-1]))

    if plot:
        U_fine = np.linspace(U.min(), U.max(), 200)
        Ct_fine = get_Ct(U_fine)
        plt.figure(figsize=(7, 4))
        plt.plot(U, Ct, "o", label="Data")
        plt.plot(U_fine, Ct_fine, "-", label=f"Interpolation ({kind})", linewidth=2)
        plt.xlabel("Wind Speed U [m/s]")
        plt.ylabel("Thrust Coefficient $C_T$")
        plt.title("Thrust Coefficient vs Wind Speed")
        plt.grid(True, alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return get_Ct


def get_Ct_value(U, TI=None, kind="linear", plot=False):
    get_Ct = Ct_lookup(kind=kind, plot=plot)
    return float(get_Ct(U))


# 4. Read wind file
# Note that in my file all wind files are in the same folder
def read_wind_file(speed, TI):
    folder = Path("C:/Users/IngCa/Project2_46W38/inputs/wind_files/")
    file_name = f"wind_{speed}_ms_TI_{TI:.1f}.txt"
    file_path = folder / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    wind_data = np.loadtxt(file_path, skiprows=1)
    return wind_data


# 5. Wind interpolation
def build_wind_func(t_u, u_t, kind="linear"):
    """Returns an interpolating function u(t) from time and wind data vectors"""
    return interp1d(t_u, u_t, kind=kind, fill_value="extrapolate")


# 6. Rotor area
def rotor_area(params):
    return np.pi * (params["Dr"] / 2) ** 2


# 7. Aerodynamic forcing
def aero_forcing(t, y, params, U_mean, Ct_value, wind_func=None):
    x1_dot = y[2]
    u = wind_func(t) if wind_func is not None else U_mean
    # debugging output
    # if t < 1.0:
    #    print(f"[DEBUG] t={t:.2f}s, u(t)={u:.3f}, mean={U_mean:.3f}")
    A = rotor_area(params)
    rho = params["rho"]
    f_aero = 0.5 * rho * Ct_value * A * (u - x1_dot) * abs(u - x1_dot)
    return np.array([f_aero, 0.0])


# 8. State derivative
def y_dot(t, y, M, C, K, params, U_mean, Ct_value, wind_func=None):
    F = aero_forcing(t, y, params, U_mean, Ct_value, wind_func)
    x = y[0:2]
    x_dot = y[2:4]
    x_ddot = np.linalg.solve(M, F - C @ x_dot - K @ x)
    return np.concatenate([x_dot, x_ddot])


# 9. Simulation
def simulate_turbie(M, C, K, params, U_mean, Ct_value, t_span, y0=None, wind_func=None):
    if y0 is None:
        y0 = np.zeros(4)

    sol = solve_ivp(
        fun=lambda t, y: y_dot(t, y, M, C, K, params, U_mean, Ct_value, wind_func),
        t_span=t_span,
        y0=y0,
        method="RK45",
        max_step=0.1,
        dense_output=True,
    )
    return sol
