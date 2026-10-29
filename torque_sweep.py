"""
Torque sweep for the reduced-order model, Eq. (17):
    tau_req = I_eff*w_dot + w x (I_eff*w) + I_eff_dot*w
Idealised conditions: rigid needle + yarn loops as rigid rings on the shaft.
Needle axis = body x-axis.  Run:  python torque_sweep.py
All parameters are in the PARAMS dict -- edit and re-run.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PARAMS = dict(
    rho_al=2700.0,        # kg/m^3
    r_needle=1.5e-3,      # m
    L_needle=0.200,       # m
    tex=100.0,            # g/km  -> lambda = tex*1e-6 kg/m
    L_loop=0.020,         # m yarn per loop
    R_loop=2.5e-3,        # m effective loop ring radius about needle axis
    n_loops=20,           # loops on this needle
    loop_span=0.5,        # fraction of shaft occupied by loops (centred)
    f_knit=2.0,           # Hz, stitch/oscillation frequency
    theta_amp=np.deg2rad(15),   # rad, V-angle swing amplitude
    wax_ratio=0.5,        # axial spin amplitude as fraction of transverse w
    stitch_rate=2.0,      # loops transferred per second
)

def needle_inertia(p):
    m = p["rho_al"] * np.pi * p["r_needle"]**2 * p["L_needle"]
    Ia = 0.5 * m * p["r_needle"]**2
    It = m * (3 * p["r_needle"]**2 + p["L_needle"]**2) / 12
    return m, np.diag([Ia, It, It])          # x = axis

def yarn_inertia(p, n):
    """n loops as thin rings, spread uniformly over loop_span of the shaft."""
    lam = p["tex"] * 1e-6
    m1 = lam * p["L_loop"]
    if n <= 0:
        return 0.0, np.zeros((3, 3)), np.zeros((3, 3))
    span = p["loop_span"] * p["L_needle"]
    z = np.linspace(-span/2, span/2, max(int(round(n)), 1)) if n > 1 else np.array([0.0])
    nn = n / len(z)                           # allow fractional n
    Ia = np.sum(nn * m1 * p["R_loop"]**2)
    It = np.sum(nn * m1 * (0.5 * p["R_loop"]**2 + z**2))
    # per-loop inertia (mean) for dI/dt from loop transfer
    Ia1 = m1 * p["R_loop"]**2
    It1 = m1 * (0.5 * p["R_loop"]**2 + np.mean(z**2))
    return n * m1, np.diag([Ia, It, It]), np.diag([Ia1, It1, It1])

def torque_terms(p, w_amp, f, n_loops=None, stitch_rate=None):
    """Peak magnitudes of each term of Eq.17 over one cycle."""
    n = p["n_loops"] if n_loops is None else n_loops
    sr = p["stitch_rate"] if stitch_rate is None else stitch_rate
    _, In = needle_inertia(p)
    _, Iy, Iy1 = yarn_inertia(p, n)
    I = In + Iy
    Idot = sr * Iy1                           # dn/dt * per-loop inertia
    t = np.linspace(0, 1 / f, 400)
    om = 2 * np.pi * f
    # transverse (y) swing + axial (x) spin, 90 deg out of phase
    w = np.stack([p["wax_ratio"] * w_amp * np.sin(om * t),
                  w_amp * np.cos(om * t),
                  np.zeros_like(t)], axis=1)
    wd = np.stack([p["wax_ratio"] * w_amp * om * np.cos(om * t),
                   -w_amp * om * np.sin(om * t),
                   np.zeros_like(t)], axis=1)
    t_acc = (wd @ I.T)
    t_gyr = np.cross(w, w @ I.T)
    t_dot = (w @ Idot.T)
    mag = lambda a: np.linalg.norm(a, axis=1).max()
    tot = np.linalg.norm(t_acc + t_gyr + t_dot, axis=1).max()
    return mag(t_acc), mag(t_gyr), mag(t_dot), tot

def sweep_amp_freq(p):
    f = np.logspace(-1, 1.7, 60)              # 0.1 .. 50 Hz
    amp = np.deg2rad(np.linspace(2, 45, 50))
    out = np.zeros((len(amp), len(f), 4))
    for i, a in enumerate(amp):
        for j, ff in enumerate(f):
            out[i, j] = torque_terms(p, a * 2 * np.pi * ff, ff)  # w_amp = a*om
    return f, amp, out

def main():
    p = PARAMS
    m, In = needle_inertia(p)
    print(f"Needle mass {m*1e3:.2f} g | I_axial {In[0,0]:.2e} | I_trans {In[1,1]:.2e} kg m^2")

    # 1) amplitude x frequency map of each term and ratios
    f, amp, out = sweep_amp_freq(p)
    names = ["|I w_dot| (accel)", "|w x I w| (gyro)", "|I_dot w| (inertia change)", "total"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for k in range(3):
        c = ax.flat[k].contourf(f, np.rad2deg(amp), np.log10(out[..., k] + 1e-30), 20)
        ax.flat[k].set_xscale("log"); ax.flat[k].set_title(names[k] + "  [log10 N·m]")
        ax.flat[k].set_xlabel("frequency (Hz)"); ax.flat[k].set_ylabel("swing amplitude (deg)")
        plt.colorbar(c, ax=ax.flat[k])
    r1 = out[..., 1] / out[..., 0]; r2 = out[..., 2] / out[..., 0]
    for k, (r, ttl) in enumerate([(r1, "gyro / accel"), (r2, "inertia-change / accel")]):
        c = ax.flat[3 + k].contourf(f, np.rad2deg(amp), np.log10(r + 1e-30), 20)
        ax.flat[3 + k].set_xscale("log"); ax.flat[3 + k].set_title(ttl + "  [log10]")
        ax.flat[3 + k].set_xlabel("frequency (Hz)"); plt.colorbar(c, ax=ax.flat[3 + k])
    ax.flat[5].axis("off")
    ax.flat[5].text(0, 0.5, f"Peak total torque over map:\n{out[...,3].max():.2e} N·m\n"
                    f"Max gyro/accel: {r1.max():.2e}\nMax Idot/accel: {r2.max():.2e}", fontsize=12)
    plt.tight_layout(); plt.savefig("/mnt/user-data/outputs/sweep1_amp_freq.png", dpi=130); plt.close()

    # 2) yarn load: tex x loop count at fixed motion
    tex_v = np.logspace(1, 3, 40); n_v = np.arange(0, 61, 3)
    w_amp = p["theta_amp"] * 2 * np.pi * p["f_knit"]
    tot = np.zeros((len(n_v), len(tex_v))); frac = np.zeros_like(tot)
    for j, tx in enumerate(tex_v):
        q = dict(p, tex=tx)
        for i, n in enumerate(n_v):
            a, g, d, T = torque_terms(q, w_amp, p["f_knit"], n_loops=n)
            a0 = torque_terms(q, w_amp, p["f_knit"], n_loops=0)[0]
            tot[i, j] = T; frac[i, j] = (a - a0) / a0
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    c = ax[0].contourf(tex_v, n_v, np.log10(tot + 1e-30), 20); ax[0].set_xscale("log")
    ax[0].set_title("Total torque, log10 N·m"); plt.colorbar(c, ax=ax[0])
    c = ax[1].contourf(tex_v, n_v, 100 * frac, 20); ax[1].set_xscale("log")
    ax[1].set_title("Torque increase from yarn (%)"); plt.colorbar(c, ax=ax[1])
    for a_ in ax: a_.set_xlabel("yarn tex (g/km)"); a_.set_ylabel("loops on needle")
    plt.tight_layout(); plt.savefig("/mnt/user-data/outputs/sweep2_yarn_load.png", dpi=130); plt.close()

    # 3) loop-transfer rate: effect of I_dot term
    sr = np.logspace(-1, 3, 50)
    fig, ax = plt.subplots(figsize=(7, 5))
    for tx in (30, 100, 300, 1000):
        q = dict(p, tex=tx)
        y = [torque_terms(q, w_amp, p["f_knit"], stitch_rate=s)[2] /
             torque_terms(q, w_amp, p["f_knit"], stitch_rate=s)[0] for s in sr]
        ax.loglog(sr, y, label=f"{tx} tex")
    ax.set_xlabel("loop transfer rate (loops/s)"); ax.set_ylabel("|I_dot w| / |I w_dot|")
    ax.legend(); ax.grid(True, which="both", alpha=.3)
    plt.tight_layout(); plt.savefig("/mnt/user-data/outputs/sweep3_inertia_rate.png", dpi=130); plt.close()

    # 4) baseline numbers at nominal knitting motion
    a, g, d, T = torque_terms(p, w_amp, p["f_knit"])
    print(f"Nominal ({np.rad2deg(p['theta_amp']):.0f} deg @ {p['f_knit']} Hz, {p['n_loops']} loops, {p['tex']} tex):")
    print(f"  accel {a:.3e} | gyro {g:.3e} | I_dot {d:.3e} | total {T:.3e} N·m")
    print(f"  gyro/accel {g/a:.2e}   Idot/accel {d/a:.2e}")

if __name__ == "__main__":
    main()
