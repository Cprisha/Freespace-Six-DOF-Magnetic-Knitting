"""
Eddy-current torque supply sweep for an aluminium needle (idealised).
Companion to torque_sweep.py (the *required* torque, Eq. 17).

Model: long conducting cylinder (radius a) in a uniform field of amplitude B0
rotating at angular frequency W (needle ~stationary, so slip = W).
  Transverse polarisability per unit length : a_tr = -(2 pi a^2/mu0) F(ka)
  Axial      polarisability per unit length : a_ax = -(  pi a^2/mu0) F(ka)
  F(ka) = 1 - 2 J1(ka)/(ka J0(ka)),   k^2 = -j W mu0 sigma
Time-averaged torque (rotating field):
  * axial torque (field rotating in the plane transverse to the needle):
        tau = L * B0^2 * |Im a_tr|
  * tilt torque (field rotating in a plane containing the needle axis):
        tau = L * B0^2 * (|Im a_ax| + |Im a_tr|) / 2
L = length of needle exposed to the (assumed uniform) field.
Losses (stationary needle): P = tau * W.
Run:  python eddy_torque_sweep.py
"""
import numpy as np
from scipy.special import jv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torque_sweep as ts            # reuses the required-torque model

MU0 = 4e-7 * np.pi
P = dict(
    sigma20=3.5e7,       # S/m aluminium at 20 C
    alpha_T=0.0039,      # 1/K resistivity temperature coefficient
    T=40.0,              # C needle temperature
    a=1.5e-3,            # m needle radius
    L_act=0.030,         # m needle length exposed to the field
    B0=0.020,            # T field amplitude (fundamental)
)

def sigma(T): return P["sigma20"] / (1 + P["alpha_T"] * (T - 20))

def F(f, a, sig):
    W = 2 * np.pi * np.asarray(f, float)
    x = np.sqrt(-1j * W * MU0 * sig) * a
    x = np.where(np.abs(x) < 1e-9, 1e-9 + 0j, x)
    return 1 - 2 * jv(1, x) / (x * jv(0, x))

def torques(f, B0, a=None, sig=None, L=None):
    a = P["a"] if a is None else a
    sig = sigma(P["T"]) if sig is None else sig
    L = P["L_act"] if L is None else L
    ImF = np.abs(F(f, a, sig).imag)
    a_tr = 2 * np.pi * a**2 / MU0 * ImF
    a_ax = np.pi * a**2 / MU0 * ImF
    tau_axial = L * B0**2 * a_tr
    tau_tilt = L * B0**2 * (a_ax + a_tr) / 2
    return tau_axial, tau_tilt

def skin_depth(f, sig): return np.sqrt(2 / (2*np.pi*f * MU0 * sig))

def required():
    p = ts.PARAMS
    w_amp = p["theta_amp"] * 2*np.pi*p["f_knit"]
    tot = ts.torque_terms(p, w_amp, p["f_knit"])[3]
    _, In = ts.needle_inertia(p)
    ax = In[0, 0] * p["wax_ratio"] * w_amp * 2*np.pi*p["f_knit"]
    return ax, tot

def main():
    sig = sigma(P["T"]); req_ax, req_tilt = required()
    # sanity check against analytic low-frequency limit  Im F = (a/delta)^2 / 4
    f0 = 50.0
    assert abs(F(f0, P["a"], sig).imag / ((P["a"]/skin_depth(f0, sig))**2/4) - 1) < 1e-3
    print(f"Required torque: axial {req_ax:.2e} N·m, tilt {req_tilt:.2e} N·m")

    # ---- Fig 1: frequency x B0 maps -------------------------------------
    f = np.logspace(2, 5.3, 120); B = np.logspace(-3, -1, 100)
    FF, BB = np.meshgrid(f, B)
    tax, ttl = torques(FF, BB)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for k, (T, rq, ttl_) in enumerate([(tax, req_ax, "Axial torque"), (ttl, req_tilt, "Tilt torque")]):
        c = ax[k].contourf(f, B*1e3, np.log10(T), 24); plt.colorbar(c, ax=ax[k], label="log10 N·m")
        ax[k].contour(f, B*1e3, T, [rq], colors="r", linewidths=2)
        ax[k].set_title(f"{ttl_} (red = required {rq:.1e} N·m)")
    ratio = req_tilt and ttl / req_tilt
    c = ax[2].contourf(f, B*1e3, np.log10(ratio), np.linspace(-3, 3, 25), cmap="RdBu"); plt.colorbar(c, ax=ax[2], label="log10(available/required)")
    ax[2].contour(f, B*1e3, ratio, [1], colors="k", linewidths=2); ax[2].set_title("Tilt margin (black = 1)")
    for a_ in ax: a_.set_xscale("log"); a_.set_yscale("log"); a_.set_xlabel("field frequency (Hz)"); a_.set_ylabel("B0 (mT)")
    plt.tight_layout(); plt.savefig("eddy_sweep1_maps.png", dpi=130); plt.close()

    # ---- Fig 2: optimum frequency vs needle radius ----------------------
    f = np.logspace(2, 5.5, 300)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for a in (1e-3, 1.5e-3, 2e-3, 3e-3):
        t_ax, t_tl = torques(f, P["B0"], a=a)
        l, = ax[0].loglog(f, t_tl, label=f"a = {a*1e3:.1f} mm")
        i = np.argmax(t_tl); ax[0].plot(f[i], t_tl[i], "o", color=l.get_color())
        print(f"a={a*1e3:.1f} mm: peak tilt torque {t_tl[i]:.2e} N·m at {f[i]:.0f} Hz "
              f"(a/delta = {a/skin_depth(f[i], sig):.2f})")
        ax[1].semilogx(f, a / skin_depth(f, sig), color=l.get_color())
    ax[0].axhline(req_tilt, color="r", ls="--", label="required (nominal)")
    ax[0].set_xlabel("field frequency (Hz)"); ax[0].set_ylabel("tilt torque (N·m)")
    ax[0].set_title(f"Torque vs frequency, B0 = {P['B0']*1e3:.0f} mT, L = {P['L_act']*1e3:.0f} mm"); ax[0].legend(); ax[0].grid(True, which="both", alpha=.3)
    ax[1].set_xlabel("field frequency (Hz)"); ax[1].set_ylabel("a / skin depth"); ax[1].grid(True, alpha=.3)
    plt.tight_layout(); plt.savefig("eddy_sweep2_optimum.png", dpi=130); plt.close()

    # ---- Fig 3: heating, PWM harmonics, temperature ---------------------
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    t_ax, t_tl = torques(f, P["B0"])
    ax[0].loglog(f, t_tl * 2*np.pi*f, label="tilt"); ax[0].loglog(f, t_ax * 2*np.pi*f, label="axial")
    ax[0].set_title("Needle eddy-loss power P = τ·Ω (W)"); ax[0].set_xlabel("field frequency (Hz)"); ax[0].legend(); ax[0].grid(True, which="both", alpha=.3)
    fb = np.logspace(2, 4.5, 150); N = np.arange(1, 40, 2)
    tot = sum(torques(n*fb, P["B0"]/n)[1] for n in N)        # upper bound: all harmonics co-rotating
    ax[1].semilogx(fb, tot / torques(fb, P["B0"])[1]); ax[1].set_title("Square-wave drive: total / fundamental-only tilt torque\n(upper bound, harmonics assumed co-rotating)")
    ax[1].set_xlabel("fundamental (Hz)"); ax[1].grid(True, alpha=.3)
    for T in (20, 40, 80, 120):
        ax[2].semilogx(f, torques(f, P["B0"], sig=sigma(T))[1], label=f"{T} °C")
    ax[2].set_yscale("log"); ax[2].legend(); ax[2].set_title("Tilt torque vs needle temperature"); ax[2].set_xlabel("field frequency (Hz)"); ax[2].grid(True, which="both", alpha=.3)
    plt.tight_layout(); plt.savefig("eddy_sweep3_heat_pwm_temp.png", dpi=130); plt.close()

    t_ax, t_tl = torques(2e4, P["B0"])
    print(f"Nominal (20 kHz, {P['B0']*1e3:.0f} mT, {P['L_act']*1e3:.0f} mm): axial {t_ax:.2e}, tilt {t_tl:.2e} N·m, "
          f"loss {t_tl*2*np.pi*2e4:.1f} W, margin vs required tilt {t_tl/req_tilt:.2f}x")

if __name__ == "__main__":
    main()
