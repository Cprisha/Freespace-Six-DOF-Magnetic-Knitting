"""
Yarn tension model F_yarn,y(d, theta) and dispenser-error sweep.
Companion to torque_sweep.py, eddy_torque_sweep.py, mass_sweep.py.

Geometry (paper Eq.18): two yarn legs from the two needle tips meet at an apex.
  apex angle = theta, tip separation = d, leg length l = (d/2)/sin(theta/2).
  The bridge plane is tilted by gamma below horizontal (gamma = 0: yarn horizontal).
  Force of the yarn on ONE needle tip (tension T), components in (X, Y, Z):
      F = T * ( -+sin(theta/2),  -cos(theta/2)*sin(gamma),  cos(theta/2)*cos(gamma) )
  so the vertical (Y) pull is  F_yarn,y = T*cos(theta/2)*sin(gamma)  (downward).
Tension (taut-only linear elastic yarn):
      eps = (L_req - S)/L_ref + eps0 ,  T = EA*max(0, eps),  EA = E_spec*tex
  L_req from Eq.19, S = length actually delivered by the dispenser.
Dispenser: commanded rate = g*dL_req/dt (feedforward, gain g), first-order lag tau_d.
Run: python tension_sweep.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torque_sweep as ts

P = dict(
    tex=100.0, E_spec=0.5,        # N per tex (EA = E_spec*tex)  ~ 50 N for DK
    T0=0.05,                      # N preload tension
    kg=1.0, L_loop=0.020, x=0.002,
    d_lo=0.010, d_hi=0.030,       # m   tip separation swing
    th_lo=np.deg2rad(30), th_hi=np.deg2rad(60),
    phase=np.deg2rad(60),         # phase of theta relative to d
    f=2.0,                        # Hz cycle rate
    gamma=np.deg2rad(30),         # bridge plane tilt below horizontal
    lever=0.100,                  # m tip-to-centre lever arm for torque
)
EA = lambda p: p["E_spec"] * p["tex"]

def traj(p, t):
    w = 2*np.pi*p["f"]
    d = p["d_lo"] + (p["d_hi"]-p["d_lo"]) * (1-np.cos(w*t))/2
    th = p["th_lo"] + (p["th_hi"]-p["th_lo"]) * (1-np.cos(w*t+p["phase"]))/2
    dd = (p["d_hi"]-p["d_lo"]) * w*np.sin(w*t)/2
    dth = (p["th_hi"]-p["th_lo"]) * w*np.sin(w*t+p["phase"])/2
    return d, th, dd, dth

L_req = lambda p, d, th: p["kg"]*d/np.sin(th/2) + p["L_loop"] + p["x"]
def L_req_dot(p, d, th, dd, dth):                      # Eq.20
    cs = 1/np.sin(th/2)
    return p["kg"]*dd*cs - 0.5*p["kg"]*d*dth*cs/np.tan(th/2)

def forces(p, T, th, gamma=None):
    g = p["gamma"] if gamma is None else gamma
    return T*np.sin(th/2), T*np.cos(th/2)*np.sin(g), T*np.cos(th/2)*np.cos(g)  # |Fx|, |Fy|, |Fz|

def simulate(p, g=1.0, tau_d=0.0, cycles=3, dt=1e-4):
    t = np.arange(0, cycles/p["f"], dt)
    d, th, dd, dth = traj(p, t)
    Lr = L_req(p, d, th); Lrd = L_req_dot(p, d, th, dd, dth)
    eps0 = p["T0"]/EA(p); Lref = Lr.mean()
    v = np.zeros_like(t); S = np.zeros_like(t); S[0] = Lr[0] - eps0*Lref
    v[0] = g*Lrd[0]
    for i in range(1, len(t)):
        cmd = g*Lrd[i]
        v[i] = cmd if tau_d <= 0 else v[i-1] + dt*(cmd - v[i-1])/tau_d
        S[i] = S[i-1] + dt*0.5*(v[i-1] + v[i])   # trapezoidal
    eps = (Lr - S)/Lref
    T = EA(p)*np.maximum(0, eps)
    last = t >= t[-1] - 1/p["f"]
    return dict(t=t, d=d, th=th, Lr=Lr, Lrd=Lrd, v=v, T=T, eps=eps, last=last)

def metrics(r):
    m = r["last"]
    return r["T"][m].max(), (r["eps"][m] <= 0).mean()

def main():
    p = P; req_ax, req_tilt = 0, 0
    w_amp = ts.PARAMS["theta_amp"]*2*np.pi*ts.PARAMS["f_knit"]
    req_tilt = ts.torque_terms(ts.PARAMS, w_amp, ts.PARAMS["f_knit"])[3]
    Lm = L_req(p, *traj(p, np.linspace(0, 1/p['f'], 400))[:2]).mean()
    print(f"EA = {EA(p):.0f} N; 1% strain = {0.01*EA(p):.2f} N; 0.1 N tension = {1e6*0.1/EA(p)*Lm:.0f} um stretch over mean L_req {Lm*1e3:.0f} mm")

    # sanity: ideal dispenser keeps tension at the preload
    r0 = simulate(p); print(f"Ideal feedforward: T range {r0['T'][r0['last']].min():.3f}..{r0['T'][r0['last']].max():.3f} N")

    # rigid-V check: needles pivot about a common vertex, needle length l, so d = 2 l sin(theta/2)
    th = np.linspace(p["th_lo"], p["th_hi"], 50); ell = 0.2
    print(f"Rigid V (d=2 l sin(th/2)): bridge length d*csc(th/2) = {np.ptp(2*ell*np.sin(th/2)/np.sin(th/2))*1e3:.3f} mm variation (kg=1)"
          f" vs {np.ptp(L_req(p, *traj(p, np.linspace(0, 1/p['f'], 400))[:2]))*1e3:.0f} mm for the independent d,theta cycle")

    # ---- Fig 1: geometry maps ---------------------------------------------
    d = np.linspace(0.02, 0.12, 120); th = np.deg2rad(np.linspace(8, 90, 120))
    D, TH = np.meshgrid(d, th)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    c = ax[0].contourf(d*1e3, np.rad2deg(th), L_req(p, D, TH)*1e3, 24); plt.colorbar(c, ax=ax[0], label="mm")
    ax[0].set_title("Required yarn length L_req (Eq.19)")
    Fy = forces(p, p["T0"], TH)[1]; Wn = ts.needle_inertia(ts.PARAMS)[0]*9.81
    c = ax[1].contourf(d*1e3, np.rad2deg(th), Fy*1e3, 24); plt.colorbar(c, ax=ax[1], label="mN")
    ax[1].set_title(f"Vertical yarn pull per needle, T = {p['T0']} N, γ = {np.rad2deg(p['gamma']):.0f}°\n(needle weight {Wn*1e3:.0f} mN)")
    tau = p["lever"]*p["T0"]*np.ones_like(TH)
    Tn_ = np.linspace(0.01, 0.5, 120); cc = ax[2].plot(Tn_, p["lever"]*Tn_*1e3, label="tip-pull torque = lever × T")
    ax[2].axhline(req_tilt*1e3, color="r", ls="--", label=f"Eq.17 inertial torque {req_tilt*1e3:.2f} mN·m")
    ax[2].set_xlabel("tension (N)"); ax[2].set_ylabel("torque (mN·m)"); ax[2].legend(); ax[2].grid(True, alpha=.3)
    ax[2].set_title(f"Torque from yarn pull at the tip (lever {p['lever']*1e3:.0f} mm, upper bound)")
    for a_ in ax[:2]: a_.set_xlabel("tip separation d (mm)"); a_.set_ylabel("θ (deg)")
    plt.tight_layout(); plt.savefig("tension_sweep1_geometry.png", dpi=130); plt.close()

    # ---- Fig 2: time traces for dispenser imperfections -------------------
    cases = [("ideal", 1.0, 0.0), ("lag 2 ms", 1.0, 0.002), ("gain -0.5%", 0.995, 0.0), ("gain +0.5%", 1.005, 0.0)]
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for name, g, tau_d in cases:
        r = simulate(p, g, tau_d); m = r["last"]; tt = (r["t"][m]-r["t"][m][0])*p["f"]
        ax[0, 0].plot(tt, r["T"][m], label=name)
        ax[0, 1].plot(tt, 1e3*forces(p, r["T"][m], r["th"][m])[1], label=name)
    r = simulate(p); m = r["last"]; tt = (r["t"][m]-r["t"][m][0])*p["f"]
    ax[1, 0].plot(tt, r["Lr"][m]*1e3); ax[1, 0].set_ylabel("L_req (mm)")
    ax2 = ax[1, 0].twinx(); ax2.plot(tt, r["Lrd"][m]*1e3, "r"); ax2.set_ylabel("dL_req/dt (mm/s)", color="r")
    ax[1, 1].plot(tt, np.rad2deg(r["th"][m]), label="θ (deg)"); ax[1, 1].plot(tt, r["d"][m]*1e3, label="d (mm)"); ax[1, 1].legend()
    ax[0, 0].set_ylabel("tension (N)"); ax[0, 0].legend(); ax[0, 0].set_title("Yarn tension, one cycle")
    ax[0, 1].set_ylabel("vertical yarn pull (mN)"); ax[0, 1].set_title("F_yarn,y per needle")
    ax[1, 0].set_title("Yarn demand (Eqs.19-20)"); ax[1, 1].set_title("Prescribed trajectory")
    for a_ in ax.flat: a_.set_xlabel("cycle phase"); a_.grid(True, alpha=.3)
    plt.tight_layout(); plt.savefig("tension_sweep2_traces.png", dpi=130); plt.close()

    # ---- Fig 3: sensitivity to dispenser errors ---------------------------
    gs = np.linspace(0.99, 1.01, 25); taus = np.linspace(0, 0.01, 17)
    pk = np.zeros((len(taus), len(gs))); sl = np.zeros_like(pk)
    for i, tdd in enumerate(taus):
        for j, g in enumerate(gs):
            pk[i, j], sl[i, j] = metrics(simulate(p, g, tdd, cycles=2, dt=2e-4))
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    c = ax[0].contourf((gs-1)*100, taus*1e3, np.log10(pk + 1e-3), 24); plt.colorbar(c, ax=ax[0], label="log10 N")
    ax[0].contour((gs-1)*100, taus*1e3, pk, [0.2, 0.5], colors="w")
    ax[0].set_title("Peak tension (white contours: 0.2, 0.5 N)")
    c = ax[1].contourf((gs-1)*100, taus*1e3, 100*sl, 24); plt.colorbar(c, ax=ax[1], label="% of cycle")
    ax[1].set_title("Time slack (strain ≤ 0)")
    for a_ in ax[:2]: a_.set_xlabel("dispenser gain error (%)"); a_.set_ylabel("lag τ_d (ms)")
    for tx in (25, 100, 300, 800):
        q = dict(p, tex=tx); pkg = [metrics(simulate(q, g, 0.0, cycles=2, dt=2e-4))[0] for g in gs]
        ax[2].plot((gs-1)*100, pkg, label=f"{tx} tex")
    ax[2].set_yscale("log"); ax[2].set_xlabel("gain error (%)"); ax[2].set_ylabel("peak tension (N)"); ax[2].legend(); ax[2].grid(True, which="both", alpha=.3)
    ax[2].set_title("Yarn stiffness: tension error ∝ tex")
    plt.tight_layout(); plt.savefig("tension_sweep3_sensitivity.png", dpi=130); plt.close()

    for name, g, tau_d in cases:
        r = simulate(p, g, tau_d); m = r["last"]; Tm = r["T"][m]
        print(f"{name:10s}: T {Tm.min():.3f}..{Tm.max():.3f} N, slack {100*(r['eps'][m]<=0).mean():.0f}% of cycle,"
              f" peak F_y {1e3*forces(p, Tm.max(), r['th'][m].min())[1]:.0f} mN")
    print(f"Peak L_req rate {np.abs(r0['Lrd']).max()*1e3:.0f} mm/s, swing {np.ptp(r0['Lr'])*1e3:.0f} mm")

if __name__ == "__main__":
    main()
