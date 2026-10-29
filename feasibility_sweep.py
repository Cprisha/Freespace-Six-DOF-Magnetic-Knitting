"""
Combined feasibility sweep: lift + tilt torque + field limit + needle heating.
Reuses torque_sweep, eddy_torque_sweep, mass_sweep, tension_sweep, levitation_sweep.

For each design point (frequency f, exposed length L, array pitch p):
  required lift   F_req   = weight + yarn vertical pull              (levitation_sweep.required)
  required torque tau_req = Eq.17 inertial torque + kappa*lever*T0   (kappa = fraction of the tip-pull torque
                                                                      the needle must actually resist)
  Bs_L, P_L : surface field and needle loss to hover   (A: eddy repulsion, B: permanent-magnet strip, P_L = 0)
  Bs_T, P_T : surface field and needle loss to produce tau_req with a rotating field (P_T = tau_req * 2 pi f)
  Bs_tot = sqrt(Bs_L^2 + Bs_T^2)   (fields assumed to superpose in quadrature), P_tot = P_L + P_T
  margin = min(Bs_max/Bs_tot, P_max/P_tot)   ->  feasible if margin >= 1
Run: python feasibility_sweep.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import eddy_torque_sweep as e
import torque_sweep as ts
import tension_sweep as tn
import levitation_sweep as lv

MU0 = e.MU0
CFG = dict(
    Bs_max=0.30,      # T   peak surface field the array can produce
    P_max=2.0,        # W   needle heating the design can tolerate
    p=0.020, y=0.0025,
    kappas=(0.0, 0.1, 1.0),
    lever=0.100,
)

def requirements(kappa, lag=False):
    W, req_i, req_l = lv.required()
    w_amp = ts.PARAMS["theta_amp"]*2*np.pi*ts.PARAMS["f_knit"]
    tau_in = ts.torque_terms(ts.PARAMS, w_amp, ts.PARAMS["f_knit"])[3]
    tau = tau_in + kappa*CFG["lever"]*tn.P["T0"]
    return (req_l if lag else req_i), tau

def evaluate(f, L, p, y, F_req, tau_req, lift="eddy"):
    h = p/(2*np.pi); env = np.exp(-y/h)
    if lift == "eddy":
        F1, P1 = lv.eddy(f, 1.0, p, y, L)           # per Bs^2
        Bs_L = np.sqrt(F_req/F1); P_L = F_req*P1/F1
    else:
        V = lv.LP["duty"]*lv.LP["w_s"]*lv.LP["t_s"]*L
        Bs_L = F_req*h*MU0/(V*lv.LP["Br"]*env) ; P_L = 0.0*Bs_L
    tau1 = e.torques(f, 1.0, L=L)[1]*env**2
    Bs_T = np.sqrt(tau_req/tau1); P_T = tau_req*2*np.pi*f
    Bs = np.sqrt(Bs_L**2 + Bs_T**2); P = P_L + P_T
    return Bs, P, np.minimum(CFG["Bs_max"]/Bs, CFG["P_max"]/P)

def main():
    c = CFG; fg = np.logspace(2.5, 5.5, 140); Lg = np.linspace(0.02, 0.2, 90)
    FF, LL = np.meshgrid(fg, Lg)
    F_req, _ = requirements(0)
    print(f"Required lift {F_req*1e3:.0f} mN | limits: Bs <= {c['Bs_max']*1e3:.0f} mT, needle loss <= {c['P_max']} W")
    for k in c["kappas"]:
        print(f"  kappa={k}: required tilt torque {requirements(k)[1]*1e3:.2f} mN·m")

    # ---- Fig 1: feasibility maps (f, L) -------------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    for r, (lift, nm) in enumerate([("eddy", "A: eddy lift + eddy torque"), ("pm", "B: PM-strip lift + eddy torque")]):
        for j, k in enumerate(c["kappas"]):
            Fq, tq = requirements(k)
            *_, m = evaluate(FF, LL, c["p"], c["y"], Fq, tq, lift)
            cs = ax[r, j].contourf(fg, Lg*1e3, np.log10(np.clip(m, 1e-3, 10)), np.linspace(-3, 1, 17), cmap="RdYlGn", extend="both")
            ax[r, j].contour(fg, Lg*1e3, m, [1], colors="k", linewidths=2.5)
            ax[r, j].set_xscale("log"); ax[r, j].set_title(f"{nm}\nκ = {k} (τ_req = {tq*1e3:.2f} mN·m)")
            ax[r, j].set_xlabel("frequency (Hz)"); ax[r, j].set_ylabel("exposed length L (mm)")
            plt.colorbar(cs, ax=ax[r, j], label="log10 margin (black = 1, green feasible)")
    plt.tight_layout(); plt.savefig("feasibility_sweep1_maps.png", dpi=130); plt.close()

    # ---- Fig 2: constraint curves at L = 200 mm ------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5)); f = np.logspace(2.5, 5.5, 300)
    for k, col in zip(c["kappas"], ("C0", "C1", "C3")):
        Fq, tq = requirements(k)
        for lift, ls in (("eddy", "-"), ("pm", "--")):
            Bs, P, _ = evaluate(f, 0.2, c["p"], c["y"], Fq, tq, lift)
            ax[0].loglog(f, Bs*1e3, ls, color=col, label=f"κ={k}, {'eddy' if lift=='eddy' else 'PM'} lift")
            ax[1].loglog(f, P, ls, color=col)
    ax[0].axhline(c["Bs_max"]*1e3, color="k", ls=":"); ax[1].axhline(c["P_max"], color="k", ls=":")
    ax[0].set_ylabel("required surface field (mT)"); ax[1].set_ylabel("needle loss (W)")
    ax[0].legend(fontsize=8)
    for a_ in ax: a_.set_xlabel("frequency (Hz)"); a_.grid(True, which="both", alpha=.3)
    ax[0].set_title(f"Field needed, L = 200 mm, p = {c['p']*1e3:.0f} mm"); ax[1].set_title("Needle loss (lift + torque): torque loss = τ·Ω grows with f")
    plt.tight_layout(); plt.savefig("feasibility_sweep2_constraints.png", dpi=130); plt.close()

    # ---- Fig 3: best achievable margin vs L and vs pitch ---------------------
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5)); Ls = np.linspace(0.02, 0.2, 40); ps = np.linspace(6e-3, 40e-3, 40)
    for k, col in zip(c["kappas"], ("C0", "C1", "C3")):
        Fq, tq = requirements(k)
        for lift, ls in (("eddy", "-"), ("pm", "--")):
            mL = [evaluate(f, L_, c["p"], c["y"], Fq, tq, lift)[2].max() for L_ in Ls]
            mp = [evaluate(f, 0.2, p_, c["y"], Fq, tq, lift)[2].max() for p_ in ps]
            ax[0].semilogy(Ls*1e3, mL, ls, color=col, label=f"κ={k}, {lift}"); ax[1].semilogy(ps*1e3, mp, ls, color=col)
    for a_ in ax: a_.axhline(1, color="k", ls=":"); a_.grid(True, which="both", alpha=.3); a_.set_ylabel("best margin over frequency (≥1 feasible)")
    ax[0].set_xlabel("exposed length L (mm)"); ax[0].legend(fontsize=8); ax[0].set_title(f"Best margin vs L (p = {c['p']*1e3:.0f} mm)")
    ax[1].set_xlabel("array pitch p (mm)"); ax[1].set_title("Best margin vs pitch (L = 200 mm)")
    plt.tight_layout(); plt.savefig("feasibility_sweep3_best_margin.png", dpi=130); plt.close()

    # ---- Table ---------------------------------------------------------------
    print("\nBest design point per case (p = 20 mm). Margin >= 1 means feasible.")
    print(f"{'yarn':5s} {'kappa':>5s} {'L mm':>5s} {'lift':>5s} {'best margin':>11s} {'f*(kHz)':>8s} {'Bs mT':>7s} {'P W':>6s} | min P at Bs<=Bmax")
    for lag in (False, True):
        for k in c["kappas"]:
            for L_ in (0.03, 0.2):
                for lift in ("eddy", "pm"):
                    Fq, tq = requirements(k, lag); Bs, P, m = evaluate(f, L_, c["p"], c["y"], Fq, tq, lift)
                    i = np.argmax(m); ok = Bs <= c["Bs_max"]
                    pmin = f"{P[ok].min():.2f}" if ok.any() else "none"
                    print(f"{'lag' if lag else 'ideal':5s} {k:5.1f} {L_*1e3:5.0f} {lift:>5s} {m[i]:11.3f} {f[i]/1e3:8.1f} {Bs[i]*1e3:7.0f} {P[i]:6.2f} | {pmin}")

if __name__ == "__main__":
    main()
