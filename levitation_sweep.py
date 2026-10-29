"""
Levitation (vertical force) sweep: what the array can supply vs what the needle needs.
Companion to torque_sweep.py, eddy_torque_sweep.py, mass_sweep.py, tension_sweep.py.

Field model: a periodic array of pitch p gives a field that decays with height as
    B(y) = Bs * exp(-2 pi y / p) = Bs * exp(-y/h),   h = p/(2 pi)        (Bs = peak surface field)
Mechanisms (all idealised, needle treated as a small dipole -> valid only for a << h):
 1. Eddy-current repulsion of the aluminium body (AC field, transverse to the needle):
        F = (pi a^2 L / (mu0 h)) * Re F(ka) * B^2          [from F = (1/4) Re(alpha) grad|B|^2]
        P = (w/2) * (2 pi a^2/mu0) * Im F(ka) * L * B^2    (loss in the needle)
    => hover power per newton:  P/F = w h ImF/ReF   (independent of B and L)
 2. Soft-magnetic strip (quasi-static, thin-plate limit M = B/mu0):   F = duty V B^2/(mu0 h)
 3. Permanent-magnet strip (M = Br/mu0, quasi-static):               F = duty V (Br/mu0) B / h
Required lift = needle weight + yarn vertical pull (from mass_sweep / tension_sweep).
Run: python levitation_sweep.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import eddy_torque_sweep as e
import mass_sweep as ms
import tension_sweep as tn

MU0 = e.MU0
LP = dict(
    p=0.020,          # m   array pitch (field decay length h = p/2pi)
    y=0.0025,         # m   needle centre height above array surface
    L=0.030,          # m   needle length exposed to the field
    f=2e4,            # Hz  AC frequency
    Bs=0.10,          # T   peak surface field
    T=40.0,           # C   needle temperature
    w_s=2e-3, t_s=2e-4, duty=0.5, Br=1.2,   # strip: width, thickness, lm/pm, PM remanence
    c_al=900.0,       # J/kg/K
)
a = e.P["a"]; sig = lambda T: e.sigma(T)

h_of = lambda p: p / (2*np.pi)
Bneedle = lambda Bs, p, y: Bs*np.exp(-y/h_of(p))

def eddy(f, Bs, p, y, L, T=None, a_=None):
    a_ = a if a_ is None else a_
    Ff = e.F(f, a_, sig(LP["T"] if T is None else T)); h = h_of(p); B = Bneedle(Bs, p, y)
    F = np.pi*a_**2*L/(MU0*h) * Ff.real * B**2
    P = 0.5*(2*np.pi*f)*(2*np.pi*a_**2/MU0) * np.abs(Ff.imag) * L * B**2
    return F, P

def strips(Bs, p, y, L):
    h = h_of(p); B = Bneedle(Bs, p, y); V = LP["duty"]*LP["w_s"]*LP["t_s"]*L
    return V*B**2/(MU0*h), V*(LP["Br"]/MU0)*B/h        # soft, PM

def required():
    W = ms.m_needle(ms.PARAMS)*9.81
    r_i = tn.simulate(tn.P); m = r_i["last"]; Fi = tn.forces(tn.P, r_i["T"][m], r_i["th"][m])[1].max()
    r_l = tn.simulate(tn.P, 1.0, 0.002); m = r_l["last"]; Fl = tn.forces(tn.P, r_l["T"][m], r_l["th"][m])[1].max()
    return W, W + Fi, W + Fl

def main():
    p0 = LP; W, req_i, req_l = required()
    print(f"Required lift: weight {W*1e3:.1f} mN | + ideal-feedforward yarn pull {req_i*1e3:.1f} mN | + 2 ms-lag pull {req_l*1e3:.0f} mN")
    print(f"a/h = {a/h_of(p0['p']):.2f} at p = {p0['p']*1e3:.0f} mm (dipole model needs a << h)")

    # ---- Fig 1: Bs x f map, hover power per newton -----------------------
    f = np.logspace(2.7, 5.5, 160); Bs = np.logspace(-2, np.log10(0.5), 120)
    FF, BB = np.meshgrid(f, Bs); Fm, _ = eddy(FF, BB, p0["p"], p0["y"], p0["L"])
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    c = ax[0].contourf(f, Bs*1e3, np.log10(Fm*1e3), 24); plt.colorbar(c, ax=ax[0], label="log10 lift (mN)")
    for lvl, col, nm in [(W, "w", "weight"), (req_i, "r", "weight+yarn (ideal)"), (req_l, "k", "weight+yarn (2 ms lag)")]:
        ax[0].contour(f, Bs*1e3, Fm, [lvl], colors=col, linewidths=2)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("frequency (Hz)"); ax[0].set_ylabel("surface field Bs (mT)")
    ax[0].set_title(f"Eddy lift, p={p0['p']*1e3:.0f} mm, y={p0['y']*1e3:.1f} mm, L={p0['L']*1e3:.0f} mm\n(white/red/black = weight / +yarn / +lag)")
    ff = np.logspace(2.5, 5.7, 400)
    for pp in (10e-3, 20e-3, 40e-3):
        Ff = e.F(ff, a, sig(p0["T"])); ratio = 2*np.pi*ff*h_of(pp)*np.abs(Ff.imag)/Ff.real
        l, = ax[1].loglog(ff, ratio, label=f"p = {pp*1e3:.0f} mm")
        plateau = 6*h_of(pp)/(MU0*sig(p0["T"])*a**2)          # low-frequency limit of P/F
        ax[1].axhline(plateau, color=l.get_color(), ls=":")
        print(f"p={pp*1e3:.0f} mm: P/F plateau (f->0) = 6h/(mu0 sigma a^2) = {plateau:.0f} W/N -> {plateau*req_i:.1f} W for {req_i*1e3:.0f} mN"
              f" (numeric min {ratio.min():.0f})")
    ax[1].set_xlabel("frequency (Hz)"); ax[1].set_ylabel("hover power per newton (W/N)"); ax[1].legend(); ax[1].grid(True, which="both", alpha=.3)
    ax[1].set_title("Needle eddy loss per N of lift, P/F = ω h Im F / Re F\n(dotted: low-frequency plateau 6h/(μ0 σ a²))")
    Ff = e.F(ff, a, sig(p0["T"]))
    for lvl, nm in [(W, "weight"), (req_i, "+yarn (ideal)"), (req_l, "+yarn (2 ms lag)")]:
        F0 = np.pi*a**2*p0["L"]/(MU0*h_of(p0["p"]))*Ff.real*np.exp(-2*p0["y"]/h_of(p0["p"]))
        ax[2].loglog(ff, np.sqrt(lvl/F0)*1e3, label=nm)
    ax[2].set_xlabel("frequency (Hz)"); ax[2].set_ylabel("Bs needed to hover (mT)"); ax[2].legend(); ax[2].grid(True, which="both", alpha=.3)
    ax[2].set_title("Surface field needed vs frequency")
    plt.tight_layout(); plt.savefig("levitation_sweep1_eddy.png", dpi=130); plt.close()

    # ---- Fig 2: geometry (pitch, height), equilibrium height, stiffness ---
    pv = np.linspace(5e-3, 50e-3, 100); yv = np.array([1.5e-3, 2.5e-3, 4e-3, 6e-3])
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for yy in yv:
        F1, _ = eddy(p0["f"], 1.0, pv, yy, p0["L"])          # lift per T^2
        ax[0].semilogy(pv*1e3, np.sqrt(req_i/F1)*1e3, label=f"y = {yy*1e3:.1f} mm")
    ax[0].axvline(4*np.pi*a*1e3, color="k", ls=":"); ax[0].text(4*np.pi*a*1e3+.5, ax[0].get_ylim()[0]*1.5, "a/h = 0.5\n(dipole model\nok to the right)", fontsize=8)
    ax[0].set_xlabel("array pitch p (mm)"); ax[0].set_ylabel("Bs needed to hover (mT)"); ax[0].legend(); ax[0].grid(True, which="both", alpha=.3)
    ax[0].set_title(f"Hover field vs pitch ({p0['f']/1e3:.0f} kHz, L={p0['L']*1e3:.0f} mm, load {req_i*1e3:.0f} mN)")
    Bsv = np.linspace(0.02, 0.4, 200)
    for pp in (10e-3, 20e-3, 40e-3):
        h = h_of(pp); F0 = eddy(p0["f"], Bsv, pp, 0.0, p0["L"])[0]
        yeq = 0.5*h*np.log(F0/req_i); yeq[yeq < a] = np.nan
        ax[1].plot(Bsv*1e3, yeq*1e3, label=f"p = {pp*1e3:.0f} mm")
    ax[1].set_xlabel("Bs (mT)"); ax[1].set_ylabel("equilibrium height of needle centre (mm)"); ax[1].legend(); ax[1].grid(True, alpha=.3)
    ax[1].set_title("Levitation height (open-loop equilibrium)")
    k = 4*np.pi*req_i/pv; fn = np.sqrt(k/ms.m_needle(ms.PARAMS))/(2*np.pi)
    ax[2].plot(pv*1e3, fn, label="eddy repulsion: k = 4πF/p")
    kpm = 2*np.pi*req_i/pv; ax[2].plot(pv*1e3, np.sqrt(kpm/ms.m_needle(ms.PARAMS))/(2*np.pi), "--", label="PM strip: k = 2πF/p")
    ax[2].set_xlabel("array pitch p (mm)"); ax[2].set_ylabel("vertical natural frequency (Hz)"); ax[2].legend(); ax[2].grid(True, alpha=.3)
    ax[2].set_title("Open-loop vertical stiffness at hover (restoring)")
    plt.tight_layout(); plt.savefig("levitation_sweep2_geometry.png", dpi=130); plt.close()

    # ---- Fig 3: mechanisms and thermal cost -------------------------------
    Bsv = np.logspace(-2, np.log10(0.5), 200)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for L in (0.03, 0.1, 0.2):
        ax[0].loglog(Bsv*1e3, eddy(p0["f"], Bsv, p0["p"], p0["y"], L)[0]*1e3, label=f"eddy, L={L*1e3:.0f} mm")
    soft, pm = strips(Bsv, p0["p"], p0["y"], 0.2)
    ax[0].loglog(Bsv*1e3, soft*1e3, "--", label="soft strip (L=200 mm)"); ax[0].loglog(Bsv*1e3, pm*1e3, ":", lw=2, label="PM strip (L=200 mm)")
    for lvl, col in [(W, "gray"), (req_i, "r"), (req_l, "k")]: ax[0].axhline(lvl*1e3, color=col, ls="-.", lw=1)
    ax[0].set_xlabel("Bs (mT)"); ax[0].set_ylabel("lift (mN)"); ax[0].legend(fontsize=8); ax[0].grid(True, which="both", alpha=.3)
    ax[0].set_title("Lift by mechanism (lines: weight / +yarn / +lag)")
    reqs = np.linspace(W, req_l, 50)
    m_n = ms.m_needle(ms.PARAMS)
    for fq, nm in [(2e3, "2 kHz"), (5e3, "5 kHz"), (2e4, "20 kHz"), (1e5, "100 kHz")]:
        Ff = e.F(fq, a, sig(p0["T"])); Pn = reqs*2*np.pi*fq*h_of(p0["p"])*abs(Ff.imag)/Ff.real
        ax[1].plot(reqs*1e3, Pn, label=nm)
        ax[2].plot(reqs*1e3, Pn/(m_n*p0["c_al"]), label=nm)
    ax[1].set_xlabel("required lift (mN)"); ax[1].set_ylabel("needle eddy loss while hovering (W)"); ax[1].legend(); ax[1].grid(True, alpha=.3)
    ax[1].set_title("Hover power (independent of L and B)")
    ax[2].set_xlabel("required lift (mN)"); ax[2].set_ylabel("adiabatic heating rate of needle (K/s)"); ax[2].legend(); ax[2].grid(True, alpha=.3)
    ax[2].set_title("Needle heating if heat is not removed")
    plt.tight_layout(); plt.savefig("levitation_sweep3_mechanisms.png", dpi=130); plt.close()

    print(f"Hover at {req_i*1e3:.0f} mN (weight + ideal yarn pull); needle eddy loss and Bs needed vs frequency:")
    for fq in (2e3, 5e3, 1e4, 2e4, 5e4, 1e5):
        Ff = e.F(fq, a, sig(p0["T"])); Pn = req_i*2*np.pi*fq*h_of(p0["p"])*abs(Ff.imag)/Ff.real
        Bn = [1e3*np.sqrt(req_i/eddy(fq, 1.0, p0["p"], p0["y"], L)[0]) for L in (0.03, 0.1, 0.2)]
        print(f"  {fq/1e3:5.0f} kHz: loss {Pn:5.1f} W ({Pn/(m_n*p0['c_al']):.1f} K/s) | Bs for L=30/100/200 mm: {Bn[0]:.0f}/{Bn[1]:.0f}/{Bn[2]:.0f} mT")
    soft, pm = strips(0.1, p0["p"], p0["y"], 0.2)
    print(f"At Bs=100 mT, L=200 mm: soft strip {soft*1e3:.2f} mN, PM strip {pm*1e3:.1f} mN, eddy@20kHz {eddy(2e4, .1, p0['p'], p0['y'], .2)[0]*1e3:.0f} mN")

if __name__ == "__main__":
    main()
