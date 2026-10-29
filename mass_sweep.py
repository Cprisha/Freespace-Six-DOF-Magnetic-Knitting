"""
Yarn mass vs needle mass sweep (Eq. 9) and lift-force budget (Eq. 5).
    m_eff(n) = m_needle + n * lambda * L_loop
    F_y > m_eff*g + F_yarn,y        with  F_yarn,y = T * sin(phi)  (simple tension model)
Run: python mass_sweep.py     All parameters are in PARAMS.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

G = 9.81
PARAMS = dict(
    rho=2700.0, r=1.5e-3, L=0.200,   # aluminium needle
    tex=100.0,                       # g/km
    L_loop=0.020,                    # m yarn per loop
    n=20,                            # loops on a needle
    N_total=40,                      # loops across both needles
    T=0.10,                          # N yarn tension (~10 cN)
    phi=np.deg2rad(30),              # yarn pull angle below horizontal
)
# rough yarn classes (tex) for reference lines
CLASSES = {"fingering": 25, "DK": 100, "chunky": 300, "super bulky": 800}

m_needle = lambda p: p["rho"] * np.pi * p["r"]**2 * p["L"]
m_loop = lambda tex, Ll: tex * 1e-6 * Ll          # tex g/km = 1e-6 kg/m
def m_eff(p, n, tex=None, Ll=None):
    return m_needle(p) + n * m_loop(p["tex"] if tex is None else tex, p["L_loop"] if Ll is None else Ll)
n_cross = lambda p, tex, Ll: m_needle(p) / m_loop(tex, Ll)   # loops where yarn mass = needle mass

def main():
    p = PARAMS; mn = m_needle(p)
    print(f"Needle mass {mn*1e3:.2f} g (weight {mn*G*1e3:.1f} mN)")
    print(f"One loop @ {p['tex']:.0f} tex, {p['L_loop']*1e3:.0f} mm: {m_loop(p['tex'], p['L_loop'])*1e6:.0f} mg")
    for name, tx in CLASSES.items():
        print(f"  {name:12s} ({tx:4d} tex): crossover n* = {n_cross(p, tx, p['L_loop']):8.0f} loops;"
              f" 40 loops add {100*40*m_loop(tx, p['L_loop'])/mn:.2f}% mass")

    # ---- Fig 1: yarn mass fraction and crossover -------------------------
    tex = np.logspace(1, 3.2, 120); Ll = np.linspace(0.005, 0.05, 100)
    TT, LL = np.meshgrid(tex, Ll)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    frac = 100 * p["n"] * m_loop(TT, LL) / mn
    c = ax[0].contourf(tex, Ll*1e3, np.log10(frac), 24); plt.colorbar(c, ax=ax[0], label="log10 (% of needle mass)")
    ax[0].contour(tex, Ll*1e3, frac, [1, 10, 100], colors="w", linewidths=1.5)
    ax[0].set_title(f"Yarn mass added by n = {p['n']} loops\n(white: 1%, 10%, 100%)")
    c = ax[1].contourf(tex, Ll*1e3, np.log10(n_cross(p, TT, LL)), 24); plt.colorbar(c, ax=ax[1], label="log10 n*")
    ax[1].set_title("Crossover loop count n* (yarn mass = needle mass)")
    for a_ in ax:
        a_.set_xscale("log"); a_.set_xlabel("yarn tex (g/km)"); a_.set_ylabel("loop length (mm)")
        for tx in CLASSES.values(): a_.axvline(tx, color="k", ls=":", alpha=.5)
    plt.tight_layout(); plt.savefig("mass_sweep1_fraction.png", dpi=130); plt.close()

    # ---- Fig 2: needle size vs crossover, and needle imbalance -----------
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    r = np.linspace(0.5e-3, 4e-3, 100)
    for name, tx in CLASSES.items():
        q = dict(p, r=r)
        ax[0].semilogy(r*1e3, m_needle(q) / m_loop(tx, p["L_loop"]), label=f"{name} ({tx} tex)")
    ax[0].axhline(p["N_total"], color="r", ls="--", label=f"N_total = {p['N_total']}")
    ax[0].set_xlabel("needle radius (mm)"); ax[0].set_ylabel("crossover loops n*")
    ax[0].set_title(f"Loops needed to match needle mass (L = {p['L']*1e3:.0f} mm, loop {p['L_loop']*1e3:.0f} mm)"); ax[0].legend(); ax[0].grid(True, which="both", alpha=.3)
    n1 = np.arange(0, p["N_total"] + 1)
    for name, tx in CLASSES.items():
        d = (m_eff(p, n1, tx) - m_eff(p, p["N_total"] - n1, tx))
        ax[1].plot(n1, d * G * 1e3, label=f"{name}")
    ax[1].set_xlabel("loops on needle 1 (n1)"); ax[1].set_ylabel("weight difference between needles (mN)")
    ax[1].set_title("Loop-transfer weight imbalance (n1 vs n2 = N_total - n1)"); ax[1].legend(); ax[1].grid(True, alpha=.3)
    plt.tight_layout(); plt.savefig("mass_sweep2_needle_imbalance.png", dpi=130); plt.close()

    # ---- Fig 3: lift budget (Eq. 5) --------------------------------------
    Tn = np.linspace(0.01, 1.0, 100); ph = np.deg2rad(np.linspace(0, 80, 100))
    TN, PH = np.meshgrid(Tn, ph)
    Fyarn = TN * np.sin(PH)
    W0 = mn * G
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    c = ax[0].contourf(Tn, np.rad2deg(ph), np.log10(Fyarn / W0 + 1e-6), 24, ); plt.colorbar(c, ax=ax[0], label="log10(F_yarn,y / bare-needle weight)")
    ax[0].contour(Tn, np.rad2deg(ph), Fyarn / W0, [1], colors="w", linewidths=2)
    ax[0].set_title("Yarn vertical pull vs bare-needle weight (white = 1)")
    ax[0].set_xlabel("yarn tension (N)"); ax[0].set_ylabel("pull angle below horizontal (deg)")
    labels = ["needle", "yarn loops", "yarn pull"]
    xs = list(CLASSES.items())
    bot = np.zeros(len(xs))
    for k, comp in enumerate([[W0]*len(xs),
                              [p["n"]*m_loop(tx, p["L_loop"])*G for _, tx in xs],
                              [p["T"]*np.sin(p["phi"])]*len(xs)]):
        ax[1].bar([x[0] for x in xs], np.array(comp)*1e3, bottom=bot*1e3, label=labels[k]); bot += np.array(comp)
    ax[1].set_ylabel("required lift (mN)"); ax[1].legend()
    ax[1].set_title(f"Eq. 5 budget: n = {p['n']}, T = {p['T']} N, φ = {np.rad2deg(p['phi']):.0f}°")
    tt = np.linspace(0, 1, 200)                 # one cycle: loops move from needle 2 to needle 1, tension pulses
    n_t = p["N_total"] / 2 + 10 * np.sin(2*np.pi*tt)
    T_t = p["T"] * (1 + 0.8 * np.sin(4*np.pi*tt))
    for name, tx in CLASSES.items():
        ax[2].plot(tt, 1e3 * (m_eff(p, n_t, tx) * G + T_t * np.sin(p["phi"])), label=name)
    ax[2].set_xlabel("cycle phase"); ax[2].set_ylabel("required lift on needle 1 (mN)")
    ax[2].set_title("Illustrative cycle: loop transfer ±10 + 80% tension swing"); ax[2].legend(); ax[2].grid(True, alpha=.3)
    plt.tight_layout(); plt.savefig("mass_sweep3_lift_budget.png", dpi=130); plt.close()

    Fy = p["T"] * np.sin(p["phi"])
    print(f"Nominal lift budget: weight {W0*1e3:.1f} mN + yarn loops {p['n']*m_loop(p['tex'],p['L_loop'])*G*1e3:.2f} mN"
          f" + yarn pull {Fy*1e3:.1f} mN  (pull = {Fy/W0:.1f}x needle weight)")

if __name__ == "__main__":
    main()
