"""
Free-Space Magnetic Knitting Needle Solver
==========================================
Given two 3D endpoints (r1, r2) of the knitting needle, computes:
  1. The 6 Degrees of Freedom (position + orientation)
  2. Minimum solenoid field strength for levitation
  3. Eddy current torque at each PWM frequency

Coordinate system (right-handed):
  - Y axis : vertical, normal to solenoid array surface
  - X axis : horizontal, along solenoid array length
  - Z axis : horizontal, depth

Needle (hardcoded geometry):
  - Length  : 20 cm
  - Mass    : 10 g
  - Cross-section (aluminum core): circular, radius 2 mm
  - Effective eddy current resistance: 0.05 Ohm
  - Magnetic strip: 1 inch active / 1 inch inactive, repeating
  - Each active segment = one magnetic dipole oriented along Y

Solenoid array:
  - Lies on XZ plane at Y = 0
  - Each solenoid modeled as a magnetic dipole oriented along +Y
  - Square grid, configurable density and spacing

PWM sweep range: 50, 100, 500, 1000, 2000, 5000, 10000 Hz
"""

import numpy as np
from itertools import product
import csv
import os


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MU_0         = 4 * np.pi * 1e-7   # Permeability of free space (T·m/A)
G            = 9.81                # Gravitational acceleration (m/s²)
INCH_TO_M    = 0.0254              # Unit conversion

# ── Hardcoded needle parameters ───────────────────────────────────────────────
NEEDLE_MASS          = 0.010       # kg  (10 g)
NEEDLE_LENGTH        = 0.20        # m   (20 cm)
CORE_RADIUS          = 0.002       # m   (2 mm aluminum core radius)
CORE_AREA            = np.pi * CORE_RADIUS**2   # m²  cross-sectional area
EDDY_RESISTANCE      = 0.05        # Ohm effective resistance of eddy current path
POLE_SPACING         = INCH_TO_M   # m   active segment length
POLE_PERIOD          = 2 * INCH_TO_M  # m  full on+off cycle

# Moment of inertia of needle (thin rod about center, perpendicular axis)
MOMENT_OF_INERTIA    = (1/12) * NEEDLE_MASS * NEEDLE_LENGTH**2   # kg·m²

# ── PWM frequency sweep ───────────────────────────────────────────────────────
PWM_FREQUENCIES = [50, 100, 500, 1000, 2000, 5000, 10000]   # Hz

# ── Sweep configurations ──────────────────────────────────────────────────────
ARRAY_CONFIGS = [
    (10,  5,  0.020, "10x5  @ 2.0cm"),
    (14,  7,  0.015, "14x7  @ 1.5cm"),
    (20, 10,  0.012, "20x10 @ 1.2cm"),
    (28, 14,  0.010, "28x14 @ 1.0cm"),
]

POLE_MOMENTS  = [0.05, 0.10, 0.20, 0.50, 1.00]   # A·m² per active pole


# ─────────────────────────────────────────────────────────────────────────────
# 6-DOF State from Two Endpoints
# ─────────────────────────────────────────────────────────────────────────────

def compute_6dof(r1: np.ndarray, r2: np.ndarray) -> dict:
    """
    Derives the full 6-DOF state of the needle from its two endpoints.

    Translational DOFs (position of needle center):
      X : left-right
      Y : height (levitation)
      Z : forward-back

    Rotational DOFs (Euler angles of needle axis):
      XY plane rotation : yaw   (rotation about Z-axis)
      YZ plane rotation : pitch (rotation about X-axis)
      ZX plane rotation : roll  (rotation about Y-axis)

    Returns a dictionary of all 6 DOFs plus derived quantities.
    """
    center    = (r1 + r2) / 2.0
    axis      = r2 - r1
    length    = np.linalg.norm(axis)
    axis_unit = axis / length

    # Rotational DOFs from needle axis direction vector
    # Yaw: angle of projection onto XZ plane from X-axis
    yaw   = np.degrees(np.arctan2(axis_unit[2], axis_unit[0]))

    # Pitch: angle of elevation from XZ plane
    pitch = np.degrees(np.arcsin(np.clip(axis_unit[1], -1.0, 1.0)))

    # Roll: rotation about needle's own axis — not derivable from endpoints alone
    # (requires a reference mark on the needle; set to 0 by convention here)
    roll  = 0.0

    return {
        # Translational
        "X_m":       center[0],
        "Y_m":       center[1],
        "Z_m":       center[2],
        # Rotational
        "yaw_deg":   yaw,
        "pitch_deg": pitch,
        "roll_deg":  roll,
        # Derived
        "center":    center,
        "axis_unit": axis_unit,
        "length_m":  length,
        "r1":        r1,
        "r2":        r2,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Geometry Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_pole_positions(r1: np.ndarray, r2: np.ndarray) -> list:
    """
    Returns 3D positions of each discrete magnetic pole center along the needle.
    Pattern: 1 inch active, 1 inch inactive, repeating from r1 to r2.
    """
    length = np.linalg.norm(r2 - r1)
    n      = int(length / POLE_PERIOD)
    rem    = length - n * POLE_PERIOD
    poles  = []

    for i in range(n):
        t = (i * POLE_PERIOD + POLE_SPACING / 2) / length
        poles.append(r1 + t * (r2 - r1))

    if rem >= POLE_SPACING:
        t = (n * POLE_PERIOD + POLE_SPACING / 2) / length
        poles.append(r1 + t * (r2 - r1))

    return poles


def get_solenoid_positions(p: int, q: int,
                           spacing_x: float,
                           spacing_z: float) -> list:
    """
    Returns 3D positions of all solenoids centered at origin on Y=0 plane.
    """
    positions = []
    x_offset  = (p - 1) * spacing_x / 2
    z_offset  = (q - 1) * spacing_z / 2
    for i in range(p):
        for j in range(q):
            positions.append(np.array([
                i * spacing_x - x_offset,
                0.0,
                j * spacing_z - z_offset
            ]))
    return positions


# ─────────────────────────────────────────────────────────────────────────────
# Core Physics
# ─────────────────────────────────────────────────────────────────────────────

def dipole_field_at(source_pos: np.ndarray,
                    moment: np.ndarray,
                    target_pos: np.ndarray) -> np.ndarray:
    """
    Magnetic field B at target_pos due to a magnetic dipole at source_pos.

    B = (mu_0 / 4pi) * [ 3(m·r_hat)r_hat - m ] / r^3
    """
    r_vec = target_pos - source_pos
    r     = np.linalg.norm(r_vec)
    if r < 1e-9:
        return np.zeros(3)
    r_hat = r_vec / r
    return (MU_0 / (4 * np.pi)) * (
        3 * np.dot(moment, r_hat) * r_hat - moment
    ) / r**3


def field_from_array(sol_positions: list,
                     solenoid_moment_vec: np.ndarray,
                     target_pos: np.ndarray) -> np.ndarray:
    """Total B field at target_pos from all solenoids."""
    B = np.zeros(3)
    for s in sol_positions:
        B += dipole_field_at(s, solenoid_moment_vec, target_pos)
    return B


def vertical_force_on_pole(sol_positions: list,
                            solenoid_moment_mag: float,
                            pole_pos: np.ndarray,
                            pole_moment_mag: float,
                            delta: float = 5e-4) -> float:
    """
    Vertical (Y) force on a single needle pole from the solenoid array.

    F_y = pole_moment * dB_y/dy

    Solenoid moments point along +Y to attract needle poles upward.
    Gradient estimated with central differences.
    """
    m_sol = solenoid_moment_mag * np.array([0.0, 1.0, 0.0])

    By_up   = field_from_array(
        sol_positions, m_sol, pole_pos + np.array([0,  delta, 0]))[1]
    By_down = field_from_array(
        sol_positions, m_sol, pole_pos + np.array([0, -delta, 0]))[1]

    dBy_dy = (By_up - By_down) / (2 * delta)
    return pole_moment_mag * dBy_dy


def total_vertical_force(r1: np.ndarray, r2: np.ndarray,
                          sol_positions: list,
                          solenoid_moment: float,
                          pole_moment: float) -> float:
    """
    Total upward magnetic force on the needle.
    Uses average height of all poles as the effective levitation height.
    """
    poles = get_pole_positions(r1, r2)
    return sum(
        vertical_force_on_pole(sol_positions, solenoid_moment, p, pole_moment)
        for p in poles
    )


# ─────────────────────────────────────────────────────────────────────────────
# Eddy Current Torque
# ─────────────────────────────────────────────────────────────────────────────

def eddy_current_torque(B_field: float, pwm_freq: float) -> dict:
    """
    Torque on the aluminum core due to eddy currents induced by
    a time-varying magnetic field at the given PWM frequency.

    Model:
      dB/dt  = B * 2 * pi * f          (sinusoidal approximation)
      EMF    = -dPhi/dt = -A * dB/dt
      I_eddy = EMF / R_eff
      Torque = I_eddy * A * B          (force on current loop in field)

    Where:
      A = CORE_AREA       cross-sectional area of aluminum core
      R = EDDY_RESISTANCE effective resistance of eddy current path
      B = solenoid field magnitude
      f = PWM frequency

    Returns torque (N·m) and angular acceleration (rad/s²).
    """
    dB_dt    = B_field * 2 * np.pi * pwm_freq   # T/s
    emf      = CORE_AREA * dB_dt                 # V  (Faraday's law)
    i_eddy   = emf / EDDY_RESISTANCE             # A
    torque   = i_eddy * CORE_AREA * B_field      # N·m
    alpha    = torque / MOMENT_OF_INERTIA        # rad/s²

    return {
        "pwm_freq_hz":   pwm_freq,
        "dB_dt_Tps":     dB_dt,
        "emf_V":         emf,
        "i_eddy_A":      i_eddy,
        "torque_Nm":     torque,
        "ang_accel_rps2": alpha,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Levitation Solver
# ─────────────────────────────────────────────────────────────────────────────

def min_B_to_levitate(r1: np.ndarray, r2: np.ndarray,
                       sol_positions: list,
                       pole_moment: float,
                       B_min: float = 1e-4,
                       B_max: float = 20.0,
                       tol: float = 1e-4) -> float | None:
    """
    Binary search for minimum solenoid moment magnitude required
    such that total vertical force >= needle weight.

    Returns None if not achievable within B_max.
    """
    weight = NEEDLE_MASS * G

    if total_vertical_force(r1, r2, sol_positions, B_max, pole_moment) < weight:
        return None

    lo, hi = B_min, B_max
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if total_vertical_force(r1, r2, sol_positions, mid, pole_moment) >= weight:
            hi = mid
        else:
            lo = mid
    return hi


def drift_in_half_second(r1: np.ndarray, r2: np.ndarray,
                          sol_positions: list,
                          solenoid_moment: float,
                          pole_moment: float) -> float:
    """
    Distance needle drifts in 0.5s if displaced upward by 0.5 inches
    while B is held constant. Quantifies Earnshaw instability.

    s = 0.5 * a * t²,  t = 0.5s
    """
    perturb  = 0.5 * INCH_TO_M
    pr1      = r1 + np.array([0, perturb, 0])
    pr2      = r2 + np.array([0, perturb, 0])
    F_net    = total_vertical_force(pr1, pr2, sol_positions,
                                     solenoid_moment, pole_moment) - NEEDLE_MASS * G
    return 0.5 * (F_net / NEEDLE_MASS) * (0.5 ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# Single Configuration Solver
# ─────────────────────────────────────────────────────────────────────────────

def solve(r1: np.ndarray, r2: np.ndarray,
          p: int, q: int,
          spacing: float,
          pole_moment: float,
          verbose: bool = True) -> dict:
    """
    Full solver for a single needle configuration and solenoid array.

    Parameters
    ----------
    r1, r2       : 3D endpoints of the needle (meters)
    p, q         : solenoid array dimensions
    spacing      : solenoid spacing in X and Z (meters)
    pole_moment  : magnetic moment per active pole (A·m²)
    verbose      : print results to console

    Returns
    -------
    Dictionary containing 6-DOF state, levitation result, and torque sweep.
    """
    sol_pos = get_solenoid_positions(p, q, spacing, spacing)
    dof     = compute_6dof(r1, r2)
    B_req   = min_B_to_levitate(r1, r2, sol_pos, pole_moment)

    torques = []
    if B_req is not None:
        drift = drift_in_half_second(r1, r2, sol_pos, B_req, pole_moment)
        for f in PWM_FREQUENCIES:
            torques.append(eddy_current_torque(B_req, f))
    else:
        drift = None

    if verbose:
        print(f"\n{'='*65}")
        print(f"  NEEDLE STATE — 6 Degrees of Freedom")
        print(f"{'='*65}")
        print(f"  r1 (m)         : ({r1[0]:.4f}, {r1[1]:.4f}, {r1[2]:.4f})")
        print(f"  r2 (m)         : ({r2[0]:.4f}, {r2[1]:.4f}, {r2[2]:.4f})")
        print(f"  Center (X,Y,Z) : ({dof['X_m']:.4f}, {dof['Y_m']:.4f}, {dof['Z_m']:.4f})")
        print(f"  Length         : {dof['length_m']*100:.2f} cm")
        print(f"  Yaw  (XY)      : {dof['yaw_deg']:.2f}°")
        print(f"  Pitch (YZ)     : {dof['pitch_deg']:.2f}°")
        print(f"  Roll  (ZX)     : {dof['roll_deg']:.2f}°  (convention: 0)")
        print(f"{'='*65}")
        print(f"  LEVITATION")
        print(f"{'='*65}")
        print(f"  Array          : {p}x{q}, spacing {spacing*100:.1f} cm")
        print(f"  Pole moment    : {pole_moment} A·m²")
        print(f"  Needle weight  : {NEEDLE_MASS*G:.6f} N")
        print(f"  Num poles      : {len(get_pole_positions(r1, r2))}")
        if B_req is not None:
            print(f"  Min B required : {B_req:.6f} T")
            print(f"  0.5s drift     : {drift:+.4f} m  (Earnshaw instability)")
        else:
            print(f"  Min B required : > 20.0 T  (not achievable)")
        print(f"{'='*65}")

        if torques:
            print(f"  EDDY CURRENT TORQUE vs PWM FREQUENCY  (at B = {B_req:.4f} T)")
            print(f"{'='*65}")
            print(f"  {'Freq (Hz)':<12} {'dB/dt (T/s)':<15} {'EMF (V)':<12} "
                  f"{'I_eddy (A)':<13} {'Torque (N·m)':<15} {'α (rad/s²)'}")
            print(f"  {'-'*12} {'-'*15} {'-'*12} {'-'*13} {'-'*15} {'-'*12}")
            for t in torques:
                print(f"  {t['pwm_freq_hz']:<12} {t['dB_dt_Tps']:<15.2f} "
                      f"{t['emf_V']:<12.6f} {t['i_eddy_A']:<13.6f} "
                      f"{t['torque_Nm']:<15.8f} {t['ang_accel_rps2']:.4f}")
        print(f"{'='*65}\n")

    return {
        "r1":         r1,
        "r2":         r2,
        "dof":        dof,
        "array":      f"{p}x{q}@{spacing*100:.1f}cm",
        "pole_moment": pole_moment,
        "B_req_T":    B_req,
        "drift_05s":  drift,
        "torques":    torques,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Parameter Sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(heights_in: list = None,
              save_csv: bool = True) -> list:
    """
    Sweeps solenoid array configuration, pole moment, and operating height.
    For each practical configuration (B < 2T), also sweeps PWM frequency
    for eddy current torque.

    Parameters
    ----------
    heights_in : list of needle heights above array in inches
    save_csv   : whether to save results to CSV

    Returns
    -------
    List of result dictionaries.
    """
    if heights_in is None:
        heights_in = [0.5, 1.0, 1.5, 2.0, 2.5]

    all_results   = []
    torque_results = []

    print(f"\n{'='*100}")
    print(f"  PARAMETER SWEEP  |  Needle: {NEEDLE_MASS*1000:.0f}g {NEEDLE_LENGTH*100:.0f}cm  "
          f"|  Drift: 0.5s after +0.5in perturbation")
    print(f"{'='*100}")
    print(f"  {'Array':<20} {'Pole m (A·m²)':<16} {'Height (in)':<13} "
          f"{'Min B (T)':<12} {'0.5s Drift (m)':<16} Status")
    print(f"  {'-'*20} {'-'*16} {'-'*13} {'-'*12} {'-'*16} ------")

    for (p, q, sp, label), pm, h_in in product(ARRAY_CONFIGS,
                                                 POLE_MOMENTS,
                                                 heights_in):
        h    = h_in * INCH_TO_M
        r1   = np.array([0.0,          h, 0.0])
        r2   = np.array([NEEDLE_LENGTH, h, 0.0])
        sol  = get_solenoid_positions(p, q, sp, sp)
        B    = min_B_to_levitate(r1, r2, sol, pm)

        if B is None:
            B_str     = "> 20.0"
            drift_str = "N/A"
            drift_val = None
            status    = "✗ Not achievable"
        else:
            drift_val = drift_in_half_second(r1, r2, sol, B, pm)
            B_str     = f"{B:.4f}"
            drift_str = f"{drift_val:+.4f}"
            status    = ("✓ PRACTICAL" if B < 1.0
                         else "△ MARGINAL" if B < 2.0
                         else "✗ TOO HIGH")

            # Torque sweep for every achievable configuration
            for f in PWM_FREQUENCIES:
                t = eddy_current_torque(B, f)
                torque_results.append({
                    "array":         label,
                    "pole_moment":   pm,
                    "height_in":     h_in,
                    "B_T":           B,
                    "pwm_freq_hz":   f,
                    "torque_Nm":     t["torque_Nm"],
                    "ang_accel_rps2": t["ang_accel_rps2"],
                    "status":        status,
                })

        print(f"  {label:<20} {pm:<16.2f} {h_in:<13.1f} "
              f"{B_str:<12} {drift_str:<16} {status}")

        all_results.append({
            "array_label":  label,
            "p": p, "q": q,
            "spacing_cm":   sp * 100,
            "pole_moment":  pm,
            "height_in":    h_in,
            "height_m":     h,
            "min_B_T":      B,
            "drift_05s_m":  drift_val,
            "status":       status,
        })

    print(f"{'='*100}\n")

    # ── Practical summary ─────────────────────────────────────────────────────
    practical = [r for r in all_results
                 if r["min_B_T"] is not None and r["min_B_T"] < 2.0]

    print("  PRACTICAL CONFIGURATIONS (B < 2.0 T):")
    print(f"  {'-'*85}")
    if practical:
        print(f"  {'Array':<20} {'Pole m':<10} {'Height (in)':<13} "
              f"{'Min B (T)':<12} {'0.5s Drift (m)'}")
        for r in practical:
            print(f"  {r['array_label']:<20} {r['pole_moment']:<10.2f} "
                  f"{r['height_in']:<13.1f} {r['min_B_T']:<12.4f} "
                  f"{r['drift_05s_m']:+.4f} m")
    else:
        print("  None found. Increase pole moment or array density.")
    print()

    # ── Torque summary (best config per frequency) ────────────────────────────
    if torque_results:
        print("  TORQUE AT PRACTICAL CONFIGS — Best per PWM Frequency:")
        print(f"  {'-'*70}")
        print(f"  {'Freq (Hz)':<12} {'Max Torque (N·m)':<20} {'Max α (rad/s²)':<18} Config")
        print(f"  {'-'*12} {'-'*20} {'-'*18} ------")
        for f in PWM_FREQUENCIES:
            freq_results = [t for t in torque_results
                            if t["pwm_freq_hz"] == f and t["status"] in
                            ("✓ PRACTICAL", "△ MARGINAL")]
            if freq_results:
                best = max(freq_results, key=lambda x: x["torque_Nm"])
                print(f"  {f:<12} {best['torque_Nm']:<20.8f} "
                      f"{best['ang_accel_rps2']:<18.4f} "
                      f"{best['array']} | m={best['pole_moment']:.2f} | "
                      f"h={best['height_in']:.1f}in")
        print()

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    if save_csv:
        lev_file = "levitation_sweep.csv"
        tor_file = "torque_sweep.csv"

        with open(lev_file, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_results[0].keys())
            w.writeheader(); w.writerows(all_results)

        if torque_results:
            with open(tor_file, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=torque_results[0].keys())
                w.writeheader(); w.writerows(torque_results)

        print(f"  Saved: {os.path.abspath(lev_file)}")
        print(f"  Saved: {os.path.abspath(tor_file)}\n")

    return all_results, torque_results


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Example: single configuration solve ──────────────────────────────────
    # Needle horizontal, 0.5 inches above array, tilted 15° in XY plane
    height  = 0.5 * INCH_TO_M
    tilt    = np.radians(15)
    half_l  = NEEDLE_LENGTH / 2

    r1 = np.array([-half_l * np.cos(tilt), height - half_l * np.sin(tilt), 0.0])
    r2 = np.array([ half_l * np.cos(tilt), height + half_l * np.sin(tilt), 0.0])

    solve(r1, r2,
          p=28, q=14, spacing=0.010,
          pole_moment=0.50,
          verbose=True)

    # ── Full parameter sweep ──────────────────────────────────────────────────
    run_sweep(heights_in=[0.5, 1.0, 1.5, 2.0, 2.5], save_csv=True)
