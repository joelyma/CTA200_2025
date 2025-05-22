#!/usr/bin/env python
# coding: utf-8

# In[5]:
import numpy as np
from astropy import units as u
from poliastro.core.angles import M_to_E, E_to_nu
from poliastro.bodies import Sun
from poliastro.twobody import Orbit
from astropy.constants import G
import pandas as pd
from astropy.time import Time
from poliastro.bodies import Body
import emcee
from astropy.units import Quantity

def solve_true_anomaly(t, t_p, P, e):
    """
    Solves Kepler's Equation and computes the true anomaly.

    Parameters
    ----------
    t : float or Quantity
        Time of interest (e.g., 2025.0 * u.yr).
    t_p : float or Quantity
        Time of periapsis passage (same units as t).
    P : float or Quantity
        Orbital period (same units as t).
    e : float
        Orbital eccentricity.
    

    Returns
    -------
    nu : Quantity
        True anomaly in radians
    M : Quantity
        Mean anomaly in radians (wrapped to [0, 2π])
    E : Quantity
        Eccentric anomaly in radians
    """

    # Unit Check
    t = t * u.yr if not isinstance(t, u.Quantity) else t
    t_p = t_p * u.yr if not isinstance(t_p, u.Quantity) else t_p
    P = P * u.yr if not isinstance(P, u.Quantity) else P

    # Mean motion and mean anomaly
    n = (2 * np.pi * u.rad) / P
    M = n * (t - t_p)
    M_wrapped = M.to(u.rad).value % (2 * np.pi)

    # Solve for eccentric anomaly E and true anomaly ν
    E = M_to_E(M_wrapped, e)
    nu = E_to_nu(E, e)

    return nu * u.rad, M_wrapped * u.rad, E * u.rad


def ra_dec_to_xy(ra_arcsec, dec_arcsec, D):
    arcsec_to_rad = np.pi / (180 * 3600)  # convert arcsec to radians

    ra_rad = np.asarray(ra_arcsec) * arcsec_to_rad
    dec_rad = np.asarray(dec_arcsec) * arcsec_to_rad

    x = D * ra_rad
    y = D * dec_rad

    return x.to(u.AU), y.to(u.AU)

def cartesian_to_keplerian(x, y, z, vx, vy, vz, mu=Sun.k):
    """
    Converts Cartesian position and velocity vectors to Keplerian orbital elements.

    Parameters
    ----------
    x, y, z : float
        Position vector components in AU.
    vx, vy, vz : float
        Velocity vector components in km/s.
    mu : astropy.units.Quantity, optional
        Gravitational parameter (defaults to Sun.k).
    return_wrapped : bool
        If True, wraps ν to [0, 360°] for output.

    Returns
    -------
    dict
        Orbital elements (with astropy units):
        - a: semi-major axis [AU]
        - e: eccentricity [dimensionless]
        - i: inclination [deg]
        - raan: longitude of ascending node Ω [deg]
        - argp: argument of periapsis ω [deg]
        - nu: true anomaly ν [deg]
    """
    # Convert to Quantity vectors
    r = np.array([x, y, z]) * u.AU
    v = np.array([vx, vy, vz]) * u.km / u.s

    # Generate Orbit
    orbit = Orbit.from_vectors(Sun, r, v)

    # Wrap true anomaly to [0, 360)
    nu = orbit.nu.to(u.deg) % (360 * u.deg)


    return {
        "a": orbit.a.to(u.AU),
        "e": orbit.ecc,
        "i": orbit.inc.to(u.deg),
        "raan": orbit.raan.to(u.deg),
        "argp": orbit.argp.to(u.deg),
        "nu": nu
    }


def forward_model(times, M, D, a, e, inc, raan, argp, tp):
    """
    Predicts x, y, vz given orbital parameters and observation times.

    Parameters
    ----------
    times : list of datetime or Astropy Time
    M : float
        Mass of Sgr A* [M_sun]
    D : float
        Distance to Sgr A* [pc]
    a : float
        Semi-major axis [AU]
    ecc : float
        Eccentricity
    inc, raan, argp : float
        Orbital angles [deg]
    tp : float
        Time of periapsis passage [Julian years]

    Returns
    -------
    x_proj, y_proj, vz : arrays
        Projected positions [AU] and radial velocity [km/s]
    
    M = M * u.Msun
    D = D * u.pc
    a = a
    inc = inc * u.deg
    raan = raan * u.deg
    argp = argp * u.deg

    # Define central body as a new object with custom mass
    BH = Body(name="SgrA*", parent=None, k=G * M)

    # Convert times
    t0 = Time(tp, format='jyear')  # time of periapsis
    obs_times = Time(times)

    # Create the orbit
    ecc = ecc * u.onec
    orbit0 = Orbit.from_classical(BH, a, ecc, inc, raan, argp, 0 * u.deg, epoch=t0)

    # Propagate orbit to observation times
    orbits = [orbit0.propagate(t - t0) for t in obs_times]

    # Extract positions and velocities in AU and km/s
    r_vectors = np.array([orb.r.to(u.AU).value for orb in orbits])
    v_vectors = np.array([orb.v.to(u.km/u.s).value for orb in orbits])

    # Projection: assume sky-plane is x-y, radial is z
    x_proj = r_vectors[:, 0]
    y_proj = r_vectors[:, 1]
    vz = v_vectors[:, 2]

    return x_proj, y_proj, vz
"""
    M = M * u.Msun
    D = D * u.pc
    a = a * u.AU     # Step 1: attach units
    inc = inc * u.deg
    raan = raan * u.deg
    argp = argp * u.deg
    tp = Time(tp, format='jyear')  # time of periapsis

    # Define central mass (Sgr A*) with given mass
    BH = Body(name="SgrA*", parent=None, k=G * M)

    # Create orbit from classical elements
    ecc = e * u.one
    orbit0 = Orbit.from_classical(BH, a, ecc, inc, raan, argp, 0 * u.deg, epoch=tp)

    # Propagate orbit to each observation time
    obs_times = Time(times)
    orbits = [orbit0.propagate(t - tp) for t in obs_times]

    # Extract positions and velocities (convert to unitless arrays)
    r_vectors = np.array([orb.r.to(u.AU).value for orb in orbits])
    v_vectors = np.array([orb.v.to(u.km/u.s).value for orb in orbits])

    # Project to observed coordinates
    x_proj = r_vectors[:, 0]
    y_proj = r_vectors[:, 1]
    vz = v_vectors[:, 2]

    return x_proj, y_proj, vz


'''def log_prior(theta):
    M, D, a, e, inc, raan, argp, tp = theta

    # Broad but physical bounds
    if not (1e5 < M < 1e8): return -np.inf
    if not (5000 < D < 15000): return -np.inf
    if not (100 < a < 5000): return -np.inf
    if not (0.0 < e < 0.999): return -np.inf
    if not (0 < inc < 180): return -np.inf
    if not (0 < raan < 360): return -np.inf
    if not (0 < argp < 360): return -np.inf
    if not (1995 < tp < 2025): return -np.inf

    return 0.0  # uniform prior within bounds
'''

def log_prior(theta):
    M, D, a, e, inc, raan, argp, tp = theta

    if not (1e5 < M < 1e8):
        print("Rejected M:", M)
        return -np.inf
    if not (5000 < D < 15000):
        print("Rejected D:", D)
        return -np.inf
    if not (100 < a < 5000):
        print("Rejected a:", a)
        return -np.inf
    if not (0.0 < e < 0.999):
        print("Rejected e:", e)
        return -np.inf
    if not (0 < inc < 180):
        print("Rejected inc:", inc)
        return -np.inf
    if not (0 < raan < 360):
        print("Rejected raan:", raan)
        return -np.inf
    if not (0 < argp < 360):
        print("Rejected argp:", argp)
        return -np.inf
    if not (1995 < tp < 2025):
        print("Rejected tp:", tp)
        return -np.inf

    return 0.0



def log_likelihood(theta, times, x_obs, y_obs, vz_obs, D_for_error):
    M, D, a, e, inc, raan, argp, tp = theta

    try:

        #x_obs = x_obs.values * u.AU
        # Predict model orbit
       # print(a, 'a before')
        x_model, y_model, vz_model = forward_model(
            times, M, D, a, e, inc, raan, argp, tp
        )
       # print('xm_0', x_model[0], 'ym_0', y_model[0], 'vm_0', vz_model[0])
       # print('xo_0',x_obs[0], 'yo_0',y_obs[0], 'vo_0',vz_obs[0])


        #from astropy.units import Quantity
        #x_model = x_model * u.AU
        #y_model = y_model * u.AU
        #vz_model = vz_model * u.AU

        #if isinstance(x_model, Quantity):
        #    print("x_model is a Quantity with units:", x_model.unit)
        #else:
        #    print("x_model is just a plain array or number.")
        #print(type(x_obs))
        #print(x_obs[:3])
        #if isinstance(x_obs, Quantity):
        #    print("x_obs is a Quantity with units:", x_model.unit)
        #else:
        #    print("x_obs is just a plain array or number.")
        
        #print(a, 'after')
        # Convert position error: 0.01 arcsec → radians → AU
        arcsec_to_rad = np.pi / (180 * 3600)
        angular_err_rad = 0.5 * arcsec_to_rad
        pos_err_au = (D_for_error * angular_err_rad)

        # Fixed radial velocity error from file
        vz_err = 10.0  # km/s

        # Compute chi-squared
        chi2_x = np.sum(((x_obs - x_model) / pos_err_au) ** 2)
        chi2_y = np.sum(((y_obs - y_model) / pos_err_au) ** 2)
        chi2_vz = np.sum(((vz_obs - vz_model) / vz_err) ** 2)

        # Log-likelihood: Gaussian form
        logL = -0.5 * (chi2_x + chi2_y + chi2_vz)
        #print("Success: chi-squared computed")
        return logL

    except Exception as e:
        print("Error in likelihood:", e)
        import traceback; traceback.print_exc()
        return -np.inf


def log_probability(theta, times, x_obs, y_obs, vz_obs, D_for_error):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, times, x_obs, y_obs, vz_obs, D_for_error)

