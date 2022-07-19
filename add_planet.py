"""
Script with ancillary functions to help generate additional planets
"""
import numpy as np
import pandas as pd
import astropy.units as u

# test comment
def log_space_periods(sys):
    """
    Calculate the periods intermediate to the known planets in the system in log-period space based on Kepler multi-planet statistics as described by Weiss et al. 2018

    Args:
        sys (System): Planet System object

    Returns:
        dict: Dictionary mapping a label to the period at which to add a planet

    Raises:
        AssertionError: Check that the planets are ordered with period

    """
    planets = sys.planets
    intermediate_periods = {}
    # For each consecutive pair of planets in the system, calculate the geometric mean of their periods
    for first_pl, second_pl in zip(planets, planets[1:]):
        p1, p2 = first_pl.period, second_pl.period
        assert p1 < p2, 'Check that the planets are in order of increasing period in the System object'
        # Geometric mean which is the same as even gaps in log-period space
        intermediate_p = np.sqrt(p1*p2)
        # Check we are not accidentally rerunning this command
        assert 'add' not in first_pl.letter and 'add' not in second_pl.letter, "You've gone too far and are finding additional planets in-between additional planets. Time to reset."
        # Save intermediate period to dictionary
        intermediate_periods['add_{}{}'.format(first_pl.letter,second_pl.letter)] = np.round(
            intermediate_p, 3)
    return intermediate_periods


def calc_semi_ampltiude(P, Mp, Ms, e):
    """
    Calculate the semi-amplitude of the radial velocity signal associated with a given planet as defined on https://exoplanetarchive.ipac.caltech.edu/docs/poet_calculations.html. Assume i = pi/2 i.e. the system is edge-on.

    Args:
        P (float): Planet orbital period in days
        Mp (float): Planet mass in Mearth
        Ms (float): Stellar mass in Msun
        e (float): Planet eccentricity

    Returns:
        float: Semi-ampltiude of radial velocity signal

    Raises:
        AssertionError: Planet eccentricity must be between 0 and 1

    """

    assert e >= 0 and e <= 1, "Eccentricity must be between 0 and 1."
    Mp = (Mp*u.earthMass).to(u.jupiterMass).value  # Convert Mp into Jupiter masses
    # Split up calculation into three steps for readability
    factor1 = 203*np.power(P, (-1/3))
    factor2 = Mp*np.sin(np.pi/2)/np.power(Ms+9.548E-4*Mp, (2/3))
    factor3 = 1/np.sqrt(1-np.power(e, 2))
    K = factor1*factor2*factor3
    return K
