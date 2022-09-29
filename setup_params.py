import numpy as np
from dataclasses import dataclass
import radvel
from add_planet import log_space_periods, calc_semi_ampltiude
import pandas as pd
import string
import copy
import subprocess
import pickle
import textwrap
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.ticker as plticker
import scipy.stats as stats
#%matplotlib inline


class HostStar(object):
    """
    This class represents the host star in each planetary system
    """

    def __init__(self, mass, masserr, radius, radiuserr, teff, tefferr):
        """
        Initialize HostStar object

        Args:
            mass (float): Mstar in units of Msun
            masserr (float): Uncertainty in Mstar in units of Msun
            radius (float): Rstar in units of rsun
            radiuserr (float): Uncertainty in Rstar in units of Rsun
            teff (int): Effective temperature in Kelvin
            tefferr (int): Uncertainty in Teff in Kelvin
        """

        self.mass = mass
        self.masserr = masserr
        self.radius = radius
        self.radiuserr = radiuserr
        self.teff = teff
        self.tefferr = tefferr

    def __repr__(self):
        """
        Creates representation of host star with mass, and radius for readability

        Returns:
            str : Host star mass and radius
        """

        return "(Star: {} Msun, {} Rsun)".format(self.mass, self.radius)


class Planet(object):
    """
    This class represents a planet in a given system.

    """

    def __init__(self, letter, P, t0, K, ecc, omega=0, Perr=0, t0err=0, mass=0, masserr=0, radius=0, radiuserr=0):
        """
        Initialize Planet object

        Args:
            letter (str): Planet letter, typically but not always in order of semi-major axis in the system
            mass (float): Planet mass in units of Mearth
            masserr (float): Planet mass uncertainty in units of Mearth
            radius (float): Planet radius in units of Rearth
            radiuserr (float): Planet radius uncertainty in units of Rearth
            period (float): Planet orbital period in units of days (assuming linear emphemeris)
            perioderr (float): Planet orbital period uncertainty in units of days
            t0 (float): Planet transit epoch aka time of inferior conjunction in units of Julian Days (JD)
            t0err (float): Uncertainty in t0 in units of JD
            ecc (float): Planet eccentricity, between 0 and 1
            omega (float): Planet argument of periastron in units of radians
            K (float): Planet radial velocity semi-ampltiude in units of m/s

        """

        self.letter = letter
        self.mass = mass
        self.masserr = masserr
        self.radius = radius
        self.radiuserr = radiuserr
        self.period = P
        self.perioderr = Perr
        self.t0 = t0
        self.t0err = t0err
        self.ecc = ecc
        self.omega = omega
        self.K = K

    def __repr__(self):
        """
        Creates representation of planet with it's letter, mass, and radius for readability

        Returns:
            str : Planet letter, mass, and radius
        """
        return '(Planet {}: {} Me, {} Re)'.format(self.letter, self.mass, self.radius)


@dataclass(init=True)  # init=True generates standardized __init__ method
class System:
    name: str  # Name of host star
    num_planets: int  # Number of planets in the system
    star: HostStar  # Host Star object
    planets: list  # List of Planet objects

    def __post_init__(self):
        """
        Check that the object is instantialized self-consistently

        Raises:
            AssertionError: If the system is not self-consistent, or has fewer than 2 planets

        """

        # For this science case, we only want to consider systems with 2+ planets
        assert self.num_planets >= 2, "Make sure you have at least 2 planets!"
        assert len(self.planets) == self.num_planets, "Make sure you've added all of the planets!"
        
    
    def make_setup_file(self, data_file, setup_file):
        """
        This function generates a setup file that can be read into and used by the Radvel package to perform an RV fit

        Args:
            data_file (str): Name of .csv file that contains the RV data
            setup_file (str): Name to save the setup file under

        """
        def ecc_helper(self):
            pers = []
            for i, planet in enumerate(self.planets):
                pers.append(planet.period)
            pers1 = sorted(pers)
            eccs = []
            for i in range(len(pers1)-1):
                rat = (pers1[i+1]/pers1[i])**(2/3)
                if rat > 2:
                    eccs.append(0.99)
                else:
                    eccs.append(rat-1)
           # eccs.append(0.98)
           # eccs1 = []
            #for i in pers:
                #a = pers1.index(i)
                #eccs1.append(eccs[a])
            num = len(eccs)
            return [num,eccs]
        list_imports = ['import numpy as np', 'import radvel',
                        'import pandas as pd', 'import string', 'from matplotlib import rcParams']
        with open(setup_file, 'w') as file:
            # Import packages
            for imp in list_imports:
                file.write(imp+'\n')
            # Read in RV data, errors, time stamps, and telescope names into different arrays
            file.write("\ndata = pd.read_csv('{}')\n\n".format(data_file))
            file.write("t = np.array(data.time)\nvel = np.array(data.mnvel)\nerrvel = np.array(data.errvel)\ntel = np.array(data.tel)\ntelgrps = data.groupby('tel').groups\ninstnames = telgrps.keys()\n\n")

            # Define system parameters including name, number of planets, and what basis we want to fit the data in
            file.write(f"starname = '{self.name}'\nnplanets = {self.num_planets}\nfitting_basis = 'per tc secosw sesinw k'\nbjd0 = 0.\n")

            # Generate a dictionary that maps the planet letter to a number as per Radvel formatting
            planet_letters = [pl.letter for pl in self.planets]
            planet_nums = [int(i) for i in np.arange(self.num_planets)+1]
            planet_letters = dict(zip(planet_nums, planet_letters))
            # List of telescopes used to collect data
            file.write(f"planet_letters = {planet_letters}\ntelescopes = np.unique(tel)\n\n")
            # Initialize fitting parameters
            file.write(
                f"params = radvel.Parameters({self.num_planets}, basis='per tc e w k', planet_letters=planet_letters)\n")
            # For each planet, initialize a period, time of inferior conjuction, eccentricity, argument of periastron, and RV semi-ampltiude using the values read in to the script
            for i, planet in enumerate(self.planets):
                i += 1
                file.write(f"params['per{i}'] = radvel.Parameter(value = {planet.period}, vary=True)\n")
                file.write(f"params['tc{i}'] = radvel.Parameter(value = {planet.t0})\n")
                # For simplicity, and motivated by Yee et al. 2021, set eccentricities to 0
                file.write(f"params['e{i}'] = radvel.Parameter(value = {planet.ecc}, vary=True)\n")
                file.write(f"params['w{i}'] = radvel.Parameter(value = {planet.omega})\n")
                file.write(f"params['k{i}'] = radvel.Parameter(value = {planet.K})\n\n")

            # Initialize parameters describing the global RV slope and curvature
            file.write("params['dvdt'] = radvel.Parameter(value=0.0)\n")
            file.write("params['curv'] = radvel.Parameter(value=0.0)\n\n")
            # For each telescope used, initialize an offset and a jitter instrumental term and set initial non-zero guesses
            file.write(f"for telescope in telescopes:\n")
            file.write("\tparams[f'gamma_{telescope}'] = radvel.Parameter(value=0.5, vary=True)\n")
            file.write("\tparams[f'jit_{telescope}'] = radvel.Parameter(value=3, vary=True)\n\n")
            # Transform the parameter basis to the fitting basis parameterisation (to simplify initialization)
            file.write("params = params.basis.to_any_basis(params, fitting_basis)\n")
            # Time which dvdt and curv are calculated relative to
            file.write("time_base = 2458989.783463\n")
            # Create RV model
            file.write("mod = radvel.RVModel(params, time_base=time_base)\n")

            # For each planet parameter, set whether it is allowed to vary in the fitting process
            for i, planet in enumerate(self.planets):
                i += 1
                file.write(f"mod.params['per{i}'].vary = False\n")
                file.write(f"mod.params['tc{i}'].vary = False\n")
                file.write(f"mod.params['secosw{i}'].vary = True\n")
                file.write(f"mod.params['sesinw{i}'].vary = True\n\n")

            # The same for the global RV parameters
            file.write("mod.params['dvdt'].vary = True\nmod.params['curv'].vary = False\n\n")
            # Set a prior to keep K > 0 - this can be inappropriate as it biases larger values of K
            file.write(f"priors = [radvel.prior.PositiveKPrior({self.num_planets})]\n")
            # Set priors on the planet parameters based on prior knowledge of uncertainties
            #for i, planet in enumerate(self.planets):
                #i += 1
                # Do not set a prior on period and t0 if there is no associated error as this breaks the fit
                #if planet.perioderr != 0 and planet.t0err != 0:
                    #file.write(
                        #f"priors += [radvel.prior.Gaussian('per{i}', {planet.period}, {planet.perioderr})]\npriors += [radvel.prior.Gaussian('tc{i}', {planet.t0}, {planet.t0err})]\n")
            file.write(f"priors += [radvel.prior.EccentricityPrior({ecc_helper(self)[0]}, upperlims={ecc_helper(self)[1]})]\n")
            # Set a hard bound prior on instrumental parameters to speed up the fit
            file.write(f"for telescope in telescopes:\n")
            file.write("\tpriors += [radvel.prior.HardBounds(f'jit_{telescope}', -20.0, 20.0)]\n\n")
            # Add stellar parameters
            file.write(f"stellar = dict(mstar={self.star.mass}, mstar_err={self.star.masserr})")


# Read in all systems' data
data = pd.read_csv('planets_data.csv')


def eval_missing_planets(row):
    """
    Read in planetary system information for individual planets and stars
    Create variations of the systems with additional planets at various periods
    Fit the radial velocity data assuming various system parameters
    Compare the resulting masses for each of the known planets across the system variations

    Args:
        row (Pandas Series): row of a dataframe including all of the stellar host and planet parameters

    """
    # Create system object with host star and planets for the 'default' scenario
    #star = HostStar(row['ms'], row['mserr'], row['rs'], row['rserr'], row['teff'], row['tefferr'])
    #num_planets = row['npl']
    #planets = []
    #for i in range(1, num_planets+1):
        # Add planets
        #pl = Planet(string.ascii_lowercase[i], row[f'p{i}'], row[f't0{i}'],
                    #row[f'K{i}'], mass=row[f'mp{i}'], masserr=row[f'mperr{i}'], radius=row[f'rp{i}'], radiuserr=row[f'rperr{i}'], Perr=row[f'perr{i}'], t0err=row[f't0err{i}'], ecc=row[f'e{i}'], omega=row[f'w{i}'])
        #planets.append(pl)
    #sys = System(row['name'], num_planets, star, planets)

    # Create system object with host star and planets for the 'default' scenario
    #name, period, t0, K,ecc,omega,Perr,t0err, mass, masserr, radius, radiuserr
    #star = HostStar(0.86, 0.12, 0.87, 0.1, 5151, 100)
    #pl1 = Planet('b', 4.31, 2458686.5658, 3, 0, 0, 0.00002, 0.001, 8.1, 1.1, 3.01, 0.06)
    #pl2 = Planet('c', 5.90, 2458683.4661, 3, 0, 0, 0.00008, 0.003, 8.8, 1.2, 2.51, 0.08)
    #pl3 = Planet('d', 18.66, 2458688.9653, 3, 0, 0, 0.00005, 0.009, 5.3, 1.7, 3.51, 0.09)
    #pl4 = Planet('e', 37.92, 2457000.7134, 3, 0, 0, 0.0001, 0.0089, 14.8, 2.3, 3.78, 0.16)
    #pl5 = Planet('f', 93.8, 2459462.9, 3, 0, 0, 0.0001, 0.0089, 26.6, 3.8, 0, 0,)
    #sys = System('TOI-1246', 5, star, [pl1, pl2, pl3, pl4, pl5])
    #new system
    #mass, masserr, radius, radiuserr, teff, tefferr
    #name, period, t0, K,ecc,omega,Perr,t0err, mass, masserr, radius, radiuserr
    #star = HostStar(0.81, 0.03, 0.832, 0.02, 5326, 64)
    #pl1 = Planet('b', 0.45, 2458517.4973, 3, 0, 0, 0.000032, 0.0018, 3.2, 0.8, 1.45, 0.11)
    #pl2 = Planet('c', 10.78, 2458527.05825, 3, 0, 0, 0.00015, 0.00053, 7.0, 2.3, 2.90, 0.13)
    #pl3 = Planet('d', 16.29, 2458521.8828, 3, 0, 0, 0.00005, 0.0035, 3.0, 2.4, 2.32, 0.16)
    #pl4 = Planet('e', 77.23, 2457000.7134, 3, 0, 0, 0.0001, 0.0089, 14.8, 2.3, 3.78, 0.16)
    #pl5 = Planet('f', 93.8, 2459462.9, 3, 0, 0, 0.0001, 0.0089, 26.6, 3.8, 0, 0,)
    #sys = System('TOI-561', 3, star, [pl1, pl2, pl3])
    # Generate the setup file for the default system
    ##changed setup file to TestData/
    #sys.make_setup_file(f'TestData/{sys.name}_st.csv', f"{sys.name}_default.py")

    #name, period, t0, K,ecc,omega,Perr,t0err, mass, masserr, radius, radiuserr
    star = HostStar(1.152, 0.169, 1.268, 0.055, 6114, 122.03)
    pl1 = Planet('b', 37.468, 2458967.95, 5, 0, 0, 0.00037, 0.00457, 9, 2, 2.752, 0.342)
    pl2 = Planet('c', 400, 2458967.95, 5, 0, 0, 0.0004, 0.005, 12, 3, 5, 0.2)
    sys = System('TOI-1751', 2, star, [pl1, pl2])
    sys.make_setup_file(f'TestData/{sys.name}_st.csv', f"{sys.name}_default.py")
    # Run a Radvel fit on the default system setup file
    subprocess.run(["TestData/radvel_bash.sh", f"{sys.name}_default.py", "nplanets"])

    # Calculate periods where additional planets may likely hide based on Kepler multi-planet statistics
    ##periods_to_add = log_space_periods(sys)
    # Dictionary to keep track of different system variations
    ##sys_varieties = {'default': sys}
    ##for key, P in periods_to_add.items():
        ##t0s = np.array([pl.t0 for pl in sys.planets])
        ##masses = np.array([pl.mass for pl in sys.planets])
        # Pick a random t0 within the range set by the other planets in the system for the new planet
        ##t0_add = np.random.uniform(t0s.min(), t0s.max())
        # Use the mean of the other planets in the system as the initial mass of the additional planet (Millholland et al. 2017)
        ##M_add = np.mean(masses)
        # Calculate the associated semi-amplitude of the new planet
        ##K_add = calc_semi_ampltiude(P, M_add, sys.star.mass, 0)
#add
        #ecc_add = pick random between 0 and 1? start with 0?
        ##def ecc():
            ##eccs = np.random.rayleigh(0.0355, 1)
            ##while eccs != 0:
                ##return eccs[0]
        ##ecc_add = ecc()
        ##planet_to_add = Planet(key, P, t0_add, K_add, mass=M_add, ecc = ecc_add)
        # Make a copy of the default system to add the new planet to
        ##sys_add_pl = copy.deepcopy(sys)
        ##sys_add_pl.name = sys.name+'_'+key  # Rename new system variation
        ##sys_add_pl.planets += [planet_to_add]  # Add new planet
        ##sys_add_pl.num_planets += 1
        ##sys_varieties[key] = sys_add_pl  # Store new system variation

        # Pickle system object to save it for reproducability
        ##sys_add_pl.__module__ = __name__  # Needed to pickle a dataclass
        ##with open(f'pickle_{sys_add_pl.name}', 'wb') as pickle_file:
            ##pickle.dump(sys_add_pl, pickle_file)

        # Create Radvel setup file
        ##setup_file_name = f'{sys.name}_default/{sys_add_pl.name}.py' 
        ##sys_add_pl.make_setup_file('TestData/TOI-1246_st.csv', setup_file_name)
        # Run radvel fit
        ##subprocess.run(["TestData/radvel_bash.sh", setup_file_name, "nplanets"])

        # Read in results from radvel fit
        ##derived_params = pd.read_csv(
            ##f'{sys_add_pl.name}/{sys_add_pl.name}_derived.csv.bz2', index_col=0)
        # Create dictionary mapping planet letters to numbers
        ##planet_letters = dict(zip([int(i) for i in np.arange(
            ##sys_add_pl.num_planets)+1], [pl.letter for pl in sys_add_pl.planets]))
        # Calculate and update planet masses and associated errors from the fit
        ##for i, planet in enumerate(sys_add_pl.planets):
            ##planet.mass = np.mean(derived_params[f'mpsini{i+1}'])
            ##planet.masserr = np.std(derived_params[f'mpsini{i+1}'])

    # Create a figure comparing the results from different system variations
    ##fig, ax = plt.subplots()
    ##colours = iter(cm.viridis(np.linspace(0, 1, len(sys_varieties.keys()))))
    ##for key, system in sys_varieties.items():
        ##c = next(colours)
        ##for i, pl in enumerate(system.planets):
            ##scatter = ax.errorbar(pl.period, pl.mass, yerr=pl.masserr, fmt='o', c=c,
                                  ##label=key if i == 0 else "", ecolor='lightgray', ms=5)
    ##ax.vlines(x=[0.45, 10.78, 16.29], ymin=0, ymax=40, linestyle='--', alpha=0.5)
    ##ax.set_xlabel('Orbital Period (d)', fontsize=16)
    ##ax.set_ylabel('Planet Mass ($M_\oplus$)', fontsize=16)
    ##ax.tick_params(axis='x', labelsize=16)
    ##ax.tick_params(axis='y', labelsize=16)
    ##ax.xaxis.set_tick_params(size=10)
    ##ax.yaxis.set_tick_params(size=10)
    ##ax.set_xscale('log')
    ##ax.set_xlim(3, 100)
    ##ax.set_ylim(-1, 35)
    ##ax.xaxis.set_major_formatter(plticker.ScalarFormatter())
    ##ax.set_xticks([3, 5, 10, 30, 50, 100])
    ##ax.legend(loc='upper left', fontsize=14)
    ##plt.tight_layout()
    ##plt.savefig(f"{sys.name}_default/Mass_comp_{sys_varieties['default'].name}.png", dpi=300)

   
#plotting mass difference
    #fig, ax = plt.subplots()
    #colours = iter(cm.viridis(np.linspace(0, 1, len(sys_varieties.keys()))))
    #for key, system in sys_varieties.items():
        #c = next(colours)
        #for i, pl in enumerate(system.planets):
            #scatter = ax.errorbar(pl.period, pl.mass, yerr=pl.masserr, fmt='o', c=c,
                                  #label=key if i == 0 else "", ecolor='lightgray', ms=5)
    
    
eval_missing_planets(1)

# Use multiprocessing to run several systems at once
#pool = Pool(processes=4)
#pool.map(eval_missing_planets, data)
#del pool
