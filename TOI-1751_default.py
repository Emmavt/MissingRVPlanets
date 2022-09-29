import numpy as np
import radvel
import pandas as pd
import string
from matplotlib import rcParams

data = pd.read_csv('TestData/TOI-1751_st.csv')

t = np.array(data.time)
vel = np.array(data.mnvel)
errvel = np.array(data.errvel)
tel = np.array(data.tel)
telgrps = data.groupby('tel').groups
instnames = telgrps.keys()

starname = 'TOI-1751'
nplanets = 2
fitting_basis = 'per tc secosw sesinw k'
bjd0 = 0.
planet_letters = {1: 'b', 2: 'c'}
telescopes = np.unique(tel)

params = radvel.Parameters(2, basis='per tc e w k', planet_letters=planet_letters)
params['per1'] = radvel.Parameter(value = 37.468)
params['tc1'] = radvel.Parameter(value = 2458967.95)
params['e1'] = radvel.Parameter(value = 0, vary=True)
params['w1'] = radvel.Parameter(value = 0)
params['k1'] = radvel.Parameter(value = 3)

params['per2'] = radvel.Parameter(value = 400, vary=True)
params['tc2'] = radvel.Parameter(value = 2458967.95)
params['e2'] = radvel.Parameter(value = 0, vary=True)
params['w2'] = radvel.Parameter(value = 0)
params['k2'] = radvel.Parameter(value = 3)

params['dvdt'] = radvel.Parameter(value=0.0)
params['curv'] = radvel.Parameter(value=0.0)

for telescope in telescopes:
	params[f'gamma_{telescope}'] = radvel.Parameter(value=0.5, vary=True)
	params[f'jit_{telescope}'] = radvel.Parameter(value=3, vary=True)

params = params.basis.to_any_basis(params, fitting_basis)
time_base = 2458989.783463
mod = radvel.RVModel(params, time_base=time_base)
mod.params['per1'].vary = False
mod.params['tc1'].vary = False
mod.params['secosw1'].vary = True
mod.params['sesinw1'].vary = True

mod.params['per2'].vary = True
mod.params['tc2'].vary = False
mod.params['secosw2'].vary = True
mod.params['sesinw2'].vary = True

mod.params['dvdt'].vary = True
mod.params['curv'].vary = False

priors = [radvel.prior.PositiveKPrior(2)]
priors += [radvel.prior.EccentricityPrior(2, upperlims=[0.99, 0.99])]
for telescope in telescopes:
	priors += [radvel.prior.HardBounds(f'jit_{telescope}', -20.0, 20.0)]

stellar = dict(mstar=1.152, mstar_err=0.169)
