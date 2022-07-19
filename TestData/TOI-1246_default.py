import numpy as np
import radvel
import pandas as pd
import string
from matplotlib import rcParams

data = pd.read_csv('TestData/TOI-1246_st.csv')

t = np.array(data.time)
vel = np.array(data.mnvel)
errvel = np.array(data.errvel)
tel = np.array(data.tel)
telgrps = data.groupby('tel').groups
instnames = telgrps.keys()

starname = 'TOI-1246'
nplanets = 5
fitting_basis = 'per tc secosw sesinw k'
bjd0 = 0.
planet_letters = {1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f'}
telescopes = np.unique(tel)

params = radvel.Parameters(5, basis='per tc e w k', planet_letters=planet_letters)
params['per1'] = radvel.Parameter(value = 4.31)
params['tc1'] = radvel.Parameter(value = 2458686.5658)
params['e1'] = radvel.Parameter(value = 0, vary=False)
params['w1'] = radvel.Parameter(value = 0)
params['k1'] = radvel.Parameter(value = 3)

params['per2'] = radvel.Parameter(value = 5.9)
params['tc2'] = radvel.Parameter(value = 2458683.4661)
params['e2'] = radvel.Parameter(value = 0, vary=False)
params['w2'] = radvel.Parameter(value = 0)
params['k2'] = radvel.Parameter(value = 3)

params['per3'] = radvel.Parameter(value = 18.66)
params['tc3'] = radvel.Parameter(value = 2458688.9653)
params['e3'] = radvel.Parameter(value = 0, vary=False)
params['w3'] = radvel.Parameter(value = 0)
params['k3'] = radvel.Parameter(value = 3)

params['per4'] = radvel.Parameter(value = 37.92)
params['tc4'] = radvel.Parameter(value = 2457000.7134)
params['e4'] = radvel.Parameter(value = 0, vary=False)
params['w4'] = radvel.Parameter(value = 0)
params['k4'] = radvel.Parameter(value = 3)

params['per5'] = radvel.Parameter(value = 93.8)
params['tc5'] = radvel.Parameter(value = 2459462.9)
params['e5'] = radvel.Parameter(value = 0, vary=False)
params['w5'] = radvel.Parameter(value = 0)
params['k5'] = radvel.Parameter(value = 3)

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
mod.params['secosw1'].vary = False
mod.params['sesinw1'].vary = False

mod.params['per2'].vary = False
mod.params['tc2'].vary = False
mod.params['secosw2'].vary = False
mod.params['sesinw2'].vary = False

mod.params['per3'].vary = False
mod.params['tc3'].vary = False
mod.params['secosw3'].vary = False
mod.params['sesinw3'].vary = False

mod.params['per4'].vary = False
mod.params['tc4'].vary = False
mod.params['secosw4'].vary = False
mod.params['sesinw4'].vary = False

mod.params['per5'].vary = False
mod.params['tc5'].vary = False
mod.params['secosw5'].vary = False
mod.params['sesinw5'].vary = False

mod.params['dvdt'].vary = True
mod.params['curv'].vary = False

priors = [radvel.prior.PositiveKPrior(5)]
priors += [radvel.prior.Gaussian('per1', 4.31, 2e-05)]
priors += [radvel.prior.Gaussian('tc1', 2458686.5658, 0.001)]
priors += [radvel.prior.Gaussian('per2', 5.9, 8e-05)]
priors += [radvel.prior.Gaussian('tc2', 2458683.4661, 0.003)]
priors += [radvel.prior.Gaussian('per3', 18.66, 5e-05)]
priors += [radvel.prior.Gaussian('tc3', 2458688.9653, 0.009)]
priors += [radvel.prior.Gaussian('per4', 37.92, 0.0001)]
priors += [radvel.prior.Gaussian('tc4', 2457000.7134, 0.0089)]
priors += [radvel.prior.Gaussian('per5', 93.8, 0.0001)]
priors += [radvel.prior.Gaussian('tc5', 2459462.9, 0.0089)]
for telescope in telescopes:
	priors += [radvel.prior.HardBounds(f'jit_{telescope}', -20.0, 20.0)]

stellar = dict(mstar=0.86, mstar_err=0.12)
