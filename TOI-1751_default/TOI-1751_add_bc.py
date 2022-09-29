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

starname = 'TOI-1751_add_bc'
nplanets = 3
fitting_basis = 'per tc secosw sesinw k'
bjd0 = 0.
planet_letters = {1: 'b', 2: 'c', 3: 'add_bc'}
telescopes = np.unique(tel)

params = radvel.Parameters(3, basis='per tc e w k', planet_letters=planet_letters)
params['per1'] = radvel.Parameter(value = 37.468, vary=True)
params['tc1'] = radvel.Parameter(value = 2459707.815)
params['e1'] = radvel.Parameter(value = 0, vary=True)
params['w1'] = radvel.Parameter(value = 0)
params['k1'] = radvel.Parameter(value = 3)

params['per2'] = radvel.Parameter(value = 400, vary=True)
params['tc2'] = radvel.Parameter(value = 2000000)
params['e2'] = radvel.Parameter(value = 0, vary=True)
params['w2'] = radvel.Parameter(value = 0)
params['k2'] = radvel.Parameter(value = 3)

params['per3'] = radvel.Parameter(value = 122.422, vary=True)
params['tc3'] = radvel.Parameter(value = 2346526.959843159)
params['e3'] = radvel.Parameter(value = 0.02647701226187682, vary=True)
params['w3'] = radvel.Parameter(value = 0)
params['k3'] = radvel.Parameter(value = 1.229032090407812)

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

mod.params['per2'].vary = False
mod.params['tc2'].vary = False
mod.params['secosw2'].vary = True
mod.params['sesinw2'].vary = True

mod.params['per3'].vary = False
mod.params['tc3'].vary = False
mod.params['secosw3'].vary = True
mod.params['sesinw3'].vary = True

mod.params['dvdt'].vary = True
mod.params['curv'].vary = False

priors = [radvel.prior.PositiveKPrior(3)]
priors += [radvel.prior.Gaussian('per1', 37.468, 0.00037)]
priors += [radvel.prior.Gaussian('tc1', 2459707.815, 0.00457)]
priors += [radvel.prior.Gaussian('per2', 400, 0.0004)]
priors += [radvel.prior.Gaussian('tc2', 2000000, 0.005)]
priors += [radvel.prior.EccentricityPrior(3, upperlims=[0.99, 0.98, 0.99])]
for telescope in telescopes:
	priors += [radvel.prior.HardBounds(f'jit_{telescope}', -20.0, 20.0)]

stellar = dict(mstar=1.152, mstar_err=0.169)