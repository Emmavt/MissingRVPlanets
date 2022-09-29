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

starname = 'TOI-561_add_bc'
nplanets = 4
fitting_basis = 'per tc secosw sesinw k'
bjd0 = 0.
planet_letters = {1: 'b', 2: 'c', 3: 'd', 4: 'add_bc'}
telescopes = np.unique(tel)

params = radvel.Parameters(4, basis='per tc e w k', planet_letters=planet_letters)
params['per1'] = radvel.Parameter(value = 0.45)
params['tc1'] = radvel.Parameter(value = 2458517.4973)
params['e1'] = radvel.Parameter(value = 0, vary=True)
params['w1'] = radvel.Parameter(value = 0)
params['k1'] = radvel.Parameter(value = 3)

params['per2'] = radvel.Parameter(value = 10.78)
params['tc2'] = radvel.Parameter(value = 2458527.05825)
params['e2'] = radvel.Parameter(value = 0, vary=True)
params['w2'] = radvel.Parameter(value = 0)
params['k2'] = radvel.Parameter(value = 3)

params['per3'] = radvel.Parameter(value = 16.29)
params['tc3'] = radvel.Parameter(value = 2458521.8828)
params['e3'] = radvel.Parameter(value = 0, vary=True)
params['w3'] = radvel.Parameter(value = 0)
params['k3'] = radvel.Parameter(value = 3)

params['per4'] = radvel.Parameter(value = 2.202)
params['tc4'] = radvel.Parameter(value = 2458517.8921982893)
params['e4'] = radvel.Parameter(value = 0.02248012647962938, vary=True)
params['w4'] = radvel.Parameter(value = 0)
params['k4'] = radvel.Parameter(value = 2.48592961568797)

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

mod.params['per4'].vary = False
mod.params['tc4'].vary = False
mod.params['secosw4'].vary = True
mod.params['sesinw4'].vary = True

mod.params['dvdt'].vary = True
mod.params['curv'].vary = False

priors = [radvel.prior.PositiveKPrior(4)]
priors += [radvel.prior.Gaussian('per1', 0.45, 3.2e-05)]
priors += [radvel.prior.Gaussian('tc1', 2458517.4973, 0.0018)]
priors += [radvel.prior.Gaussian('per2', 10.78, 0.00015)]
priors += [radvel.prior.Gaussian('tc2', 2458527.05825, 0.00053)]
priors += [radvel.prior.Gaussian('per3', 16.29, 5e-05)]
priors += [radvel.prior.Gaussian('tc3', 2458521.8828, 0.0035)]
priors += [radvel.prior.EccentricityPrior(4, upperlims=[0.99, 0.3168456768287693, 0.98, 0.99])]
for telescope in telescopes:
	priors += [radvel.prior.HardBounds(f'jit_{telescope}', -20.0, 20.0)]

stellar = dict(mstar=0.81, mstar_err=0.03)