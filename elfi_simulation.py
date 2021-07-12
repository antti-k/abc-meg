import matlab.engine
import numpy as np
import elfi
from scipy.io import loadmat
eng = matlab.engine.start_matlab()

#mat = loadmat('MOD_strength_wPLI_FCmean.mat')
#observed = np.array(mat['MOD_strength_wPLI_FCmean'])
mat = loadmat('wPLImats_strength_groupmat1.mat')
a = np.array(mat['wPLImats_strength_groupmat1'])
indices = [1 + a * 2 for a in range(74)]
a = a[:, indices]
a = a[indices, :]
observed = a

def simulator(wee, batch_size=1, random_state=None):
    print(wee)
    ret = eng.simulate(wee.item())
    return np.array(ret)

#def rms(simulated, observed):
    #return np.square(simulated - observed).mean(axis=None)

def distance(simulated):
    return np.square(simulated - observed).mean(axis=None)

wee = elfi.Prior('uniform', 14, 2)
sim = elfi.Simulator(simulator, wee, observed=0)

S1 = elfi.Summary(distance, sim)
#S2 = elfi.Summary(var, sim)


#d = elfi.Distance('euclidean', S1, S2)
d = elfi.Distance('euclidean', S1)

rej = elfi.Rejection(d)
res = rej.sample(100, threshold=.05)
print(res)
