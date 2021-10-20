import numpy as np
from scipy.sparse import diags
from scipy.integrate import odeint
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment

# Five-point stencils used for derivatives
d1 = np.array([0.08333333, -0.66666667,  0., 0.66666667, -0.08333333])
d2 = np.array([-0.08333333, 1.33333333, -2.5, 1.33333333, -0.08333333])
d4 = np.array([1., -4.,  6., -4.,  1.])


def adjsol(sol):
    return sol-np.average(sol, axis=1)[:, None]


def makeop(d, N, periodic=True):
    M = diags(d, np.arange(-2, 3), shape=(N, N))
    if periodic:
        M += diags(d[[0, 1, 3, 4]], [N-2, N-1, 1-N, 2-N], shape=(N, N))
    return M


def kurasiva(L=7.0, N=100, dt=0.01, tsteps=100, init=None, periodic=True,
             d1_sign=-1):

    # x axis
    x = np.linspace(0, L, N)
    dx = x[1]
    # time axis
    t = np.arange(1, tsteps+1)*dt

    # Initialise function
    if init is None:
        def init_default(x):
            rnd = np.random.random(len(x))-0.5
            rnd -= np.sum(rnd)/len(x)

            return np.cumsum(rnd)

        init = init_default

    u = init(x)

    # Define the matrices
    D1 = makeop(d1, N, periodic)/dx
    D2 = makeop(d2, N, periodic)/dx**2
    D4 = makeop(d4, N, periodic)/dx**4

    d1_sign /= abs(d1_sign)

    def dudt(u, t):
        return -D2*u-D4*u+0.5*d1_sign*(D1*u)**2

    sol = odeint(dudt, u, t)

    return sol


def follow_peaks(sol):

    peaks = []
    for sl in adjsol(sol):
        p, _ = find_peaks(sl)
        peaks.append(p)

    # Now create lines by matching
    peak_lines = [[[0], [p]] for p in peaks[0]]

    for step, pline in enumerate(peaks):
        if step == 0:
            continue
        # Which lines are still ongoing?
        current_lines = [l for l in peak_lines if l[0][-1] == step-1]
        # Where are they?
        current_tips = np.array([l[1][-1] for l in current_lines])
        
        Cmat = abs(current_tips[:,None]-pline[None,:])
        row_assign, col_assign = linear_sum_assignment(Cmat)
        
        for (r, c) in zip(row_assign, col_assign):
            current_lines[r][0].append(step)
            current_lines[r][1].append(pline[c])
        
        new_peaks = set(range(len(pline)))-set(col_assign)
        for i in new_peaks:
            peak_lines.append([[step], [pline[i]]])

    return peak_lines