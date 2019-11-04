#!/usr/bin/env python3
r"""
Test on the following analytic case with 2 j_1's

\int 1/(1+x^2) j_1(ax)j_1(bx) x^2dx = pi/(2a^2b^2)
    / (1+b) e^-b (a cosha - sinha),  a<=b
    \ (1+a) e^-a (b coshb - sinhb),  a>b

This is one of the simplest cases where not all the terms of the sin-cos
expansion converges.

The output of each sine or cosine integrals evaluated with FFTLog algorithm is
logarithmically spaced, from which interpolation is needed at (a+b) or (a-b).

Compared to cubic spline (C2), (cubic) Hermite interpolation is only C1 but
local (so fast).
Because the user need to provide derivative, therefore supposedly it should be
more accurate than spline when interpolating for derivatives.

In the case of the multiple-j algorithm here, for example two j's, when one of
the two output arguments is much bigger than the other, the product-to-sum
identities would sometimes produce cancellations that needs accurate
interpolation for the first derivatives.
And for more j's, even higher derivatives are needed for accuracy when one of
the output argument is much bigger than all others.

This test compares CubicSpline to BPoly.from_derivatives for this purpose.
"""

import numpy as np
from mcfit.transforms import FourierSine, FourierCosine
from scipy.interpolate import CubicSpline, BPoly
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm

lgxmin, lgxmax = -2, 4
Nx_perdex = 50
Nx = int(Nx_perdex * (lgxmax - lgxmin))
x = np.logspace(lgxmin, lgxmax, num=Nx, endpoint=False)
F = 1 / (1 + x*x)
Fc0 = np.sqrt(np.pi / 2) * F * x**2
Fsm1 = np.sqrt(np.pi / 2) * F * x
Fcm2 = np.sqrt(np.pi / 2) * F
Fsm3 = np.sqrt(np.pi / 2) * F / x
Fcm4 = np.sqrt(np.pi / 2) * F / x**2

extrap = True
N = 8096

def symmetrize(y, G, dGdy=None, d2Gdy2=None, parity=0):
    """Symmetrize G(y) and G'(y) before interpolation (particularly for cubic
    spline because Hermite interp does not need the full negative half but just
    one segment) to cover [0, ymin) and to respect the symmetry.
    """
    y = np.concatenate((- y[::-1], y))
    G = np.concatenate(((-1)**parity * G[::-1], G))
    if dGdy is not None:
        dGdy = np.concatenate((-(-1)**parity * dGdy[::-1], dGdy))
        G = np.column_stack((G, dGdy))
        if d2Gdy2 is not None:
            d2Gdy2 = np.concatenate(((-1)**parity * d2Gdy2[::-1], d2Gdy2))
            G = np.column_stack((G, d2Gdy2))
    return y, G

qc0 = 2.5
print('Fourier Cosine transform of x^2 / (1+x^2), with tilt q =', qc0)
Tc0 = FourierCosine(x, q=qc0, N=N, lowring=False)
Tc0.check(Fc0)
yc0, C0 = Tc0(Fc0, extrap=extrap)
y = yc0

qsm1 = qc0 - 1
print('Fourier Sine transform of x^2 / [(1+x^2) x^1], with tilt q =', qsm1)
Tsm1 = FourierSine(x, q=qsm1, N=N, lowring=False)
Tsm1.check(Fsm1)
ysm1, Sm1 = Tsm1(Fsm1, extrap=extrap)
assert all(y == ysm1)

qcm2 = qsm1 - 1
print('Fourier Cosine transform of x^2 / [(1+x^2) x^2], with tilt q =', qcm2)
Tcm2 = FourierCosine(x, q=qcm2, N=N, lowring=False)
Tcm2.check(Fcm2)
ycm2, Cm2 = Tcm2(Fcm2, extrap=extrap)
assert all(y == ycm2)
Cm2_cspline = CubicSpline(* symmetrize(ycm2, Cm2, parity=0))
Cm2_hermite = BPoly.from_derivatives(* symmetrize(ycm2, Cm2, dGdy=-Sm1, parity=0))
Cm2_hermite5 = BPoly.from_derivatives(* symmetrize(ycm2, Cm2, dGdy=-Sm1, d2Gdy2=-C0, parity=0))

qsm3 = qcm2 - 1
print('Fourier Sine transform of x^2 / [(1+x^2) x^3], with tilt q =', qsm3)
Tsm3 = FourierSine(x, q=qsm3, N=N, lowring=False)
Tsm3.check(Fsm3)
ysm3, Sm3 = Tsm3(Fsm3, extrap=extrap)
assert all(y == ysm3)
Sm3_cspline = CubicSpline(* symmetrize(ysm3, Sm3, parity=1))
Sm3_hermite = BPoly.from_derivatives(* symmetrize(ysm3, Sm3, dGdy=Cm2, parity=1))
Sm3_hermite5 = BPoly.from_derivatives(* symmetrize(ysm3, Sm3, dGdy=Cm2, d2Gdy2=-Sm1, parity=1))

qcm4 = qsm3 - 1
print('Fourier Cosine transform of x^2 / [(1+x^2) x^4], with tilt q =', qcm4)
Tcm4 = FourierCosine(x, q=qcm4, N=N, lowring=False)
Tcm4.check(Fcm4)
ycm4, Cm4 = Tcm4(Fcm4, extrap=extrap)
assert all(y == ycm4)
Cm4_cspline = CubicSpline(* symmetrize(ycm4, Cm4, parity=0))
Cm4_hermite = BPoly.from_derivatives(* symmetrize(ycm4, Cm4, dGdy=-Sm3, parity=0))
Cm4_hermite5 = BPoly.from_derivatives(* symmetrize(ycm4, Cm4, dGdy=-Sm3, d2Gdy2=-Cm2, parity=0))

def trigsum_cspline(a, b):
    return ( (Cm4_cspline(a-b) - Cm4_cspline(a+b))
           + (Cm2_cspline(a-b) + Cm2_cspline(a+b)) * (a * b)
           + (Sm3_cspline(a-b) - Sm3_cspline(a+b)) * a
           - (Sm3_cspline(a-b) + Sm3_cspline(a+b)) * b
           ) / 2 / (a * b)**2

def trigsum_hermite(a, b):
    return ( (Cm4_hermite(a-b) - Cm4_hermite(a+b))
           + (Cm2_hermite(a-b) + Cm2_hermite(a+b)) * (a * b)
           + (Sm3_hermite(a-b) - Sm3_hermite(a+b)) * a
           - (Sm3_hermite(a-b) + Sm3_hermite(a+b)) * b
           ) / 2 / (a * b)**2

def trigsum_hermite5(a, b):
    return ( (Cm4_hermite5(a-b) - Cm4_hermite5(a+b))
           + (Cm2_hermite5(a-b) + Cm2_hermite5(a+b)) * (a * b)
           + (Sm3_hermite5(a-b) - Sm3_hermite5(a+b)) * a
           - (Sm3_hermite5(a-b) + Sm3_hermite5(a+b)) * b
           ) / 2 / (a * b)**2

def analytic(a, b):
    aleqb = (1+b) * np.exp(-b) * (a * np.cosh(a) - np.sinh(a))
    ageqb = (1+a) * np.exp(-a) * (b * np.cosh(b) - np.sinh(b))
    combined = np.pi / 2 / (a * b)**2 * np.where(a<=b, aleqb, ageqb)
    return combined

Nab_perdex = 5
Nab = int(Nab_perdex * (lgxmax - lgxmin))
ab = np.logspace(-lgxmax, -lgxmin, num=Nab, endpoint=False)
a = ab[:, None]
b = ab[None, :]
ana = analytic(a, b)
num_cspline = trigsum_cspline(a, b)
num_hermite = trigsum_hermite(a, b)
num_hermite5 = trigsum_hermite5(a, b)
# normalize the error
err_cspline = (num_cspline - ana) / np.sqrt(analytic(a, a) * analytic(b, b))
err_hermite = (num_hermite - ana) / np.sqrt(analytic(a, a) * analytic(b, b))
err_hermite5 = (num_hermite5 - ana) / np.sqrt(analytic(a, a) * analytic(b, b))

# set colorbar norm
vlim = abs(ana).max()
norm = SymLogNorm(1e-2, vmin=-1, vmax=1)

# exact
plt.figure(figsize=(4.5, 3.6))
plt.pcolormesh(ab, ab, ana, cmap='Reds', norm=LogNorm(vmin=1e-6*vlim, vmax=vlim))
plt.colorbar()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$a$')
plt.ylabel('$b$')
plt.savefig('j1j1.pdf')

# numerical vs exact, in slices
plt.figure(figsize=(3.6, 2.7))
plt.plot(ab, ana.diagonal(), c='0.7', ls='-', lw=1.0)
plt.plot(ab, ana[:,  5], c='orangered', ls='-', lw=1.0, label='$b={}$'.format(ab[5]))
plt.plot(ab, ana[:, 15], c='orangered', ls='-', lw=0.7, label='$b={}$'.format(ab[15]))
plt.plot(ab, ana[:, 25], c='orangered', ls='-', lw=0.4, label='$b={}$'.format(ab[25]))
plt.plot(ab, num_cspline[:,  5], c='steelblue', ls='--', lw=1.0, label='numerical'.format(ab[10]))
plt.plot(ab, num_cspline[:, 15], c='steelblue', ls='--', lw=0.7)
plt.plot(ab, num_cspline[:, 25], c='steelblue', ls='--', lw=0.4)
plt.plot(ab, - num_cspline[:,  5], c='steelblue', ls=':', lw=1.0)
plt.plot(ab, - num_cspline[:, 15], c='steelblue', ls=':', lw=0.7)
plt.plot(ab, - num_cspline[:, 25], c='steelblue', ls=':', lw=0.4)
plt.xscale('log')
plt.yscale('log')
plt.xlim(ab.min(), ab.max())
plt.ylim(1e-9, 1e5)
plt.legend(loc='upper right', fontsize='x-small', ncol=2)
plt.xlabel('$a$')
plt.savefig('j1j1_cspline.pdf')

plt.figure(figsize=(3.6, 2.7))
plt.plot(ab, ana.diagonal(), c='0.7', ls='-', lw=1.0)
plt.plot(ab, ana[:,  5], c='orangered', ls='-', lw=1.0, label='$b={}$'.format(ab[5]))
plt.plot(ab, ana[:, 15], c='orangered', ls='-', lw=0.7, label='$b={}$'.format(ab[15]))
plt.plot(ab, ana[:, 25], c='orangered', ls='-', lw=0.4, label='$b={}$'.format(ab[25]))
plt.plot(ab, num_hermite[:,  5], c='steelblue', ls='--', lw=1.0, label='numerical'.format(ab[10]))
plt.plot(ab, num_hermite[:, 15], c='steelblue', ls='--', lw=0.7)
plt.plot(ab, num_hermite[:, 25], c='steelblue', ls='--', lw=0.4)
plt.plot(ab, - num_hermite[:,  5], c='steelblue', ls=':', lw=1.0)
plt.plot(ab, - num_hermite[:, 15], c='steelblue', ls=':', lw=0.7)
plt.plot(ab, - num_hermite[:, 25], c='steelblue', ls=':', lw=0.4)
plt.xscale('log')
plt.yscale('log')
plt.xlim(ab.min(), ab.max())
plt.ylim(1e-9, 1e5)
plt.legend(loc='upper right', fontsize='x-small', ncol=2)
plt.xlabel('$a$')
plt.savefig('j1j1_hermite.pdf')

plt.figure(figsize=(3.6, 2.7))
plt.plot(ab, ana.diagonal(), c='0.7', ls='-', lw=1.0)
plt.plot(ab, ana[:,  5], c='orangered', ls='-', lw=1.0, label='$b={}$'.format(ab[5]))
plt.plot(ab, ana[:, 15], c='orangered', ls='-', lw=0.7, label='$b={}$'.format(ab[15]))
plt.plot(ab, ana[:, 25], c='orangered', ls='-', lw=0.4, label='$b={}$'.format(ab[25]))
plt.plot(ab, num_hermite5[:,  5], c='steelblue', ls='--', lw=1.0, label='numerical'.format(ab[10]))
plt.plot(ab, num_hermite5[:, 15], c='steelblue', ls='--', lw=0.7)
plt.plot(ab, num_hermite5[:, 25], c='steelblue', ls='--', lw=0.4)
plt.plot(ab, - num_hermite5[:,  5], c='steelblue', ls=':', lw=1.0)
plt.plot(ab, - num_hermite5[:, 15], c='steelblue', ls=':', lw=0.7)
plt.plot(ab, - num_hermite5[:, 25], c='steelblue', ls=':', lw=0.4)
plt.xscale('log')
plt.yscale('log')
plt.xlim(ab.min(), ab.max())
plt.ylim(1e-9, 1e5)
plt.legend(loc='upper right', fontsize='x-small', ncol=2)
plt.xlabel('$a$')
plt.savefig('j1j1_hermite5.pdf')

# normalized error
plt.figure(figsize=(4.5, 3.6))
plt.pcolormesh(ab, ab, err_cspline, cmap='RdBu_r', norm=norm)
plt.colorbar()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$a$')
plt.ylabel('$b$')
plt.savefig('j1j1_cspline_err.pdf')

plt.figure(figsize=(4.5, 3.6))
plt.pcolormesh(ab, ab, err_hermite, cmap='RdBu_r', norm=norm)
plt.colorbar()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$a$')
plt.ylabel('$b$')
plt.savefig('j1j1_hermite_err.pdf')

plt.figure(figsize=(4.5, 3.6))
plt.pcolormesh(ab, ab, err_hermite5, cmap='RdBu_r', norm=norm)
plt.colorbar()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$a$')
plt.ylabel('$b$')
plt.savefig('j1j1_hermite5_err.pdf')
