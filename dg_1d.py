###############################################################################
# DG Demo By ShiJingchang, 2016年 03月 16日 星期三 20:39:42 CST
# Email: jingchangshi@gmail.com
# Translate from MATLAB scripts written by ...
# Reference:
#   Cockburn, B., & Shu, C.-W. (2001). Runge – Kutta Discontinuous Galerkin
#   Methods for Convection-Dominated Problems. Journal of Scientific Computing,
#   16(3), 173–261. http://doi.org/10.1023/A:1012873910884
###############################################################################
# 1D Scalar Conservation Eqn solution using Discontinuous Galerkin Linear 
# spatial disretization, forward Euler time discretization.
# u_t + f(u)_x = 0
# u = u(x, t), u(x, 0) = u_0 = sin(2 pi x)
# Let u_h be the approximate numerical solution consisting of a linear
# combination of basis functions in the finite vector space of linear
# polynomials, with scalar coefficients: \sum Coeffi_i * u_h,i
# Let v_h,j be the jth test function in the same vector space as the basis
# functions j = 1...M and in this case M = 2.
# Discretize the domain into K elements with K+1 nodes.


import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, cos, pi, integrate, Matrix

TestfuncNMAX = 2
ElementNMAX = 32
XNode_Vec = np.linspace(0.0, 1.0, ElementNMAX + 1)
ElementSize = 1.0 / ElementNMAX
# Get the initial value of coefficients of basis functions.
# According to the reference, \int u_h v_h = \int u_0 v_h. Then we can get
# the matrix form:
#       Coeffi_1 * u_h,1 * v_h,1 + Coeffi_2 * u_h,2 * v_h,1
#   =   [u_h,1 * v_h,1  , u_h,2 * v_h,1 ]       *   [Coeffi_1]
#       [ 0             , 0             ]       *   [Coeffi_2]
#   =   \int u_0 v_h,1
# Note: the correct basis function is like (x-a)/(b-a), to ensure that the
#       basis function is defined locally, i.e. to be 0 at one end and 1 at
#       the other end of the element. Functions like x is wrong!
# Note: For a local element, Mass Matrix should be multiplied by the element size.
#       If You can't understand, integrate it one step by step will help you.
MassMatrix_Mat = np.array([[1.0/3.0, 1.0/6.0], [1.0/6.0, 1.0/3.0]])
x = symbols('x')
u_0 = cos(2*pi*x)

a, b = symbols('a b')
Integral_u0_Mat = Matrix(np.zeros((TestfuncNMAX, ElementNMAX)))
# Test function v_{h,j=1} is (b-x)/(b-a), where a is the initial point of the local element and b is the end point of the local element.
v_h = (b - x) / (b - a)
Integrand_u0 = u_0 * v_h
Integral_u0 = integrate(Integrand_u0, (x, a, b))
for ElementInd in range(ElementNMAX):
    Integral_u0_Mat[0, ElementInd] = \
        Integral_u0.subs(a, XNode_Vec[ElementInd])
    Integral_u0_Mat[0, ElementInd] = \
        Integral_u0_Mat[0, ElementInd].subs(b, XNode_Vec[ElementInd+1])
    Integral_u0_Mat[0, ElementInd] = \
        Integral_u0_Mat[0, ElementInd].evalf()

# test function v_{h,j=2} is (x-a)/(b-a)
v_h = (x - a) / (b - a)
Integrand_u0 = u_0 * v_h
Integral_u0 = integrate(Integrand_u0, (x, a, b))
for ElementInd in range(ElementNMAX):
    Integral_u0_Mat[1, ElementInd] = \
        Integral_u0.subs(a, XNode_Vec[ElementInd])
    Integral_u0_Mat[1, ElementInd] = \
        Integral_u0_Mat[1, ElementInd].subs(b, XNode_Vec[ElementInd+1])
    Integral_u0_Mat[1, ElementInd] = \
        Integral_u0_Mat[1, ElementInd].evalf()

Integral_u0_Mat = np.array(Integral_u0_Mat).astype(np.float64)
Coeffi_Mat = np.linalg.solve(MassMatrix_Mat * ElementSize, Integral_u0_Mat)

# Assemble SemiMatrix system
# Employ upwind flux(refer to the definition of numerical flux in reference).
UpwindFlux_Mat = np.array([[-1, 0, 0], [0, 0, 1]])
StiffnessMatrix_Mat = np.array([[0, -0.5, -0.5], [0, 0.5, 0.5]])

Stencil_Mat = np.linalg.solve(MassMatrix_Mat * ElementSize, \
    StiffnessMatrix_Mat - UpwindFlux_Mat)

SemiMatrix_Mat = np.zeros((ElementNMAX*TestfuncNMAX, ElementNMAX*TestfuncNMAX))
for ElementInd in range(ElementNMAX-1):
    StencilInd = ElementInd * 2
    SemiMatrix_Mat[StencilInd:StencilInd+2, StencilInd:StencilInd+3] = \
        Stencil_Mat
# Apply periodic BC
SemiMatrix_Mat = np.roll(SemiMatrix_Mat, -1, axis=1)
StencilInd = StencilInd + 2
SemiMatrix_Mat[StencilInd:StencilInd+2, StencilInd-1:StencilInd+2] = \
    Stencil_Mat

# Employ Euler time marching scheme
TimeStep = 1E-4
TimeNMAX = np.int(1E3)

Coeffi_Mat = Coeffi_Mat.reshape((ElementNMAX*TestfuncNMAX, 1), order='F')

for TimeInd in range(TimeNMAX):
    Coeffi_Mat = np.dot( np.eye(ElementNMAX*TestfuncNMAX) + SemiMatrix_Mat * TimeStep, \
        Coeffi_Mat)

# Combine local solutions to global solution
# Given X value, find the location in XNode_Vec.
def getUh(XValue):
    XValueIndexH = np.where(XNode_Vec >= XValue)[0][0]
    if XValueIndexH == 0:
        XValueIndexH = XValueIndexH + 1
    XValueIndexL = XValueIndexH - 1
    ElementEnd = XNode_Vec[XValueIndexH]
    ElementInit = XNode_Vec[XValueIndexL]
    Coeffi1 = Coeffi_Mat[XValueIndexL*2, 0]
    Coeffi2 = Coeffi_Mat[XValueIndexL*2+1, 0]
    FunctionUh = Coeffi1 * (ElementEnd - XValue) / ElementSize \
        + Coeffi2 * (XValue - ElementInit) / ElementSize
    return FunctionUh

# Plot the global solution
PltXNodeNMAX = np.int(ElementNMAX*4)
PltXNode_Vec = np.linspace(0.0, 1.0, PltXNodeNMAX)
UhValue_Vec = np.zeros(PltXNodeNMAX)
for PltInd in range(PltXNodeNMAX):
    XValue = PltXNode_Vec[PltInd]
    UhValue_Vec[PltInd] = getUh(XValue)

plt.plot(XNode_Vec, np.cos(2*np.pi*XNode_Vec), 'r')
plt.plot(np.linspace(0.0, 1.0, PltXNodeNMAX), UhValue_Vec, 'g')
plt.show()

