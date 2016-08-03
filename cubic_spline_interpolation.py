import numpy as np

def ClampedCubicSplineCoefficients( x, a, dxdt ) : 
    n = len(x) 
    h = np.zeros(n-1) 
    for i in range(n-1) : 
        h[i] = x[i+1]-x[i]
        
    A = np.zeros((n,n))
    A[0,0:2] = np.array([2.*h[0], h[0]])
    A[-1,n-2:n] = np.array([h[-1],2.*h[-1]])
    for i in range(1,n-1) : 
        A[i,i-1:i+2] = np.array([h[i-1],2.*(h[i-1]+h[i]),h[i]])
        
    beta = np.zeros(n)
    beta[0] = 3.*( (a[1]-a[0])/h[0] - dxdt[0] )
    beta[-1] = 3.*( dxdt[-1] - (a[-1]-a[-2])/h[-1] )

    for i in range(1,n-1) :
        beta[i] = 3.*( (a[i+1]-a[i])/h[i]-(a[i]-a[i-1])/h[i-1] )
    c = np.linalg.solve(A,beta)
    b = np.zeros(n-1)
    d = np.zeros(n-1)
    
    for i in range(n-1) :
        b[i] = (a[i+1]-a[i])/h[i] - h[i]*(2.*c[i]+c[i+1])/3.
        d[i] = (c[i+1]-c[i])/(3.*h[i])
    return a,b,c,d

def ClampedCubicSpline( x, alpha, dxdt, intervals ) :
    [a,b,c,d] = ClampedCubicSplineCoefficients( x, alpha, dxdt )
    n = len(x)
    xmod = np.zeros(([intervals,n-1]))
    ymod = np.zeros(([intervals,n-1]))
    Vmod = np.zeros(([intervals,n-1]))
    for i in range(n-1):
        xmod[:,i] = np.linspace(x[i],x[i+1],intervals)
        ymod[:,i] = a[i] + b[i]*(xmod[:,i]-x[i]) + c[i]*(xmod[:,i]-x[i])**2. \
        + d[i]*(xmod[:,i]-x[i])**3.
        Vmod[:,i] = b[i] + 2.*c[i]*(xmod[:,i]-x[i]) \
        + 3.*d[i]*(xmod[:,i]-x[i])**2.
        
    return xmod, ymod, Vmod
