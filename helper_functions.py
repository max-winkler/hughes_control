import numpy as np

from dolfin import *
# from mshr import *

def ufl_abs(a):
    #return abs(a)	
    return (a**2 + 1e-3)**0.5
def d_ufl_abs(a):
    return a/ufl_abs(a)
def ufl_max(a,b):
    return conditional(a<b, b, a)
def ufl_min(a,b):
    return conditional(a<b, a, b)

c = 0.4
eta = 20

MaxUInt = np.iinfo(np.uint32).max

# Kernel for agent potential
def K(x):

    return c*np.exp(-eta*np.linalg.norm(x)**2)

def D_K(x):
    intensity = 0.4
    return -2*eta*c*np.exp(-eta*np.linalg.norm(x)**2)*x

def DD_K(x):
    intensity = 0.4
    return 2*eta*intensity*np.exp(-eta*np.linalg.norm(x)**2)*(2*eta*np.outer(x,x) - np.identity(2))

class PotentialAgent(Expression):	
    def __init__(self, pos, **kwargs):
        self.pos = pos
	    
    def eval(self, values, x):
        val = K(x-self.pos)
        
        values[0] = -val

class GradPotentialAgent(Expression):
    def __init__(self, pos, **kwargs):
        self.pos = pos
    def eval(self, values, x):
        val = D_K(x-self.pos)
        
        values = val

class PotentialAgentDeriv1(Expression):    
    def __init__(self, pos, **kwargs):
        self.pos = pos
	    
    def eval(self, values, x):
        val = D_K(x-self.pos)
        
        values[0] = val[0]

class PotentialAgentDeriv2(Expression):
    def __init__(self, pos, **kwargs):
        self.pos = pos
	    
    def eval(self, values, x):
        val = D_K(x-self.pos)
        
        values[0] = val[0]
            
def get_example_geo(i):
    if i==0:
        mesh = UnitSquareMesh(99,99)
        return mesh
    elif i==1:
        base = Rectangle(Point(0, 0), Point(2, 1))        
        wall1 = Rectangle(Point(1, 0), Point(1.1,0.2))
        wall2 = Rectangle(Point(1, 0.4), Point(1.1,0.7))
        wall3 = Rectangle(Point(1, 0.9), Point(1.1,1))
        
        geometry = base - wall1 - wall2 - wall3
        mesh = generate_mesh(geometry,50)

        return mesh
        
    
