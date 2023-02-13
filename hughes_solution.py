import sys
from math import *
from dolfin import *
from ufl import ge

# Set some fenics parameters
# parameters["form_compiler"]["quadrature_degree"] = 150
# dolfin.parameters['ghost_mode'] = 'shared_facet'

from setup_example import *
from helper_functions import *

##############################################################
#     Define options, model and discretization parameters    #
##############################################################

# Define options
with_plot = True               # Enable plotting
verbosity = 2                  # Verbosity1=less, 2=more output

# Choose algorithm for agent dynamics
with_fixed_point_it=True       # Solution of ODE with fixed-point method
with_regularized_dirac = True  # Use regularized Dirac for point evaluations

# Define regularization parameters
alpha_1 = 1.e-4
alpha_2 = 1.e-4 # ATTENTION: Code works only when alpha1=alpha2 (see ValueJ)

# Define directory for plots and log files
output_basedir = 'output'

# Choose example (defined in setup_example.py)
example = Example2()

# Setup model parameters
delta_1 = Constant(2.e-1)      # Laplacian in Eikonal eq.
delta_2 = Constant(1.e-1)      # Right-hand side of Eikonal eq.
delta_3 = Constant(1.e-2)      # Regularization of normed velocity vec.
delta_4 = Constant(1.e-1)      # Regularization in barrier function
delta_5 = Constant(1.e-2)      # Regularization parameter of the Dirac delta
                               
barrier_function = 0           # 0 for log barrier, 1 for quadratic penalty
epsilon = Constant(1.e-5)      # Laplacian in rho eq.
penalty = Constant(1.e0)       # jump penalty term in DG formulation (must be one for zero-order)

outflow_vel = 10.0             # Outflow velocity (gamma in the boundary condition)

# Note: More parameters are defined in the example class (see setup_example.py)

##############################################################
#       Main program starts here (don't cross this line)     #
##############################################################

gamma = example.gamma

# Define parameters for time discretization
T           = example.T                          # Final time T
N           = example.N                        # Number of time steps
tau         = T/N                      # Time discretization parameter

# Get data from example
mesh        = example.mesh
rho_0       = example.rho_0
ag_pos_0    = example.ag_pos_0
inside_room = example.inside_room
wall_region = example.wall_region

BoundingBox = mesh.bounding_box_tree()

# Indicator functions for the boundaries
boundary_doors = lambda x,on_boundary : example.exits.inside(x, on_boundary)
boundary_walls = lambda x,on_boundary : example.walls.inside(x, on_boundary)

nr_agents = np.size(ag_pos_0, 0)

# Define finite element spaces
R = FunctionSpace(mesh, "DG", 0)           # Density
P = FunctionSpace(mesh, "CG", 1)           # Potential
V_vec = VectorFunctionSpace(mesh, "DG", 0) # Gradients of P-functons

# Dirichlet boundary condition for phi
bc_phi = DirichletBC(P, Constant(0), boundary_doors)

# Set some trial and test functions
phi_ = TrialFunction(P)
z = TestFunction(P)
rho_ = TrialFunction(R)
w = TestFunction(R)

# Mesh entities
n = FacetNormal(mesh)
h = CellDiameter(mesh)
x = SpatialCoordinate(mesh)

# Define new surface measure for the boundary conditions
boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_markers.set_all(0)

exits = example.exits
exits.mark(boundary_markers, 1)

dss = Measure('ds', domain=mesh, subdomain_data=boundary_markers) # Exits

# Define new measure to distinguish between inside and outside of room
class InsideRoom(SubDomain):
    def inside(self, x, on_boundary):
        return inside_room(x)

# Indicator function for subdomain where density is penalized in objective
inside = InsideRoom()
domain_inside = MeshFunction("size_t", mesh, mesh.topology().dim())
domain_inside.set_all(0)
inside.mark(domain_inside, 1)
dx = Measure('dx', domain=mesh, subdomain_data=domain_inside)

sip = Constant(1.0)

# Compute Barrier function for state constraints
zeta_ = TrialFunction(P)
zeta = Function(P)

if barrier_function == 0:
    a_barrier = delta_4*inner(grad(zeta_), grad(z))*dx + zeta_*z*dx
    rhs_barrier = z*dx

    bc_doors = DirichletBC(P, Constant(0.0), boundary_doors)
    bc_walls = DirichletBC(P, Constant(0.0), boundary_walls)    

    solve(a_barrier == rhs_barrier, zeta, [bc_doors, bc_walls])

    # zeta = project(Constant(1.), P)
    
elif barrier_function == 1:
    F_barrier = delta_4*inner(grad(zeta), grad(z))*dx + inner(grad(zeta),grad(zeta))*z*dx - z*dx

    bc_doors = DirichletBC(P, Constant(0.0), boundary_doors)
    bc_walls = DirichletBC(P, Constant(0.0), boundary_walls)

    solve(F_barrier == 0, zeta, [bc_doors, bc_walls])

grad_zeta = project(grad(zeta), V_vec)

file_barrier = File(output_basedir + '/barrier.pvd')
file_barrier << zeta

# Test if point is in the domain
def PointInside(ag_point):
    return BoundingBox.compute_first_entity_collision(ag_point) < MaxUInt

# Get UFL expression of the numerical flux
def Flux(rho_, beta, n):
    
    # Lax-Friedrichs flux
    flux = inner(avg(rho_*beta), n('+')) - 0.5*jump(rho_)
    return -flux

# Plot density
def plot_rho(rhos):
    
    if with_plot:
        file_rho = File(output_basedir + '/rho.pvd')
        file_mass = open(output_basedir + '/mass.csv', "w")
        
        for k in range(N):
            rho = Function(R)
            rho.vector().set_local(rhos[k,:])
            rho.rename("rho", "density")
            file_rho << rho,k
            
            # mass = sqrt(assemble(rho*dx));
            # file_mass.write("%f\n" % mass)

# Plot potential
def plot_phi(phis, ag_poss, cs):
    
    if with_plot:
        file_phi = File(output_basedir + '/phi.pvd')
        file_grad_phi = File(output_basedir + '/grad_phi.pvd')

        for k in range(N):
            full_phi = Function(P)

            if nr_agents > 0:
                ag_pos_ = as_matrix([[Constant(ag_poss[ii,k,0]), Constant(ag_poss[ii,k,1])] \
                                     for ii in range(nr_agents)])
                intensity_ = as_vector([Constant(cs[ii,k]) for ii in range(nr_agents)])

            # Agent potential
            phi_K = Constant(0.0)
            for i in range(nr_agents):
                phi_K += -intensity_[i] * AgentKernel(x,ag_pos_[i,:])
            
            phi_ag = project(phi_K, P)
            phi_orig = Function(P)
            phi_orig.vector().set_local(phis[k,:])
            
            full_phi.vector()[:] = phi_ag.vector().get_local() + phi_orig.vector().get_local()
            full_phi.rename("phi", "potential")
            file_phi << full_phi,k

            # Plot gradient of potential
            grad_full_phi = project(grad(full_phi), V_vec)
            grad_full_phi.rename("grad phi", "direction")
            file_grad_phi << grad_full_phi

# Plot adjoint density
def plot_rho_adj(ps):
    
    if with_plot:
        file_p = File(output_basedir + '/p.pvd')

        for k in range(N):
            p = Function(R)
            p.vector().set_local(ps[k,:])
            p.rename("p", "dual density")
            file_p << p,N-1-k
    
# Potential function
def f(rho):
    return 1.-rho
def Df(rho):
    return -1.0

# Smoothed dirac delta
def dirac(x):
    return 1/(2*pi*delta_5)*exp(-inner(x,x)/(2*delta_5))

# Velocity constraint
def v_trunc(Dphi):
    return conditional(lt(inner(Dphi,Dphi),1), Dphi, Dphi/sqrt(inner(Dphi,Dphi)))

# Kernel function for the agent (eg. a bump function)
def AgentKernel(x, ag_pos):
    #return conditional(dot(x-ag_pos,x-ag_pos) < eta**2, \
    #                   exp(5)*exp(-5*eta**2/(eta**2-dot(x-ag_pos,x-ag_pos))), \
    #                   Constant(0.0))
    dist = sqrt(dot(x-ag_pos,x-ag_pos))
    return -2*exp(-2*4*(dist-0.2)) + 4*exp(-3*(dist-0.2))

def SolveForward(us, cs, get_initial_guess=False):

    # Set initial values
    ag_pos = ag_pos_0
    ag_poss = np.empty((nr_agents,N+1,2), dtype=float)   
    ag_poss[:,0,:] = ag_pos

    rhos = np.empty((N+1,len(R.dofmap().dofs())))
    phis = np.empty((N+1,len(P.dofmap().dofs())))
    
    rho = interpolate(rho_0, R)
    rhos[0,:] = rho.vector().get_local()
    
    rho_old = Function(R)
    rho_old.assign(rho)

    # Compute initial potential
    phi = Function(P)    
    phi_ = TrialFunction(P)    
        
    # Define final phi equation
    eq_phi = delta_1 * inner(grad(phi), grad(z))*dx \
             + inner(grad(phi), grad(phi))*z*dx \
             - (1.0/(f(rho)**2 + delta_2))*z*dx
    
    # Solve Eikonal equation with Newton solver
    solve(eq_phi == 0, phi, bc_phi)

    phis[0,:] = phi.vector().get_local()

    ############################
    # Solve the state equation #
    ############################
    if verbosity > 1:
        print("  solving forward equation")

    for k in range(N):
        
        if verbosity > 1:
            print("    time step ", k+1, " of ", N, "(density: ", assemble(rho*dx(1)), ")   ", end="\r")

        #######################################
        # Solve convection diffusion equation #
        #######################################

        # Define variables for agent position and intensity
        if nr_agents > 0:
            ag_pos_ = as_matrix([[Constant(ag_pos[ii,0]), Constant(ag_pos[ii,1])] \
                                 for ii in range(nr_agents)])
            intensity_ = as_vector([Constant(cs[ii,k]) for ii in range(nr_agents)])

        # Agent potential
        phi_K = Constant(0.0)
        for i in range(nr_agents):
            phi_K += -intensity_[i] * AgentKernel(x,ag_pos_[i,:])        
            
        # Bilinear form for DIFFUSION
        a_sip = epsilon * inner(grad(rho_), grad(w))*dx\
                - sip*epsilon * inner(avg(grad(rho_)),jump(w,n))*dS \
                - sip*epsilon * inner(avg(grad(w)),jump(rho_,n))*dS \
                + sip*(penalty/avg(h)) * epsilon * jump(rho_)*jump(w)*dS \
                + sip*outflow_vel*rho_*w*dss(1)
        
        # Transport direction
        beta = f(rho_old)*v_trunc(grad(phi + phi_K))

        # Bilinear form for TRANSPORT
        flux = Flux(rho_old, beta, n)

        # higher order
        # a_upw = rho_old*inner(beta, grad(w))*dx + flux*jump(w)*dS
        # 0-order
        a_upw = flux*jump(w)*dS(metadata = {"quadrature_degree": 5})
        
        # Bilinear form for TIME DISCRETIZATION
        a_time = rho_*w*dx
        
        a_full = a_time + tau*a_sip 
        L_rho = rho_old*w*dx - tau*a_upw

        # Solve system and store solution
        solve(a_full == L_rho, rho)
        rhos[k+1,:] = rho.vector().get_local()
        rho_old.assign(rho)
                               
        ##################################
        # Solve agent dynamics equation
        ##################################
        
        # Initial guess
        ag_pos_new = np.copy(ag_pos);
        
        if with_fixed_point_it:
            # Use fixed-point iteration
            for i in range(nr_agents):                
                fp_iter = 0
                err = 1.
                while (fp_iter < 1000 and err > 1.e-15):
                
                    fp_iter +=1
                    
                    ag_pos_old = ag_pos_new[i,:]

                    # Compute rho(x_i)
                    ag_point = Point(ag_pos_new[i,:])                    
                    if with_regularized_dirac:
                        ag_point_ = as_vector([Constant(ag_pos_new[i,l]) for l in range(2)])
                        vel = 1.-assemble(dirac(x-ag_point_)*rho*dx(metadata = {"quadrature_degree": 20}))/assemble(dirac(x-ag_point_)*dx(metadata = {"quadrature_degree": 20}))  
                    else:
                        if not PointInside(ag_point):
                            vel = 1.
                        else:
                            rho_val = rho(ag_point)
                            #vel = f(rho_val)
                            vel = 1.-rho_val                            

                    # Compute resid
                    err = np.linalg.norm(ag_pos_new[i,:] - ag_pos[i,:] - tau*vel*us[i,k+1,:])

                    # Set new iterate
                    ag_pos_new[i,:] = ag_pos[i,:] + tau*vel*us[i,k+1,:]
                    
                # Test if the ODE has been solved correctly                
                if fp_iter >= 999:
                    print("\n    Warning: the fixed-point iteration for the implicit Euler did not converge!")
                    print("    Warning: The resid is ", err)
                
        else:
            # SIMPLIFIED VERSION: When the ODE is x'(t) = u(t)
            for i in range(nr_agents):
                ag_pos_new[i,:] = ag_pos[i,:] + tau*us[i,k+1,:]

        ag_pos = np.copy(ag_pos_new)
        ag_poss[:,k+1,:] = ag_pos
        
        ##############################
        # Solve the Eikonal equation #
        ##############################
        if k <= N-1:
            solve(eq_phi == 0, phi, bc_phi)
            phis[k+1,:] = phi.vector().get_local()

        sys.stdout.flush()

    if verbosity > 1:
        print('\n')
        
    return [rhos, phis, ag_poss]

# Solve the adjoin equation system and return the adjoint variables
def SolveBackward(us, cs, rhos, phis, ag_poss):
    
    # Assemble adjoint equation
    p_ = TrialFunction(R)
    p = interpolate(Constant(0.0),R)
    p_old = Function(R)
    adjs = np.empty((N+1,len(R.dofmap().dofs())))

    # Trial and test functions
    q_ = TrialFunction(P)
    q = Function(P)

    rho_old = Function(R)
    rho_new = Function(R)

    phi_old = Function(P)
    phi_new = Function(P)
    
    # Initialize multiplier for dynamics equation
    # ... index l=0/1 is the dual agent position
    # ... index l=2 is the dual intensity
    vs = np.zeros((nr_agents,N+1,2))
    ints = np.zeros((nr_agents,N+1))
        
    v = np.zeros((nr_agents,2))

    if verbosity > 1:
        print("  solving backward equation")

    for k in range(N+1)[::-1]:

        if verbosity > 1:
            print("    time step ", k, " of ", N-1, "   ", end="\r")

        # Initialize FE functions required for the computation
        rho_new.vector().set_local(rhos[k,:])            
        phi_new.vector().set_local(phis[k,:])

        # Write agent position and intensity to UFL variables
        ag_pos_new = np.copy(ag_poss[:,k,:])
        ag_pos_new_ = as_matrix([[variable(Constant(ag_pos_new[ii,0])), \
                                  variable(Constant(ag_pos_new[ii,1]))] \
                                 for ii in range(nr_agents)])
        
        intensity_new_ = as_vector([variable(Constant(cs[i,k])) for i in range(nr_agents)])
        
        # Get agent potential at new and old time step
        phi_K_new = Constant(0.0)

        for i in range(nr_agents):            
            phi_K_new += -intensity_new_[i]*AgentKernel(x, ag_pos_new_[i,:])
                
            #if k <= N-2:
            #    phi_K_old += -intensity_old_[i]*AgentKernel(x, ag_pos_old_[i,:])
                         
        # Get transport direction
        if k <= N-1:
            beta_new = f(rho_new)*v_trunc(grad(phi_new + phi_K_new))
        
        p_old.assign(p)

        ##################################################################
        # Adjoint eq. for v (Compute v^n by differentiating for x^{n+1}) #
        ##################################################################

        v_old = np.copy(v)

        # Contribution of the barrier function
        for i in range(nr_agents):
            rhs_v = np.copy(v_old[i,:])
            
            cur_ag_pos = Point(ag_pos_new[i,:])
            if not PointInside(cur_ag_pos):
                if(wall_region(cur_ag_pos)):

                    if barrier_function == 0:
                        val = 0.
                    elif barrier_function == 1:
                        val = -np.Inf
                        
                    print("Agent in wall. This should not happen.")

                    #TODO: Uncomment next line
                    # val = 1.
                    Dval = np.zeros((2))
                else:
                    val = 1.e-12
                    Dval = np.zeros((2))
            else:
                if with_regularized_dirac and k > 0:
                    cur_ag_pos_ = as_vector([variable(Constant(ag_pos_new[i,l])) for l in range(2)])

                    U = assemble(dirac(x-cur_ag_pos_)*zeta*dx(metadata = {"quadrature_degree": 20}))
                    V = assemble(dirac(x-cur_ag_pos_)*dx(metadata = {"quadrature_degree": 20}))
                    DU = np.array([assemble(diff(dirac(x-cur_ag_pos_)*zeta*dx(metadata = {"quadrature_degree": 20}),
                                                 cur_ag_pos_[l])) for l in range(2)])
                    DV = np.array([assemble(diff(dirac(x-cur_ag_pos_)*dx(metadata = {"quadrature_degree": 20}),
                                                 cur_ag_pos_[l])) for l in range(2)])
                    
                    deriv = (V*DU - U*DV) / (V*U)
                    
                    if barrier_function == 0:
                        rhs_v -= tau*gamma*deriv
                    elif barrier_function == 1:
                        print("ERROR: Not implemented yet")
                        exit(0)                        
                else:
                    val = zeta(cur_ag_pos)
                    Dval = grad_zeta(cur_ag_pos)

                    if barrier_function == 0:
                        rhs_v-= (tau/val)*gamma*Dval
                    elif barrier_function == 1:
                        rhs_v -= tau/gamma*np.maximum(0,0.1-val)*Dval

            if k <= N-1:                        
                rhs_v-= tau*np.array([assemble(diff(Flux(rho_new, beta_new, n)*jump(p_old)*dS(metadata = {"quadrature_degree": 5}), ii)) \
                                              for ii in ag_pos_new_[i,:]])        
                
            if with_regularized_dirac:
                
                cur_ag_pos_ = as_vector([variable(Constant(ag_poss[i,k,l])) for l in range(2)])
                
                U = assemble(dirac(x-cur_ag_pos_)*rho_new*dx(metadata = {"quadrature_degree": 20}))
                V = assemble(dirac(x-cur_ag_pos_)*dx(metadata = {"quadrature_degree": 20}))
                                   
                grad_vel = np.array([(V*assemble(diff(dirac(x-cur_ag_pos_)*rho_new*dx(metadata = {"quadrature_degree": 20}), cur_ag_pos_[l])) - U*assemble(diff(dirac(x-cur_ag_pos_)*dx(metadata = {"quadrature_degree": 20}), cur_ag_pos_[l]))) / V**2
                                     for l in range(2)])
                lhs_v = np.identity(2) + tau*np.outer(grad_vel, us[i,k,:])
            else:
                lhs_v = np.identity(2) - tau*Df(rho_at_ag_pos[i]) * \
                        np.outer(grad_rho_at_ag_pos[i,:], us[i,k,:])

            # SIMPLIFIED VERSION
            # lhs_v = np.identity(2)
            
            v[i,:] = np.linalg.solve(lhs_v, rhs_v)
            vs[i,k,:] = v[i,:]

        ####################
        # Adjoint eq. for q
        ####################

        if k <= N-1:
            # Derivative of the phi eq \ 
            a_q = delta_1 * inner(grad(q_), grad(z))*dx + 2*inner(grad(phi_new), grad(z))*q_*dx

            # Derivative of rho equation wrt. phi
            # rhs_q = -tau*derivative(rho_old*inner(beta_new, grad(p))*dx, phi_new, z)
            rhs_q = -tau*derivative(Flux(rho_new, beta_new, n)*jump(p_old)*dS(metadata = {"quadrature_degree": 5})
, phi_new, z)
            
            # rhs_q -= tau*derivative(0.5*jump(rho_old)*ufl_abs(inner(avg(beta_new),n('+')))*jump(p)*dS, phi_new, z)
            # rhs_q += tau*derivative(avg(rho_old)*inner(avg(beta_new),jump(p,n))*dS, phi_new, z)
            
            solve(a_q == rhs_q, q, bc_phi)

        
        ####################################################################
        # Adjoint eq. for p (Compute p^n by differentiating for rho^{n+1}) #
        ####################################################################

        # time derivative
        a_p = w*p_*dx

        # Diffusion part
        a_p += tau*epsilon * inner(grad(w), grad(p_))*dx\
	       - tau*epsilon * inner(avg(grad(w)),jump(p_,n))*dS \
	       - tau*epsilon * inner(avg(grad(p_)),jump(w,n))*dS \
	       + tau*(penalty/avg(h)) * epsilon * jump(w)*jump(p_)*dS \
               + tau*outflow_vel*w*p_*dss(1)

        # For higher-order elements
        # a_p += tau*0.5*derivative(jump(rho_old)*ufl_abs(inner(avg(beta_new),n('+')))*jump(p_)*dS, rho_old, w)
        # a_p -= tau*derivative(avg(rho_old)*inner(avg(beta_new),jump(p_,n))*dS, rho_old, w)
                        
        # Tracking term
        if k == 0:
            rhs_p = Constant(0.)*w*dx
        else:
            rhs_p = tau*Constant(np.exp(k*tau/T))*w*dx(1)

        # time derivate
        rhs_p += w*p_old*dx
        
        if k <= N-1:
            # Transport terms
            rhs_p -= tau*derivative(Flux(rho_new, beta_new, n)*jump(p_old)*dS(metadata = {"quadrature_degree": 5}), rho_new, w)
            
        # Derivative of phi eq
        rhs_p += derivative((1.0/(f(rho_new)**2 + delta_2))*q*dx, rho_new, w)

        # Derivative of the ODE part w.r.t. rho
        if with_regularized_dirac:
            for i in range(nr_agents):
                ag_point_ = as_vector([Constant(ag_pos_new[i,k]) for k in range(2)])
                denominator = assemble(dirac(x-ag_point_)*dx(metadata = {"quadrature_degree": 20}))


                # Comment out for simplified version (rho not entering in ODE)
                rhs_p -= tau/Constant(denominator)*dirac(x-ag_point_)*w*Constant(np.dot(us[i,k,:],vs[i,k,:]))*dx(metadata = {"quadrature_degree": 20})            
        else:
            Rhs_c = assemble(Constant(0.0)*z*dx)
            for i in range(nr_agents):
                
                ag_point = Point(ag_pos_new[i,:])
                if PointInside(ag_point):
                    ps = PointSource(P, ag_point, \
                                     -tau*np.dot(us[i,k,:],vs[i,k,:])*Df(rho_new))
                
                if k > 0:
                    ps.apply(Rhs_c)                    
        
        solve(a_p == rhs_p, p)
        adjs[k,:] = p.vector()[:]
        
        # Compute derivatives w.r.t the intensities c[i,k]
        for i in range(nr_agents):
            #ints[i,k] = assemble(diff(rho_new*inner(beta_new, grad(p_old))*dx, intensity_new_[i]))
            if k <= N-1:
                ints[i,k] = -assemble(diff(Flux(rho_new, beta_new, n)*jump(p_old)*dS(metadata = {"quadrature_degree": 5}), intensity_new_[i]))
                
    return vs,ints
