import sys, os
import time as timer
import math
import re
import scipy.sparse as sp
from hughes_solution import *

# Advanced: uncomment to start with a previously computed control stored in file
# control_file = "output/control.csv"

set_log_level(LogLevel.WARNING)

# Initial guess for the control
u = example.ag_vel_0

# Is the intensity a control?
control_intensity = True

# Initialize controls: us[i,k,l]
# i...agent index,
# k...time step,
# l... agent movement directions
us = np.zeros((nr_agents,N+1,2))
cs = np.zeros((nr_agents,N+1))

print(nr_agents, "agents are configured")

# Copy initial guess for controls
if hasattr(example, "trajectory"):
        for i in range(nr_agents):
                traj = example.trajectory[i]
                for part in range(traj.shape[0]-1):
                        n_begin = int(traj[part,0])
                        n_end = int(traj[part+1,0])

                        for k in range(n_begin,n_end):
                                us[i,k,:] = traj[part,1:3]
                                cs[i,k] = 1.0
elif 'control_file' in vars():
        print("Read control from file")

        file = open(control_file, 'r')
        file.readline() # header

        k = 0
        for line in file:
                data = re.split(" |,|;|\t", line.strip())
                for i in range(nr_agents):
                        us[i,k,0] = float(data[4*i])
                        us[i,k,1] = float(data[4*i+1])
                        cs[i,k] = float(data[4*i+3])

                k += 1
                
else:
        print("Set default values for controls")
        for i in range(nr_agents):
                for k in range(N+1):
                        us[i,k,:] = [u[i,0],u[i,1]]
                        cs[i,k] = example.intensity*(1-k/N)

np.set_printoptions(threshold=sys.maxsize)                        

# exit(0)
# File output
res_file = open(output_basedir + '/result.dat', "w+")

# Assemble FD matrix
main_diag = 2*np.ones(N+1)
main_diag[0] = 1
main_diag[N] = 1
sub_diag = -np.ones(N+1)

# Matrix realizing the H1-inner product
A = 1/tau**2*sp.diags([sub_diag,main_diag,sub_diag],[-1,0,1],[N+1,N+1]) + sp.eye(N+1)

# Computes the function value of the reduced objective for a given control matrix
# and the corresponding density rho. The latter one can be computed with SolveForward
def ValueJ(us, cs, rhos, ag_poss):

        # Tracking term
        J_track = 0.
        for k in range(1,N+1):
                rho = Function(R)
                rho.vector().set_local(rhos[k,:])

                if objective_type == 0:
                        J_track += tau*np.exp(k*tau/T)*assemble(rho*dx(1))
                elif objective_type == 1:
                        mass = Constant(assemble(rho*dx))
                        barycenter = as_vector([Constant(assemble(1/mass*x[i]*rho*dx))\
                                                for i in range(2)])
                        J_track += 0.5*tau*assemble(1/mass*rho*inner(x-barycenter,x-barycenter)*dx)

        # Value of barrier function
        J_barrier = 0.
        for k in range(1,N+1):
                for i in range(nr_agents):
                        ag_pos = Point(ag_poss[i,k,:])

                        if not PointInside(ag_pos):

                                if(wall_region(ag_pos)):
                                        if barrier_function == 0:
                                                val = 0.
                                        elif barrier_function == 1:
                                                val = -np.Inf
                                                
                                        print("I am in the wall")

                                        # TODO: Uncomment
                                        # val = 1.
                                else:
                                        val = 0.
                                        
                        else:
                                if with_regularized_dirac:
                                        ag_pos_ = as_vector([Constant(ag_poss[i,k,l]) for l in range(2)])
                                        val = assemble(dirac(x-ag_pos_)*zeta*dx(metadata = {"quadrature_degree": 20}))/assemble(dirac(x-ag_pos_)*dx(metadata = {"quadrature_degree": 20}))
                                else:
                                        val = zeta(ag_pos)
                                        
                                if barrier_function == 1 and val < 0.1:
                                        print("I am close to a wall")                                

                        if barrier_function == 0:                        
                                J_barrier -= gamma*tau*np.log(val)
                        elif barrier_function == 1:
                                J_barrier += 0.5/gamma*tau*np.maximum(0,0.1-val)**2
        
        # Regularization term (TODO: Make it work for different reg. parameters)
        J_reg = 0.5*alpha_1*DotProduct(us,cs,us,cs)

        return [J_track + J_reg + J_barrier, J_track, J_reg, J_barrier]

# Computes the gradient of the reduced objective.
# Input: Control matrix, the states, computed by SolveForward and the adjoint states,
# computed by SolveBackward
def GradientJ(us, cs, rhos, ag_poss, vs, ds):
        
        Dj_u = np.zeros((nr_agents,N+1,2))
        Dj_c = np.zeros((nr_agents,N+1))
        
        Dj_norm = 0
        
        for i in range(nr_agents):

                # Gradient of regularization term
                Dj_u[i,:,:] = alpha_1*us[i,:,:]

                # Gradient of the term f(rho(xi,t))*lam_xi(t)*ui(t) (requires solution of a FD sytem)

                # Assemble rhs vector
                f_rho_v = np.zeros((N+1,2))
                z = np.zeros((N+1,2))

                for k in range(1,N+1): 
                        
                        rho_cur = Function(R)
                        rho_cur.vector().set_local(rhos[k,:])

                        if with_regularized_dirac:
                                ag_pos = ag_poss[i,k,:]
                                ag_point_ = as_vector([Constant(ag_pos[l]) for l in range(2)])
                                vel = 1.-assemble(dirac(x-ag_point_)*rho_cur*dx(metadata = {"quadrature_degree": 20}))/assemble(dirac(x-ag_point_)*dx(metadata = {"quadrature_degree": 20}))
                                f_rho_v[k,:] = vel*vs[i,k,:]
                        else:
                                ag_pos = Point(ag_poss[i,k,:])
                                if not PointInside(ag_pos):
                                        rho_at_ag_pos = 0.
                                else:
                                        rho_at_ag_pos = rho_cur(ag_pos)
                                        
                                # f_rho_v[k,:] = f(rho_at_ag_pos)*vs[i,k,:]
                                f_rho_v[k,:] = (1.-rho_at_ag_pos)*vs[i,k,:]
                        
                        # SIMPLIFIED VERSION
                        # f_rho_v[k,:] = vs[i,k,:]

                # L2 Gradient
                # z = f_rho_v
                
                # H1 Gradient
                z = sp.linalg.spsolve(A, f_rho_v)

                Dj_u[i,:,:] += z
                # Dj_u[i,0,:] = 0 # u0 does not enter in our system!

                if control_intensity:

                        # L2 gradient
                        # ds_grad = ds[i,:]
                        # H1 gradient
                        ds_grad = sp.linalg.spsolve(A, ds[i,:])
                        
                        Dj_c[i,:] = alpha_2*cs[i,:] + ds_grad

        Dj_norm = sqrt(DotProduct(Dj_u,Dj_c, Dj_u,Dj_c))
        print("Norm of gradient: ", Dj_norm)
        print("ds      =", np.linalg.norm(ds))
        print("ds_grad =", np.linalg.norm(ds_grad))
        return [Dj_u, Dj_c, Dj_norm]

s = 1.e-1

# Implements the H1 scalar product in time
def DotProduct(a_u,a_c,b_u,b_c):
        val = 0
        for i in range(nr_agents):

                # L2 part
                for k in range(N+1):
                        val += tau*np.dot(a_u[i,k,:],b_u[i,k,:])
                        if control_intensity:
                                val += tau*a_c[i,k] * b_c[i,k]

                # H1 part
                for k in range(N):
                        val += (1/tau)*np.dot(a_u[i,k+1,:]-a_u[i,k,:],b_u[i,k+1,:]-b_u[i,k,:])
                        if control_intensity:
                                val += (1/tau)*(a_c[i,k+1]-a_c[i,k]) * (b_c[i,k+1]-b_c[i,k])

        return val;

# Implements the H1 norm (in time) of the controls
def ControlNorm(a_u,a_c):
        return sqrt(DotProduct(a_u,a_c, a_u,a_c))

# Write agent movement to file
def WriteAgentPosition(ag_poss, fileprefix = "ag_pos"):

        
        for k in range(N):
                filename = output_basedir + '/' + fileprefix + "_" + str(k) + ".csv"
                ag_file = open(filename, "w")
                for i in range(nr_agents):
                        ag_file.write(" Ag%dx Ag%dy int%d" % (i,i,i))
                ag_file.write("\n")
        
                for i in range(nr_agents):
                        output = " %f %f %f" % (ag_poss[i,k,0], ag_poss[i,k,1], cs[i,min(k,N-2)])
                        ag_file.write(output)
                        
                ag_file.write("\n")
                ag_file.close()

def WriteAgentMovement(us, cs, fileprefix = "ag_mov"):
        ag_file = open(output_basedir + '/' + fileprefix + '.csv', "w")
        for i in range(nr_agents):
                ag_file.write(" Ag_x_"+str(i)+" Ag_y_"+str(i)+" Ag_mag_"+str(i)+" Ag_int_"+str(i))
        ag_file.write("\n")

        for k in range(N+1):
                for i in range(nr_agents):
                        vel = sqrt(us[i,k,0]**2 + us[i,k,1]**2)
                        ag_file.write(" "+str(us[i,k,0]))
                        ag_file.write(" "+str(us[i,k,1]))
                        ag_file.write(" "+str(vel))
                        ag_file.write(" "+str(cs[i,k]))
                ag_file.write("\n")
                
        ag_file.close()
                
                
# Plot Gradient
def WriteGradient(ag_poss, Dj_u):
        grad_file = open(output_basedir + '/gradient.dat', "w")
        scale = np.zeros((nr_agents));

        for i in range(nr_agents):
                scale[i] = np.max(np.linalg.norm(Dj_u[i,:,:],axis=1))
                
        for k in range(N):
                for i in range(nr_agents):                        
                        grad_file.write("%g, %g, %g, %g, " % \
                                        (ag_poss[i,k,0], ag_poss[i,k,1], \
                                         -Dj_u[i,k,0]/scale[i], \
                                         -Dj_u[i,k,1]/scale[i]))
                grad_file.write("\n")
        grad_file.close()

# Write evolution of objective to file
def WriteObjective(rhos):
        objective_file = open(output_basedir + '/objective.dat', 'w')
        
        for k in range(1,N+1):
                rho = Function(R)
                rho.vector().set_local(rhos[k,:])

                J_track = assemble(rho*dx(1))                

                objective_file.write(str(J_track) + "\n")

        objective_file.close()                

# Method performing a gradient test by comparing gradient obtained by adjoint approach
# with a sequence of finite difference quotients
def GradientTest(us, cs):
        
        direction_u = np.zeros_like(us)
        direction_u[0,:,0] = 1.
        direction_u[0,:,1] = 0.1
        #direction_u[1,:,0] = 1.
        #direction_u[1,:,1] = 0.

        direction_c = -np.ones_like(cs)        
        #direction_c = 0.5*(np.random.rand(nr_agents,N)-0.5)

        # We need the normed direction
        dir_norm = ControlNorm(direction_u, direction_c)
        
        print("Norm of direction", dir_norm)
        
        direction_u = (1/dir_norm)*direction_u
        direction_c = (1/dir_norm)*direction_c

        print("Norm of direction", ControlNorm(direction_u, direction_c))        
        
        fd_file = open("fd_test.dat", "w")

        for j in range(floor(-math.log(1.e-14,2))):
                h = 2**(-j)
                
                us_h = us + h*direction_u
                cs_h = cs + h*direction_c
                                
                [rhos_h, _, ag_poss_h] = SolveForward(us_h, cs_h)
                [J_h,_,_,_] = ValueJ(us_h, cs_h, rhos_h, ag_poss_h)
                print("Function value J(u+h): ", J_h)

                fd_grad = 1/h*(J_h-J)                        
                ex_grad = DotProduct(Dj_u,Dj_c, direction_u,direction_c)
                                
                difference = (fd_grad - ex_grad)

                fd_file.write("%g %g\n" % (h,abs(difference/ex_grad)))
                
                print("h=", h, " diff:", difference, " (rel: ", difference/ex_grad, ") fd_grad", fd_grad, " ex_grad", ex_grad)
        sys.exit()

# Project the control onto the set of admissible controls
def Project(zs, cs):
        print("Evaluating projection operator")
        
        # L2 projection (pointwise)
        WriteAgentMovement(zs, cs, "before_project")
        
        us = np.empty_like(zs)

        norm_zs = np.linalg.norm(zs, axis=2)
        norm_zs[norm_zs < 1] = 1.0
        us = (1 / norm_zs[...,np.newaxis]) * zs

        ds = np.maximum(0., cs)

        # Uncomment this line to use L2-projection
        # In this case the inner product has to be changed as well
        # return us,cs

        # H1 projection (solve optimization problem with semismooth Newton)        
        for i in range(nr_agents):
                print("Projecting movement of agent ", i)
                z = zs[i,:,:];

                # Initial guess
                u = np.copy(z)
                #norm_u = np.linalg.norm(u, axis=1)
                #u = (1./norm_u[...,np.newaxis]) * u
                mu = np.zeros((N+1,))

                converged = False
                Iter = 0
                MaxIter = 100
                while not(Iter >= MaxIter or converged):
                        norm_u = np.linalg.norm(u, axis=1)
                        chi = np.zeros((N+1,))
                        chi[0.5*(norm_u**2-1) + mu > 0] = 1.0
                        
                        I_A = sp.dia_matrix((chi,0),(N+1,N+1))
                        I_I = sp.eye(N+1) - I_A
                                                
                        U0 = sp.dia_matrix((u[:,0], 0),(N+1,N+1))
                        U1 = sp.dia_matrix((u[:,1], 0),(N+1,N+1))
                        Mu = sp.dia_matrix((mu, 0),(N+1,N+1))
                        
                        F1 = A.dot(u[:,0] - z[:,0]) + Mu.dot(u[:,0])
                        F2 = A.dot(u[:,1] - z[:,1]) + Mu.dot(u[:,1])
                        F3 = mu - np.maximum(0, 0.5*(norm_u**2-1) + mu)
                        
                        F = np.hstack((F1,F2,F3))

                        resid = np.linalg.norm(F)
                        print("norm of resid: %.2E" % resid)

                        if resid < 1.e-11:                                
                                converged = True
                                continue
                        
                        DF = sp.bmat([[A+Mu, None, U0],
                                      [None, A+Mu, U1],
                                      [-I_A.multiply(U0), -I_A.multiply(U1), I_I]])

                        delta = sp.linalg.spsolve(DF, -F)

                        delta_u0 = delta[0:N+1]
                        delta_u1 = delta[N+1:2*(N+1)]
                        delta_mu = delta[2*(N+1):3*(N+1)]
                        
                        u[:,0] = u[:,0] + delta_u0
                        u[:,1] = u[:,1] + delta_u1
                        mu = mu + delta_mu                

                        Iter += 1

                if Iter >= MaxIter:
                        print("ERROR: Semismooth Newton for the projection did not converge")
                        exit(0)
                        
                us[i,:,:] = u

                # Project intensities
                print("Projecting intensity of agent ", i)
                
                d = ds[i,:]
                mu = A.dot(d-cs[i,:])
                
                converged = False
                Iter = 0
                while not(Iter >= MaxIter or converged):
                        chi = np.zeros((N+1,))
                        chi[mu - d > 0] = 1.0
                        
                        I_A = sp.dia_matrix((chi,0),(N+1,N+1))
                        I_I = sp.eye(N+1) - I_A
                        
                        F = np.hstack((np.zeros(N+1), I_I.dot(mu) + I_A.dot(d)))
                        resid = np.linalg.norm(F)
                        print("norm of resid: %.2E" % resid)
                        
                        if resid < 1.e-12:                                
                                converged = True
                                continue

                        DF = sp.bmat([[A, -sp.eye(N+1)],
                                      [I_A, I_I]])
                        
                        delta = sp.linalg.spsolve(DF, -F)
                        delta_d = delta[0:N+1]
                        delta_mu = delta[N+1: 2*(N+1)]
                        
                        d = d + delta_d
                        mu = mu + delta_mu
                        
                        Iter += 1
                        
                if Iter >= MaxIter:
                        print("ERROR: Semismooth Newton for the projection of intensity did not converge")
                        exit(0)

                ds[i,:] = d
                        
        WriteAgentMovement(us, ds, "after_project")
        return us,ds
        
##############################
# MAIN program starts here
#############################

# Project initial guess onto the set of admissible controls
us,cs = Project(us,cs)
Dj_norm = 1.
proj_grad_norm = 1.

# Loop of GRADIENT METHOD
for Iter in range(2000):
        print("=================================")
        print("Gradient method, iteration", Iter)
        print("=================================")
        
        # Solve forward equation
        if Iter == 0:
                [rhos, phis, ag_poss] = SolveForward(us, cs)
        else:
                # If already computed during Armijo line search do not recompute
                
                rhos = rhos_new
                phis = phis_new
                ag_poss = ag_poss_new

        # Write current iterate to files
        if verbosity > 1:
                print("  writing output files")
                
        # plot_phi(phis,ag_poss,cs)
        if True: # Iter % 50 == 0:
            plot_rho(rhos)
            WriteAgentPosition(ag_poss)

        WriteAgentMovement(us, cs)
        
        # Compute Functional Value
        print("J in gradient loop")

        [J, J_track, J_reg, J_barrier] = ValueJ(us, cs, rhos, ag_poss)

        # Compute number of active points
        us_norm = np.linalg.norm(us,axis=2)
        N_active = np.sum(us_norm >= 1, axis=1)
        N_inactive = N-N_active        

        # Console output
        print("  function value        : ", J)
        print("  - tracking term       : ", J_track)
        print("  - regularization term : ", J_reg)
        print("  - barrier term        : ", J_barrier)
        
        for i in range(nr_agents):
                print("  active time steps: ", N_active[i])

        WriteObjective(rhos)
                
        # If there are no agents we only want to solve the forward equation once
        if nr_agents == 0:
                print("There are no agents configured.")
                sys.exit()
        
        # Solve backward equation
        [vs, ds] = SolveBackward(us, cs, rhos, phis, ag_poss)
        
        # Compute Gradient       
        Dj_norm_old = Dj_norm
        [Dj_u, Dj_c, Dj_norm] = GradientJ(us, cs, rhos, ag_poss, vs, ds)
        
        # Perform a gradient
        # GradientTest(us, cs)
                
        proj_grad_norm_old = proj_grad_norm
        [d_us,d_cs] = Project(us - Dj_u, cs - Dj_c)
        proj_grad_norm = ControlNorm(us - d_us, cs - d_cs)

        # Check stopping criterion
        print("  norm of projected gradient: ", proj_grad_norm)
        if proj_grad_norm < 1.e-8:
                print("Stopping criterion fulfilled.")
                break;

        # Write iteration status to results file
        res_file.write("%d, %e, %e, %e, %e\n" % (Iter, J, J_track, J_reg, proj_grad_norm))
        res_file.flush()

        # Write gradient to file
        WriteGradient(ag_poss, Dj_u)
        
        # Update the iterate with Armijo rule
        Armijo = False
        sigma = 10**(-2)
        
        while not Armijo:
                # Compute new iterate
                us_new = us - s * Dj_u
                cs_new = cs - s * Dj_c

                # Projection step (due to projected gradient method)
                WriteAgentMovement(us_new, fileprefix = "us_before_")
                us_new_proj,cs_new_proj = Project(us_new, cs_new)
                WriteAgentMovement(us_new_proj, fileprefix = "us_after")

                # Solve forward system for new iterate
                [rhos_new, phis_new, ag_poss_new] = SolveForward(us_new_proj, cs_new_proj)

                # Compute functional value in new iterate
                [J_new, J_track_new, J_reg_new, J_barrier_new] = ValueJ(us_new_proj,
                                                                        cs_new_proj,
                                                                        rhos_new,
                                                                        ag_poss_new)
                # Pint functional value
                print("  function value        : ", J_new)
                print("  - tracking term       : ", J_track_new)
                print("  - regularization term : ", J_reg_new)
                print("  - barrier term        : ", J_barrier_new)

                # TODO: This stopping criterion is not good but it works
                if J_new < J:
                # if J_new < J - sigma*s*Dj_norm**2:
                        # Armijo condition fulfilled

                        Armijo = True
                        us = us_new_proj
                        cs = cs_new_proj

                        print("  Armijo condition satisfied for s =", s)
                        print("  decreased function value by", J-J_new)                       
                        if Iter>0:
                            s *= min(1.5, 1.1*max(1,proj_grad_norm_old**2/proj_grad_norm**2))

                elif s < 1.e-5 and J_new < 100:
                        
                        # Armijo ended in infinite loop. Trying to rescue the computation                        
                        print("I think our gradient is not a descent direction.")
                        print("Norm of gradient is", Dj_norm, ".")
                        print("I will accept the next step and increase Armijo parameter.")
                        s *= 10

                        Armijo = True
                        us = us_new_proj
                        cs = cs_new_proj                       
                else:
                        # Armijo condition violated                        
                        print("  Armijo not satisfied for s = ", s, ": J_new =", J_new, ", J =", J)
                        s *= 0.5
                        print("  repeat for s =", s)

res_file.close()
