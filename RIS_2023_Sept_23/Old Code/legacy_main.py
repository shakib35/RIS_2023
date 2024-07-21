
import cvxpy as cvx
from RIS_Channel_Generation import *
from Optimize_RIS_Elements import HarvestingElementOptimizer

'''
Commands to install required libraries:
pip install cvxpy
pip install mosek
pip install scs
pip install cvxopt
pip install pyscipopt 

'''

# cvxpy is an embedded modeling language for convex optimization problems
# https://www.cvxpy.org/tutorial/index.html
# https://www.cvxpy.org/api_reference/cvxpy.html
# https://ajfriendcvxpy.readthedocs.io/en/latest/tutorial/functions/index.html
# http://web.cvxr.com/cvx/doc/basics.html
# https://stanford.edu/~boyd/papers/cvx_short_course.html
# https://www.solver.com/defining-constraints

eta_RIS = 0.8  # RF energy conversion efficiency
numAntennasPS = 10  # number of antennas at power station (PS)
numAntennasAP = 10  # number of antennas at access point (AP)
numRISelements = 16  # temporary number of RIS elements
numIoTs = 4  # number of IoTs in the system
# vec_theta_ = rand(1,N)*pi/2

print('eta_RIS: ', eta_RIS)
print('numAntennasPS: ', numAntennasPS)
print('numAntennasAP: ', numAntennasAP)
print('numRISelements: ', numRISelements)
print('numIoTs: ', numIoTs)

# PowerStation Power
PowerStation_dBm = 30
PowerStation_lin = PowerStation_lin(PowerStation_dBm, eta_RIS)

# noise power (sigma)
noisePower_dB = -100
noise_power = noise_power(noisePower_dB)

# Locations on X-Y coordinate (in meters)
PS_loc = [-10.000001, 0.000001]  # [-100, 30]
AP_loc = [ 10.000001, 0.000001]  # [100, 50]
RIS_loc = [-2, 6]  # [0, 100] 

# Devices random angles
PS_angle = rand(1, 1)
AP_angle = rand(1, 1)
RIS_angle = rand(1, 1)
print('\nPS_angle: ', PS_angle)
print('AP_angle: ', AP_angle)
print('RIS_angle: ', RIS_angle)


# channel generation Rician factor
# https://www.mathworks.com/help/comm/ug/fading-channels.html
eb = 10
eb1, eb2 = ricianChannel(eb)
print('\nRician channel:\neb1:', eb1, '\neb2', eb2)

# Generate IoT locations
center = [0, 0]  # [10, 20]
radius = 4
IoT_loc = generateIoTLocations(center, radius, numIoTs)

print('\nIoT locations:\n', IoT_loc)
# Plotting the system
#plotMap(IoT_loc, center, radius, RIS_loc, PS_loc, AP_loc)

# Uniform Linear Array for RIS, AP, and PS: precalculated
ULA_RIS = ULA_fun(RIS_angle, numRISelements)  # [vector] (constant. real)
ULA_AP = ULA_fun(AP_angle, numAntennasAP)  # [vector] (constant. real)
ULA_PS = ULA_fun(PS_angle, numAntennasPS)  # [vector] (constant. real)

# Populating channels related to RIS, AP, and PS, and IoTs
# LOS link: RIS to AP (H channel matrix value)
H = ch_RIS_to_AP(numRISelements, numAntennasAP, RIS_loc, AP_loc, eb1, eb2, ULA_RIS, ULA_AP)

# LOS link: PS to RIS (G channel matrix)
G = ch_PS_to_RIS(numAntennasPS, numRISelements, PS_loc, RIS_loc, eb1, eb2, ULA_RIS, ULA_PS)

# NLoS link: PS to IoTs g_d,k
g_d = ch_PS_to_IoTs(numIoTs, numAntennasPS, PS_loc, IoT_loc)

# NLOS link: IoT_i to AP h_d,k
h_d = ch_IoTs_to_AP(numAntennasAP, numIoTs, AP_loc, IoT_loc)

# g_r,k channel from RIS to IoTs
g_r = ch_RIS_to_IoTs(numRISelements, numIoTs, RIS_loc, IoT_loc,eb1, eb2, ULA_RIS )

# LOS link: IoTs to RIS (h_r,k channel from RIS to IoT_i) Ricing fading channel
h_r = ch_IoTs_to_RIS(numRISelements, numIoTs, RIS_loc, IoT_loc,eb1, eb2, ULA_RIS)


contributing_channel = np.sort(abs(G) ** 2, axis=0)  # I don't know why this is sorted

E_RIS = eta_RIS * PowerStation_lin * sum(contributing_channel[0:int(numRISelements / 2)])  # Energy harvested
E_RIS_TH = 1e-6  # Energy harvested threshold

# This is the
#tau_0, Ah, A_h_elements = HarvestingElementOptimizer(G, E_RIS_TH, eta_RIS, PowerStation_lin)
A_r = numRISelements
Ah = numRISelements
g_r = g_r[:A_r, :]
G = G[:A_r, :]


'''This is the CVX Solver Portion of the Program'''
# Initialize Omega (multiple receiver vectors)
omega = array(rand(numAntennasAP, numIoTs) + 1j * rand(numAntennasAP, numIoTs), dtype=complex)

# single receiver vector w:
w = array(rand(numAntennasAP, 1) + 1j * rand(numAntennasAP, 1), dtype=complex)
w_H = w.T

# I think these need to be variables.
# Never mind, converting these to variables breaks DCP compliance
p1_next = 0.9 * ones((numIoTs, 1))  # denoted with p1_i^(n-1)
q1_next = 0.9 * ones((numIoTs, 1))  # denoted with q1_i^(n-1)
p2_next = 0.9 * ones((numIoTs, 1))  # denoted with p2_i^(n-1)
q2_next = 0.9 * ones((numIoTs, 1))  # denoted with q2_i^(n-1)
# p3_next = ones((numIoTs, 1))                                 # unused
# q3_next = ones((numIoTs, 1))                                 # unused
# tau_k_next = 0.05*ones((numIoTs,1))                          # unused
# b_k_next = 0.05*ones((numIoTs,1))                            # unused
# Z_k_next = 0.05*ones((numIoTs,1))                            # unused

b_next = 0.9 * ones((numIoTs, 1))  # denoted with B_i^(n-1)
z_next = 0.9 * ones((numIoTs, 1))  # denoted with z_i^(n-1)
p_next = 0.9 * ones((numIoTs, 1))  # denoted with p_i^(n-1)
w_next = omega
# w_next = rand(numAntennasAP,numIoTs) + sqrt(-1)*rand(numAntennasAP,numIoTs);

constraints = []
u = cvx.Variable(shape=(numIoTs - 1, 1), pos=True, name="u")
t = cvx.Variable(shape=(numIoTs, 1), pos=True, name="t")
z = cvx.Variable(shape=(numIoTs, 1), pos=True, name="z")
b = cvx.Variable(shape=(numIoTs, 1), pos=True, name="b")
p = cvx.Variable(shape=(numIoTs, 1), pos=True, name="p")
# constraints = [ z>=0, u>=0, t>=0, p>=0, b>=0 ]
constraints += [
    cvx.max(b) <= cvx.Constant(1)
]

# RIS reflection coefficients during IoT harvesting
vr = cvx.Variable((A_r, 1), complex=True, name="vr") # complex=True,
# RIS reflection coefficients during IoT transmission
v = cvx.Variable((numRISelements, 1), complex=True, name="v") # complex=True,

# Constrain v_r and v to stay between => e^(theta), theta = [0, 2*pi)
#
# -> basically v and v_r stay between [0,1)
constraints += [cvx.max(cvx.real(vr)) <= cvx.Constant(1),
                cvx.min(cvx.real(vr)) >= cvx.Constant(0),
                cvx.max(cvx.imag(vr)) <= cvx.Constant(1),
                cvx.min(cvx.imag(vr)) >= cvx.Constant(0),
                cvx.max(cvx.real(v)) <= cvx.Constant(1),
                cvx.min(cvx.real(v)) >= cvx.Constant(0),
                cvx.max(cvx.imag(v)) <= cvx.Constant(1),
                cvx.min(cvx.imag(v)) >= cvx.Constant(0),
                ]

# RIS reflection coefficients during IoT harvesting REAL
w_re = cvx.Variable((numAntennasAP, numIoTs), pos=True, name="w_re")
# RIS reflection coefficients during IoT harvesting IMAGINARY
w_im = cvx.Variable((numAntennasAP, numIoTs), pos=True, name="w_im")

constraints += [
    #cvx.max(w_re) <= 100,
    #cvx.min(w_re) >= 0,
    #cvx.max(w_im) <= 100,
    #cvx.min(w_im) <= 0,
    ]


p1 = cvx.Variable((numIoTs, 1), pos=True, name="p1")
q1 = cvx.Variable((numIoTs, 1), pos=True, name="q1")
p2 = cvx.Variable((numIoTs, 1), pos=True, name="p2")
q2 = cvx.Variable((numIoTs, 1), pos=True, name="q2")
# p3 = cvx.Variable((numIoTs,1), name="p3")
# q3 = cvx.Variable((numIoTs,1), name="q3")

# Constrain p1, q1, p2, q2 to [0, 1] to maintain accuracy within Taylor Polynomials radius of convergence
constraints += [cvx.max(p1) <= cvx.Constant(1),
                # cvx.min(p1) >= cvx.Constant(0),
                cvx.max(q1) <= cvx.Constant(1),
                # cvx.min(q1) >= cvx.Constant(0),
                cvx.max(p2) <= cvx.Constant(1),
                # cvx.min(p2) >= cvx.Constant(0),
                cvx.max(q2) <= cvx.Constant(1),
                # cvx.min(q2) >= cvx.Constant(0),
                ]

tau_0 = 1  # 0.5
tau_1 = 1  # 1-tau_0
# PowerStation_lin = cvx.Parameter(name="PowerStation_lin")
# noise_power = cvx.Parameter(name="noise_power")
eta_tau0_P0 = eta_RIS * tau_0 * PowerStation_lin
eta_tau0_P0_tilde = 1 / eta_tau0_P0



# Objective function Eq.(13a)
# T_max = cvx.Variable(shape=(numIoTs,1),   pos=True, name="T_max")
objective = cvx.Maximize(u[2, 0])

objective_constraints = [
    # cvx.norm(x: [vertical stack], p: type of norm) u[0] = t[0] * t[1]
    (cvx.norm(cvx.vstack([t[0, 0] - t[1, 0], 2 * u[0, 0]])) <= t[0, 0] + t[1, 0]),
    (cvx.norm(cvx.vstack([t[2, 0] - t[3, 0], 2 * u[1, 0]])) <= t[2, 0] + t[3, 0]),
    (cvx.norm(cvx.vstack([u[0, 0] - u[1, 0], 2 * u[2, 0]])) <= u[0, 0] + u[1, 0]),
    #u[2, 0] <= 100
]


'''
# Notes Taken From Chris's Paper:
maximize(z);
norm([b(1)-b(2); 2*z]) <= b(1)+b(2);   # z = t(1) * t(2)
t(1) >= b(1)-1;
t(2) >= b(2)-1;
b(1) >= 1;
b(2) >= 1;
t(1) >= 0;
t(2) >= 0;
'''

b_obj = cvx.Variable(shape=(numIoTs, 1), pos=True, name="b_obj")
_objective_constraints = [  # These are taken from Chris's paper. Unused.

    (cvx.norm(cvx.vstack([b_obj[0, 0] - b_obj[1, 0], 2 * u[0, 0]])) <= b_obj[0, 0] + b_obj[1, 0]),
    (cvx.norm(cvx.vstack([b_obj[2, 0] - b_obj[3, 0], 2 * u[1, 0]])) <= b_obj[2, 0] + b_obj[3, 0]),
    (cvx.norm(cvx.vstack([u[0, 0] - u[1, 0], 2 * u[2, 0]])) <= u[0, 0] + u[1, 0]),

    t[0, 0] >= b[0, 0] - 1,
    t[1, 0] >= b[1, 0] - 1,
    t[2, 0] >= b[2, 0] - 1,
    t[3, 0] >= b[3, 0] - 1,

    b_obj[0, 0] >= 1,
    b_obj[1, 0] >= 1,
    b_obj[2, 0] >= 1,
    b_obj[3, 0] >= 1,

    t[0, 0] >= 0,
    t[1, 0] >= 0,
    t[2, 0] >= 0,
    t[3, 0] >= 0,

    u[2, 0] <= 100000,

]

constraints += objective_constraints

for i in range(0, numIoTs):
    print('IoT #:', i)
    constraints += [
        # Eq 13(b) ==> (14)
        # try <= here
        # t[i,0] <= z_next[i,0]**(tau_1) + tau_1 * (z_next[i,0]**(1-tau_1)) * (z[i,0] - z_next[i,0]) # [correct]

        t[i, 0] >= z[i, 0] ** tau_1,
        z[i, 0] ** tau_1 <= z_next[i, 0] ** (tau_1) + tau_1 * (z_next[i, 0] ** (1 - tau_1)) * (z[i, 0] - z_next[i, 0])
    ]

    Phi = 1e8 * np.diag(h_r[:, i].T) @ H
    #Phi = np.diag(h_r[:, i].T) @ H

    Phi_H = Phi.T

    pq_1_complex = w_H @ (Phi_H @ v + h_d[:, [i]])
    constraints += [
        p1[i, 0] <= cvx.real(pq_1_complex),  # Line (17)
        q1[i, 0] <= cvx.imag(pq_1_complex),  # Line (18)
    ]

    # Line (23c) This is part of B) Sub problem-2
    '''constraints += [
        # Line (23c)
        (2*p_next[i,0]*cvx.real(w_next[:,i])*(p[i, 0]-p_next[i,0])+p_next[i, 0]**2*w_re[:,i] >= cvx.real(omega[:, i])),
        (2*p_next[i,0]*cvx.imag(w_next[:,i])*(p[i, 0]-p_next[i,0])+p_next[i, 0]**2*w_im[:,i] >= cvx.imag(omega[:, i])),
    ]'''

    # Line (19) - (expression 13c): LHS Eq(19) >= RHS Eq(15):
    constraints += [
        2 * (p1_next[i, 0] / b_next[i, 0]) * (p1[i, 0] - p1_next[i, 0]) + 2 * (q1_next[i, 0] / b_next[i, 0]) * (
                    q1[i, 0] - q1_next[i, 0]) \
        + ((p1_next[i, 0] ** 2 + q1_next[i, 0] ** 2) / b_next[i, 0]) * (1 - (b[i, 0] - b_next[i, 0]) / b_next[i, 0]) \
        >= (1 / p_next[i, 0]) * (z[i, 0] - z_next[i, 0] + (p[i, 0] * (z_next[i, 0] - 1)) / (p_next[i, 0]))
    ]

    # z[i,0]-1
    # (1/p_next[i,0]) * (z[i,0] - z_next[i,0] + (p[i,0] * (z_next[i,0]-1))/(p_next[i,0]))
    # z(i)-1;
    # replace with RHS of Eq15:
    # 1/p_next(i)*(z(i)-z_next(i)+(z_next(i)-1)/(p_next(i)*p(i))

    # Eq (13d) accounting for the interference terms
    interference_terms = []
    # returns list [IOT indices >  current loop index] that have not been looped through yet

    for j in range(i + 1, numIoTs):  # for j = 1:length(not_i)
        # print('interference_j:', j)
        Phi_j = 1e8*np.diag(h_r[:, j].T) @ H
        # Phi_j = np.diag(h_r[:, j].T) @ H

        interference_terms += [omega[:, [j]].T @ (Phi_j.T @ v + h_d[:, [j]])]  # resolves to a shape(1,1) scalar value
    #print('interference terms:', len(interference_terms))
    if len(interference_terms) < 1:
        interference_terms += [0]  # avoid calling cvx.vstack([]) with an empty list [ ]

    constraints += [
        cvx.norm(cvx.vstack([cvx.vstack([cvx.sqrt(noise_power)]), cvx.vstack(interference_terms)])) <= cvx.sqrt(b[i, 0])
    ]

    # Eq (20)
    # psi_i = 1e-8 * np.diag( g_r[i,:] ) @ G.T
    psi_i = 1e8 * np.diag(g_r[i, :]) @ G.T
    # psi_i = np.diag(g_r[i, :]) @ G.T

    # print('\n',vr.T.shape, psi_i.shape)
    pq_2_complex = vr.T @ psi_i + g_d[[i], :]  # Incompatible dimensions (1, 17) (16, 10)

    constraints += [
        p2[i] <= cvx.real(pq_2_complex),
        q2[i] <= cvx.imag(pq_2_complex),

        # Eq 22
        # p2_next[i,0]*(p2[i,0] - p2_next[i,0]) + q2_next[i,0]*(q2[i,0] - q2_next[i,0]) >= 1e-6*p[i,0] * eta_tau0_P0_tilde #1/(eta_tau0_P0)
        p2_next[i, 0] * (p2[i, 0] - p2_next[i, 0]) + q2_next[i, 0] * (q2[i, 0] - q2_next[i, 0]) >= p[i, 0] * eta_tau0_P0_tilde  # 1/(eta_tau0_P0)

    ]

problem = cvx.Problem(objective, constraints)

print('\nCompliance (all constraints meet these):')
print('DCP :      ', problem.is_dcp())
print('DCP (DPP): ', problem.is_dcp(dpp=True))
print('DGP :      ', problem.is_dgp())
print('DGP (DPP): ', problem.is_dgp(dpp=True))
print('DPP :      ', problem.is_dpp())
print('DQCP:      ', problem.is_dqcp())
print('QP  :      ', problem.is_qp())
print(f'num constraints: {len(problem.constraints)}\n')


def SolveIt(problem):
    problem.solve(solver='SCS', verbose=True)
    # problem.solve(solver='SCS', qcp=True, verbose=True)

    print('status:', problem.status)
    print('optimal value:', problem.value)

    print('\nVariable Values:')
    for variable in problem.variables():
        # break
        print(f"{variable.name()}: {variable.shape} \n{variable.value}")


problem = cvx.Problem(objective, constraints)
SolveIt(problem)

# opt_iterations = 5
# problem.solve(solver='SCS', verbose=True)
print(f'\nObjective: Maximize(u[2,0]):\nValue: {problem.value}')
'''
for i in range(opt_iterations):

    p1_next = p1.value
    q1_next = q1.value
    p2_next = p2.value
    q2_next = q2.value
    b_next  = b.value
    z_next  = z.value
    p_next  = p.value
    problem = cvx.Problem( objective, constraints )
    problem.solve(solver='SCS', verbose=False)
    print(f'ITERATION: {i} - {problem.value}')
    print(f'p1: {p1.value.flatten()}\nq1: {q1.value.flatten()}\np2: {p2.value.flatten()}\nq2: {q2.value.flatten()}')

'''

'''
if __name__ == '__main__':
    print('PyCharm')'''
