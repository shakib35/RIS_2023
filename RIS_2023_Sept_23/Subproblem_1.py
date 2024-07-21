from RIS_Channel_Generation import RISNomaSystem
from Optimize_RIS_Elements import HarvestingElementOptimizer
import numpy as np
from numpy import sqrt, pi, sin, cos, multiply, linspace, ones, zeros, \
    log10, transpose, exp, arange, concatenate, abs, sort, \
    sum, array, argsort, diag, matrix, argwhere, ndarray
from numpy.random import rand
import cvxpy as cvx


def Sub_Problem_1(RIS_System, next_values):
    """
        This is just the part of the Paper pertaining to Sub Problem 1

    """
    A_r = RIS_System.A_r
    E_RIS_TH = RIS_System.E_RIS_TH
    eta_RIS = RIS_System.eta_RIS
    PowerStation_lin = RIS_System.PowerStation_lin
    numAntennasAP = RIS_System.numAntennasAP
    numRISelements = RIS_System.numRISelements
    numIoTs = RIS_System.numIoTs
    noise_power = RIS_System.noise_power
    Phi = RIS_System.Phi
    Psi = RIS_System.Psi
    h_d = RIS_System.h_d
    g_d = RIS_System.g_d
    tau_0 = RIS_System.tau_0
    tau_1 = RIS_System.tau_1


    p1_next = next_values['p1_next']
    q1_next = next_values['q1_next']
    p2_next = next_values['p2_next']
    q2_next = next_values['q2_next']
    b_next = next_values['b_next']
    z_next = next_values['z_next']
    p_next = next_values['p_next']
    w = next_values['omega']
    w_H = w.T

    _constraints = []
    constraints = []

    v_r = cvx.Variable((A_r, 1), complex=True, name="v_r")  # RIS reflection coefficients during IoT harvesting
    v = cvx.Variable((numRISelements, 1), complex=True, name="v")  # RIS reflection coefficients during IoT transmission

    # Constrain v_r and v to stay between e^(j*theta), theta = [0, 2*pi)
    # v and v_r are complex unit vector arrays: e.g., e^(j*theta) = cos(theta) + j*sin(theta)
    constraints += [
        cvx.norm(v_r) ** 2 <= cvx.Constant(A_r),
        cvx.norm(v) ** 2 <= cvx.Constant(numRISelements),
    ]

    _constraints += [
        cvx.max(cvx.real(v_r)) <= cvx.Constant(1),
        cvx.min(cvx.real(v_r)) >= cvx.Constant(-1),

        cvx.max(cvx.imag(v_r)) <= cvx.Constant(1),
        cvx.min(cvx.imag(v_r)) >= cvx.Constant(-1),

        cvx.max(cvx.real(v)) <= cvx.Constant(1),
        cvx.min(cvx.real(v)) >= cvx.Constant(-1),

        cvx.max(cvx.imag(v)) <= cvx.Constant(1),
        cvx.min(cvx.imag(v)) >= cvx.Constant(-1),
    ]

    u = cvx.Variable(shape=(numIoTs - 1, 1), pos=True, name="u")
    t = cvx.Variable(shape=(numIoTs, 1), pos=True, name="t")
    z = cvx.Variable(shape=(numIoTs, 1), pos=True, name="z")
    b = cvx.Variable(shape=(numIoTs, 1), pos=True, name="b")
    p = cvx.Variable(shape=(numIoTs, 1), pos=True, name="p")
    p1 = cvx.Variable((numIoTs, 1), pos=True, name="p1")
    q1 = cvx.Variable((numIoTs, 1), pos=True, name="q1")
    p2 = cvx.Variable((numIoTs, 1), pos=True, name="p2")
    q2 = cvx.Variable((numIoTs, 1), pos=True, name="q2")
    # p3 = cvx.Variable((numIoTs,1), name="p3")
    # q3 = cvx.Variable((numIoTs,1), name="q3")

    # constrain the power p_i to be greater than 0.001
    constraints += [
        cvx.min(p) >= 0.001
    ]

    # Constrain p1, q1, p2, q2 to [0, 1] to maintain accuracy within Taylor Polynomial radius of convergence
    constraints += [
        cvx.norm(cvx.vstack([p1, q1])) <= cvx.Constant(numIoTs),
        cvx.norm(cvx.vstack([p2, q2])) <= cvx.Constant(numIoTs),
        # cvx.max(b) <= cvx.Constant(1),

        # cvx.max(p1) <= cvx.Constant(1),
        # cvx.min(p1) >= cvx.Constant(0),
        # cvx.max(q1) <= cvx.Constant(1),
        # cvx.min(q1) >= cvx.Constant(0),
        # cvx.max(p2) <= cvx.Constant(1),
        # cvx.min(p2) >= cvx.Constant(0),
        # cvx.max(q2) <= cvx.Constant(1),
        # cvx.min(q2) >= cvx.Constant(0),

    ]

    eta_tau0_P0_tilde = 1 / (eta_RIS * tau_0 * PowerStation_lin)

    # Objective function Eq.(13a)
    # T_max = cvx.Variable(shape=(numIoTs,1),   pos=True, name="T_max")
    objective = cvx.Maximize(u[-1, 0])
    objective_constraints = []
    '''
    objective_constraints = [
        # cvx.norm(x: [vertical stack], p: type of norm) u[0] = t[0] * t[1]
        (cvx.norm(cvx.vstack([t[0, 0] - t[1, 0], 2 * u[0, 0]])) <= t[0, 0] + t[1, 0]),
        (cvx.norm(cvx.vstack([t[2, 0] - t[3, 0], 2 * u[1, 0]])) <= t[2, 0] + t[3, 0]),
        
        (cvx.norm(cvx.vstack([u[0, 0] - u[1, 0], 2 * u[2, 0]])) <= u[0, 0] + u[1, 0]),
    ]'''

    '''
        Attempting to parameterize the construction of the Objective Constraints for arbitrary number of IoTs
        Only works for powers of 2 [1,2,4,8,16,...]
    '''
    # Add constraints for
    # u[2*i] = t[2*i] * t[2*i+1]
    # u[0] = t[0] * t[1] # 2 IoTs
    # u[1] = t[2] * t[3] # 4 IoTs
    # u[2] = t[4] * t[5] # 6 IoTs
    # u[3] = t[6] * t[7] # 8 IoTs
    # ... and so on

    obj_constraint_list = []
    for i in range(numIoTs // 2):
        u_iot_i = (cvx.norm(cvx.vstack([t[2 * i, 0] - t[2 * i + 1, 0], 2 * u[2 * i, 0]])) <= t[2 * i, 0] + t[
            2 * i + 1, 0])
        obj_constraint_list.append(u_iot_i)

    # Add constraints for
    # u[numIoTs+i] = u[2*i] * u[2*i+1]
    # u[4] = u[0] * u[1] # 4 IoTs
    # u[5] = u[2] * u[3] # 8 IoTs
    # if only 2 IoTs, then one single u[0] = t[0] * t[1] from the above loop will suffice
    if numIoTs > 2:
        iters_needed = int(np.log2(numIoTs))
        ranges_needed = [2 ** i for i in range(iters_needed)][::-1]
        for n_range in ranges_needed:
            num_constraints = len(obj_constraint_list)
            temp_constraints_list = obj_constraint_list[-n_range:]
            for i in range(len(temp_constraints_list) // 2):
                u_i = (cvx.norm(cvx.vstack([u[2 * i, 0] - u[2 * i + 1, 0], 2 * u[2 * i + 2, 0]])) <= u[2 * i, 0] + u[
                    2 * i + 1, 0])
                obj_constraint_list.append(u_i)
    else:
        pass

    constraints += obj_constraint_list

    for i in range(0, numIoTs):
        constraints += [
            # Line (13b) ==> Line (14)
            t[i, 0] <= z_next[i, 0] ** tau_1 + tau_1 * (z_next[i, 0] ** (1 - tau_1)) * (z[i, 0] - z_next[i, 0])
        ]

        Phi_i = Phi[:, :, i] * 1e8
        Phi_i_H = Phi_i.T

        pq_1_complex = w_H @ (Phi_i_H @ v + h_d[:, [i]])
        constraints += [
            p1[i, 0] <= cvx.real(pq_1_complex),  # Line (17)
            q1[i, 0] <= cvx.imag(pq_1_complex),  # Line (18)
        ]

        # Line (19) - (expression 13c): LHS Eq(19) >= RHS Eq(15):
        constraints += [
            2 * (p1_next[i, 0] / b_next[i, 0]) * (p1[i, 0] - p1_next[i, 0]) + 2 * (q1_next[i, 0] / b_next[i, 0]) * (
                    q1[i, 0] - q1_next[i, 0]) \
            + ((p1_next[i, 0] ** 2 + q1_next[i, 0] ** 2) / b_next[i, 0]) * (1 - (b[i, 0] - b_next[i, 0]) / b_next[i, 0]) \
            >= (1 / p_next[i, 0]) * (z[i, 0] - z_next[i, 0] + (p[i, 0] * (z_next[i, 0] - 1)) / (p_next[i, 0]))
        ]

        # Eq (13d) accounting for the interference terms
        interference_terms = []
        # returns list [IOT indices >  current loop index] that have not been looped through yet

        for j in range(i + 1, numIoTs):  # for j = 1:length(not_i)
            # print('interference_j:', j)
            Phi_j = RIS_System.Phi[:, :, j] * 1e8

            interference_terms += [w_H @ (Phi_j.T @ v + h_d[:, [j]])]  # resolves to a shape(1,1) scalar value
        # print('interference terms:', len(interference_terms))
        if len(interference_terms) < 1:
            interference_terms += [0]  # avoid calling cvx.vstack([]) with an empty list [ ]
        constraints += [
            cvx.norm(cvx.vstack([cvx.vstack([cvx.sqrt(noise_power)]), cvx.vstack(interference_terms)])) <= cvx.sqrt(
                b[i, 0])
        ]

        # Eq (13e) -> Lines (20 - 22)
        psi_i = Psi[:, :, i] * 1e8
        pq_2_complex = v_r.T @ psi_i.T + g_d[[i], :]

        constraints += [
            p2[i] <= cvx.real(pq_2_complex),
            q2[i] <= cvx.imag(pq_2_complex),

            # Eq 22
            p2_next[i, 0] * (p2[i, 0] - p2_next[i, 0]) + q2_next[i, 0] * (q2[i, 0] - q2_next[i, 0]) >= p[
                i, 0] * eta_tau0_P0_tilde * 1e-6  # 1/(eta_tau0_P0)
        ]
    problem = cvx.Problem(objective, constraints)
    return problem


def Optimize_Solve_Sub_Problem_1(RIS_System, input_values, n_iterations=10, solver='SCS', verbose=False, debug=False):
    # Initial values
    numIoTs = RIS_System.numIoTs
    numAntennasAP = RIS_System.numAntennasAP

    w = input_values['omega']

    if np.array(w).any() == False:
        # w = array(rand(numAntennasAP, 1) + 1j * rand(numAntennasAP, 1), dtype=complex)
        w_theta = np.random.uniform(0, 2 * np.pi, (numAntennasAP, 1))
        w = np.exp(1j * w_theta)
    else:
        w = w

    next_dict = {
        'p1_next': 0.009 * ones((numIoTs, 1)),  # denoted with p1_i^(n-1)
        'q1_next': 0.009 * ones((numIoTs, 1)),  # denoted with q1_i^(n-1)
        'p2_next': 0.009 * ones((numIoTs, 1)),  # denoted with p2_i^(n-1)
        'q2_next': 0.009 * ones((numIoTs, 1)),  # denoted with q2_i^(n-1)
        'b_next': 0.009 * ones((numIoTs, 1)),  # denoted with B_i^(n-1)
        'z_next': 0.009 * ones((numIoTs, 1)),  # denoted with z_i^(n-1)
        'p_next': 0.009 * ones((numIoTs, 1)),  # denoted with p_i^(n-1)
        'omega': w
    }

    solve_dict = dict(status=[], value=[], v=[], v_r=[], omega=w)

    for i in range(n_iterations):
        base_string = f'\tSub_1: {i},'

        sub_problem_1 = Sub_Problem_1(RIS_System,
                                      next_values=next_dict)

        # Try solving the iteration and saving the variable values
        try:
            sub_problem_1.solve(solver=solver, verbose=verbose)
            return_string = base_string + f'\tStatus: {sub_problem_1.status}, \t\tValue: {sub_problem_1.value}'
            if debug:
                print(return_string)

            if sub_problem_1.status in ['infeasible', 'unbounded', 'unbounded_inaccurate']:
                if debug:
                    print(f'\tproblem is {sub_problem_1.status}. Returning previous values.')
                return solve_dict

            solve_dict['status'].append(sub_problem_1.status)
            solve_dict['value'].append(sub_problem_1.value)

            next_dict['p1_next'] = sub_problem_1.var_dict['p1'].value
            next_dict['q1_next'] = sub_problem_1.var_dict['q1'].value
            next_dict['p2_next'] = sub_problem_1.var_dict['p2'].value
            next_dict['q2_next'] = sub_problem_1.var_dict['q2'].value
            next_dict['b_next'] = sub_problem_1.var_dict['b'].value
            next_dict['z_next'] = sub_problem_1.var_dict['z'].value
            next_dict['p_next'] = sub_problem_1.var_dict['p'].value
            # print(' [p, z]:\n', np.block([next_dict['p_next'], next_dict['z_next']]))

            solve_dict['v'] = sub_problem_1.var_dict['v'].value
            solve_dict['v_r'] = sub_problem_1.var_dict['v_r'].value
            solve_dict['t'] = sub_problem_1.var_dict['t'].value

            #print(' [v, v_r, t]:\n', np.block([solve_dict['v'], solve_dict['v_r'], solve_dict['t']]))




        except Exception as inst:
            if debug:
                print(f'{base_string}\tArbitrary error, the code will return values from the previous iterations solved states.\n')
            '''
            print(type(inst))
            print(inst.args)
            print(inst)
            if "rescode.err_huge_aij(1380)" in str(inst):
                # Get constraint name
                inst_str = str(inst)
                a = "constraint '' ("
                b = ")"
                a_pos = inst_str.find(a) + len(a)
                b_pos = inst_str[a_pos:].find(b) + a_pos
                constraint_num = int(inst_str[a_pos:b_pos])
                print(sub_problem_1.constraints[constraint_num])
            '''
            # return the previous iterations solved state
            return solve_dict
        else:
            # This runs when there is no error
            pass
        finally:
            # This runs no matter what
            pass
    return solve_dict


if __name__ == '__main__':
    RIS_System = RISNomaSystem(numRISelements=10,
                               numIoTs=32,
                               PowerStation_dBm=30,
                               verbose=True,
                               plot=False)

    E_RIS_TH = RIS_System.E_RIS_TH
    eta_RIS = RIS_System.eta_RIS
    PowerStation_lin = RIS_System.PowerStation_lin
    G = RIS_System.G

    '''
        This part is lines 3) to 4f). Optimizing Harvesting and Reflecting RIS Elements.

        To represent the energy being harvested during tau_0 time period, I think we should set the rows associated with
        the used harvesting elements to (1-tau_0) x original_value, or just tau_1 x original_value
    '''
    tau_0, tau_1, A_h, A_r, A_h_idx, A_r_idx = HarvestingElementOptimizer(RIS_System,
                                                                          verbose=True,
                                                                          plot=False)

    RIS_System.tau_0 = tau_0
    RIS_System.tau_1 = tau_1

    # Update RIS_System Channel Matrices to reflect the devices consuming power during charging
    # RIS_System.H[A_h_idx, :] = RIS_System.H[A_h_idx, :] - RIS_System.E_RIS_TH / RIS_System.numAntennasAP
    # RIS_System.G[:, A_h_idx] = RIS_System.G[:, A_h_idx] - RIS_System.E_RIS_TH / RIS_System.numAntennasPS
    # RIS_System.g_r[:, A_h_idx] = RIS_System.g_r[:, A_h_idx] - RIS_System.E_RIS_TH / RIS_System.numIoTs
    # RIS_System.h_r[A_h_idx, :] = RIS_System.h_r[A_h_idx, :] - RIS_System.E_RIS_TH / RIS_System.numIoTs

    # Recalculate Phi and Psi Matrices (they're deterministic upon Channel Matrices)
    # RIS_System.reflection_Phi()
    # RIS_System.reflection_Psi()

    w_theta = np.random.uniform(0, 2 * np.pi, (RIS_System.numAntennasAP, 1))
    w_initial = np.exp(1j * w_theta)

    system_values = dict()
    system_values['omega'] = w_initial

    print('\tSub Problem 1:\nGiven w, solve for v and v_r')

    solve_dict_p1 = Optimize_Solve_Sub_Problem_1(RIS_System,
                                                 input_values=system_values,
                                                 n_iterations=5, solver='MOSEK', verbose=True)

    print('final value:', solve_dict_p1['value'], '\noptimized v:', solve_dict_p1['v'], '\noptimized v_r:',
          solve_dict_p1['v_r'], '\noptimized t:', solve_dict_p1['t'], '\nw:', solve_dict_p1['omega'])
