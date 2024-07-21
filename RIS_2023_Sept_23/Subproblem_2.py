from typing import Dict, Union, Any

from RIS_Channel_Generation import RISNomaSystem
from Optimize_RIS_Elements import HarvestingElementOptimizer

from Subproblem_1 import Optimize_Solve_Sub_Problem_1
import numpy as np
from numpy import sqrt, pi, sin, cos, multiply, linspace, ones, zeros, \
    log10, transpose, exp, arange, concatenate, abs, sort, \
    sum, array, argsort, diag, matrix, argwhere, ndarray

from numpy.random import rand
import cvxpy as cvx


def Sub_problem_2(RIS_System, input_values):
    """

    :param RIS_System:
    :param input_values:
    :return:
    """

    numAntennasAP = RIS_System.numAntennasAP
    numIoTs = RIS_System.numIoTs
    noise_power = RIS_System.noise_power
    Phi = RIS_System.Phi
    Psi = RIS_System.Psi
    g_d = RIS_System.g_d

    h_d = RIS_System.h_d
    tau_0 = RIS_System.tau_0
    tau_1 = RIS_System.tau_1

    eta_tau0_P0_tilde = 1 / (RIS_System.eta_RIS * tau_0 * RIS_System.PowerStation_lin)

    constraints = []

    p1_next = input_values['p1_next']
    q1_next = input_values['q1_next']
    p2_next = input_values['p2_next']
    q2_next = input_values['q2_next']
    b_next = input_values['b_next']
    p_next = input_values['p_next']
    z_next = input_values['z_next']
    omega_next = input_values['omega_next']
    v = np.array(input_values['v'])
    v_r = input_values['v_r']

    u = cvx.Variable(shape=(numIoTs - 1, 1), pos=True, name="u")
    t = cvx.Variable(shape=(numIoTs, 1), pos=True, name="t")  # lower bound input from subproblem 1
    z = cvx.Variable(shape=(numIoTs, 1), pos=True, name="z")
    b = cvx.Variable(shape=(numIoTs, 1), pos=True, name="b")
    p = cvx.Variable(shape=(numIoTs, 1), pos=True, name="p")
    p1 = cvx.Variable((numIoTs, 1), pos=True, name="p1")
    q1 = cvx.Variable((numIoTs, 1), pos=True, name="q1")
    p2 = cvx.Variable((numIoTs, 1), pos=True, name="p2")
    q2 = cvx.Variable((numIoTs, 1), pos=True, name="q2")

    # These are used during Sub Problem 2
    # RIS reflection coefficients during IoT harvesting
    # w_re = cvx.Variable((numAntennasAP, 1), pos=True, name="w_re")  # REAL
    # w_im = cvx.Variable((numAntennasAP, 1), pos=True, name="w_im")  # IMAGINARY
    omega = cvx.Variable((numAntennasAP, 1), complex=True, name="omega")

    # Objective function Eq.(13a)
    objective = cvx.Maximize(u[-1, 0])

    # subproblem 1: maximize(t_1 * t_2 * ... t_n)
    # subporblem 2: maximize(z_1^tau_1 * z_2^tau_1 * ... )

    '''
        Constraints to maximize individual IoT values
        # 1) Add constraints for
        # u[2*i] = t[2*i] * t[2*i+1]
        # u[0] = t[0] * t[1] # 2 IoTs
        # u[1] = t[2] * t[3] # 4 IoTs
        # u[2] = t[4] * t[5] # 6 IoTs
        # u[3] = t[6] * t[7] # 8 IoTs
        # ... and so on
        
        # 2) Add constraints for
        # u[numIoTs+i] = u[2*i] * u[2*i+1]
        # u[4] = u[0] * u[1] # 4 IoTs
        # u[5] = u[2] * u[3] # 8 IoTs
        # if only 2 IoTs, then one single u[0] = t[0] * t[1] from the above loop will suffice
        
    '''
    obj_constraint_list = []
    for i in range(numIoTs // 2):
        u_iot_i = (cvx.norm(cvx.vstack([t[2 * i, 0] - t[2 * i + 1, 0], 2 * u[2 * i, 0]])) <= t[2 * i, 0] + t[
            2 * i + 1, 0])
        obj_constraint_list.append(u_iot_i)

    if numIoTs > 2:
        iters_needed = int(np.log2(numIoTs))
        ranges_needed = [2 ** i for i in range(iters_needed)][::-1]
        for n_range in ranges_needed:
            #num_constraints = len(obj_constraint_list)
            temp_constraints_list = obj_constraint_list[-n_range:]
            for i in range(len(temp_constraints_list) // 2):
                u_i = (cvx.norm(cvx.vstack([u[2 * i, 0] - u[2 * i + 1, 0], 2 * u[2 * i + 2, 0]])) <= u[2 * i, 0] + u[
                    2 * i + 1, 0])
                obj_constraint_list.append(u_i)
    else:
        pass
    constraints += obj_constraint_list

    for i in range(numIoTs):

        # Line (23b)
        # constraints += [t[i, 0] <= z[i, 0] ** tau_1]
        # Line (23c)

        # replace above line with similar
        # Line (13b) ==> Line (14)
        constraints += [
            t[i, 0] <= z_next[i, 0] ** tau_1 + tau_1 * (z_next[i, 0] ** (1 - tau_1)) * (z[i, 0] - z_next[i, 0])]

        '''
        constraints += [
            (2 * p_next[i, 0] * cvx.real(omega_next) * (p[i, 0] - p_next[i, 0]) + p_next[i, 0] ** 2 * w_re[:, i] >= cvx.real(omega)),
            (2 * p_next[i, 0] * cvx.imag(omega_next) * (p[i, 0] - p_next[i, 0]) + p_next[i, 0] ** 2 * w_im[:, i] >= cvx.imag(omega)),
        ]'''
        constraints += [
            (2 * p_next[i, 0] * cvx.real(omega_next) * (p[i, 0] - p_next[i, 0]) + p_next[i, 0] ** 2 * cvx.real(
                omega_next) >= cvx.real(omega)),
            (2 * p_next[i, 0] * cvx.imag(omega_next) * (p[i, 0] - p_next[i, 0]) + p_next[i, 0] ** 2 * cvx.imag(
                omega_next) >= cvx.imag(omega)),
        ]

        # Line (23d) is very similar to (17 - 19)
        '''
        norm(omega^H @ (phi_i^H @ v + h_d_i)) / B_i >= (z_i - 1)
        '''
        Phi_i_H = (1e8 * Phi[:, :, i]).T
        #print('shapes: ',omega.T.shape, Phi_i_H.shape, v.shape)#, h_d[:, [i]].shape)
        pq_1_complex = omega.T @ (Phi_i_H @ v + h_d[:, [i]])

        constraints += [
            p1[i, 0] == cvx.real(pq_1_complex),  # Line (17)
            q1[i, 0] == cvx.imag(pq_1_complex),  # Line (18)
        ]
        # Line (19) - (expression 13c): LHS Eq(19) >= RHS Eq(15):
        constraints += [
            2 * (p1_next[i, 0] / b_next[i, 0]) * (p1[i, 0] - p1_next[i, 0]) + 2 * (q1_next[i, 0] / b_next[i, 0]) * (
                    q1[i, 0] - q1_next[i, 0]) \
            + ((p1_next[i, 0] ** 2 + q1_next[i, 0] ** 2) / b_next[i, 0]) * (1 - (b[i, 0] - b_next[i, 0]) / b_next[i, 0]) \
            >= z[i, 0] - 1
        ]

        # constraint_23d = cvx.norm(pq_1_complex)/b[i, 0] >= (z[i, 0]-1)
        # constraints += [constraint_23d]

        # Line (23e) is similar to Line (13d)
        interference_terms = []
        for j in range(i + 1, numIoTs):  # for j = 1:length(not_i)
            # print('interference_j:', j)
            Phi_j_H = (1e8 * RIS_System.Phi[:, :, j]).T

            interference_terms += [omega.T @ (Phi_j_H @ v + h_d[:, [j]])]  # resolves to a shape(1,1) scalar value
        # print('interference terms:', len(interference_terms))
        if len(interference_terms) < 1:
            interference_terms += [0]  # avoid calling cvx.vstack([]) with an empty list [ ]
        constraints += [
            cvx.norm(cvx.vstack([cvx.vstack([noise_power]), cvx.vstack(interference_terms)])) <= b[i, 0]
        ]

        '''Eq (13e) -> Lines (20 - 22)
            I want to add this constraint because a constraint including the optimized v_r is not included in 
            Subproblem 2.'''

        '''

        psi_i = 1e8 * Psi[:, :, i]
        pq_2_complex = v_r.T @ psi_i.T + g_d[[i], :]

        constraints += [
            p2[i] <= cvx.real(pq_2_complex),
            q2[i] <= cvx.imag(pq_2_complex),

            # Eq 22
            p2_next[i, 0] * (p2[i, 0] - p2_next[i, 0]) + q2_next[i, 0] * (q2[i, 0] - q2_next[i, 0])
            >= p[i, 0] * eta_tau0_P0_tilde * 1e-6  # 1/(eta_tau0_P0)
        ]
        '''

    sub_problem_2 = cvx.Problem(objective, constraints)

    return sub_problem_2


def Optimize_Solve_Sub_Problem_2(RIS_System, input_values, n_iterations=10, solver='SCS',
                                 verbose=False, debug=False):
    # Initial solve
    numIoTs = RIS_System.numIoTs
    numAntennasAP = RIS_System.numAntennasAP

    next_dict = dict()
    next_dict['p1_next'] = 0.009 * ones((numIoTs, 1))  # denoted with p1_i^(n-1)
    next_dict['q1_next'] = 0.009 * ones((numIoTs, 1))  # denoted with q1_i^(n-1)
    next_dict['p2_next'] = 0.009 * ones((numIoTs, 1))  # denoted with p2_i^(n-1)
    next_dict['q2_next'] = 0.009 * ones((numIoTs, 1))  # denoted with q2_i^(n-1)
    next_dict['b_next'] = 0.009 * ones((numIoTs, 1))  # denoted with B_i^(n-1)
    next_dict['p_next'] = 0.009 * ones((numIoTs, 1))  # denoted with p_i^(n-1)
    next_dict['z_next'] = 0.009 * ones((numIoTs, 1))  # denoted with z_i^(n-1)
    next_dict['omega_next'] = input_values['omega']
    next_dict['v'] = input_values['v_r']
    next_dict['v_r'] = input_values['v_r']

    solve_dict = dict(status=[], value=[], omega=[])

    for i in range(n_iterations):
        base_string = f'\tSub_2: {i}, '
        sub_problem_2 = Sub_problem_2(RIS_System, input_values=next_dict)

        # Try solving the iteration and saving the variable values
        try:
            sub_problem_2.solve(solver=solver, verbose=verbose)
            return_string = base_string + f'Status: {sub_problem_2.status},\t\tValue: {sub_problem_2.value}'
            # print('\tStatus:', sub_problem_2.status)
            # print('\tValue:', sub_problem_2.value)
            if debug:
                print(return_string)
            if sub_problem_2.status in ['infeasible', 'unbounded', 'unbounded_inaccurate']:
                if debug:
                    print(f'\tproblem is {sub_problem_2.status}. Returning previous values.')
                return solve_dict

            solve_dict['status'].append(sub_problem_2.status)
            solve_dict['value'].append(sub_problem_2.value)
            solve_dict['omega'] = sub_problem_2.var_dict['omega'].value

            next_dict['p1_next'] = sub_problem_2.var_dict['p1'].value
            next_dict['q1_next'] = sub_problem_2.var_dict['q1'].value
            # next_dict['p2_next'] = sub_problem_2.var_dict['p2'].value
            # next_dict['q2_next'] = sub_problem_2.var_dict['q2'].value
            next_dict['b_next'] = sub_problem_2.var_dict['b'].value
            next_dict['z_next'] = sub_problem_2.var_dict['z'].value
            next_dict['p_next'] = sub_problem_2.var_dict['p'].value
            next_dict['z_next'] = sub_problem_2.var_dict['z'].value
            next_dict['omega_next'] = sub_problem_2.var_dict['omega'].value

            # print(' [p, z]:\n', np.block([next_dict['p_next'], next_dict['z_next']]))
            # print(' [omega]:\n', np.block([solve_dict['omega_next']]))


        except Exception as inst:
            if debug:
                print(f'{base_string}\tArbitrary error, the code will return values from the previous iterations solved states.')

            #print(type(inst))
            #print(inst.args)
            #print(inst)
            '''
            if "rescode.err_huge_aij(1380)" in str(inst):
                # Get constraint name
                inst_str = str(inst)
                a = "constraint '' ("
                b = ")"
                a_pos = inst_str.find(a) + len(a)
                b_pos = inst_str[a_pos:].find(b) + a_pos
                constraint_num = int(inst_str[a_pos:b_pos])
                print(sub_problem_2.constraints[constraint_num])
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


if __name__ == "__main__":

    RIS_System = RISNomaSystem(numRISelements=144,
                               numIoTs=16,
                               PowerStation_dBm=30,
                               noisePower_dB=-100,
                               verbose=True,
                               plot=False)

    '''
        This part is lines 3) to 4f). Optimizing Harvesting and Reflecting RIS Elements.

        To represent the energy being harvested during tau_0 time period, I think we should set the rows associated with
        the used harvesting elements to (1-tau_0) x original_value, or just tau_1 x original_value
    '''

    RIS_System.tau_0, RIS_System.tau_1, A_h, A_r, A_h_idx, A_r_idx = HarvestingElementOptimizer(RIS_System,
                                                                                                verbose=True,
                                                                                                plot=False, )

    # Update RIS_System Channel Matrices to reflect the devices charging
    # RIS_System.H[A_h_idx, :] = RIS_System.H[A_h_idx, :] - RIS_System.E_RIS_TH / RIS_System.numAntennasAP
    # RIS_System.G[:, A_h_idx] = RIS_System.G[:, A_h_idx] - RIS_System.E_RIS_TH / RIS_System.numAntennasPS
    # RIS_System.g_r[:, A_h_idx] = RIS_System.g_r[:, A_h_idx] - RIS_System.E_RIS_TH / RIS_System.numIoTs
    # RIS_System.h_r[A_h_idx, :] = RIS_System.h_r[A_h_idx, :] - RIS_System.E_RIS_TH / RIS_System.numIoTs

    # Recalculate Phi and Psi Matrices (they're deterministic upon Channel Matrices)
    RIS_System.reflection_Phi()
    RIS_System.reflection_Psi()

    # Sub Problem 2
    # initial v_r and v vector values of uniform distribution over [0, 1)

    w_theta = np.random.uniform(0, 2 * np.pi, (RIS_System.numAntennasAP, 1))
    w_initial = np.exp(1j * w_theta)

    system_values = dict()
    system_values['omega'] = w_initial

    solve_dict_p1 = Optimize_Solve_Sub_Problem_1(RIS_System=RIS_System,
                                                 input_values=system_values,
                                                 n_iterations=5, solver='MOSEK', verbose=True)
    # print(f't: {solve_dict_p1["t"]}')
    print(f'omega: {solve_dict_p1["omega"]}')
    print(f'v: {solve_dict_p1["v"]}')

    print('\tSub Problem 2:\nGiven v and v_r, solve for omega')
    solve_dict_p2 = Optimize_Solve_Sub_Problem_2(RIS_System=RIS_System,
                                                 input_values=solve_dict_p1,
                                                 n_iterations=5, solver='MOSEK', verbose=True)

    print('final value:', solve_dict_p2['value'], '\noptimized omega:', solve_dict_p2['omega'])
