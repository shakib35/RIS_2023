from RIS_Channel_Generation import RISNomaSystem
from Optimize_RIS_Elements import HarvestingElementOptimizer
from Subproblem_1 import Optimize_Solve_Sub_Problem_1
from Subproblem_2 import Optimize_Solve_Sub_Problem_2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
jdsvn

def Equation_Three(system_parameters, p1_values, p2_values):
    n_iots = system_parameters.numIoTs
    tau_0 = system_parameters.tau_0
    tau_1 = system_parameters.tau_1
    eta = system_parameters.eta_RIS
    P_0 = system_parameters.PowerStation_lin
    sigma = system_parameters.noise_power
    g_d = system_parameters.g_d
    h_d = system_parameters.h_d
    Phi = system_parameters.Phi
    Psi = system_parameters.Psi

    eta_tau0_P0 = eta * tau_0 * P_0

    v_r = p1_values['v_r']
    v = p1_values['v']

    omega = p2_values['omega']

    # Calculate signal terms
    signal_terms = []
    for i in range(n_iots):
        psi_i = Psi[:, :, i]
        phi_i = Phi[:, :, i]
        power_i = eta_tau0_P0 * np.linalg.norm(v_r.T @ psi_i.T + g_d[i, :])
        signal_i = np.linalg.norm(omega.T @ (phi_i.T @ v + h_d[:, i]))
        signal_terms.append(power_i * signal_i)
    signal_terms = np.array(signal_terms)
    # print('Signal terms:\n', signal_terms)

    # Calculate noise and interference terms
    noise_terms = []
    system_noise = np.linalg.norm(omega.T) * sigma ** 2
    for j in range(1, n_iots):
        interference_terms = signal_terms[j:]
        noise_term = np.sum(interference_terms) + system_noise
        # print(f'noise term {j}: \ninterference terms: {interference_terms} + noise: {system_noise}\nnoise term {j}: {noise_term}')
        noise_terms.append(noise_term)
    noise_terms.append(np.linalg.norm(omega.T) * sigma ** 2)
    # print('Noise terms:\n', noise_terms)

    rate_terms = tau_1 * np.log2(1 + signal_terms / noise_terms)
    # print('Rate terms:\n', rate_terms)

    rate_sum = np.sum(rate_terms)
    # print('Rate sum:\n', rate_sum)
    return rate_sum



def optimizeSystem(RIS_System, opt_iterations=10, verbose=False, debug=False):
    """
    Input an RIS channel to optimize. Return the optimized channel capacity
    .
    :param RIS_System:
    :return:
    """

    '''
        This part is lines 3) to 4f). Optimizing Harvesting and Reflecting RIS Elements.

        To represent the energy being harvested during tau_0 time period, I think we should set the rows associated with
        the used harvesting elements to (1-tau_0) x original_value, or just tau_1 x original_value
    '''
    RIS_System.tau_0, RIS_System.tau_1, A_h, A_r, A_h_idx, A_r_idx = HarvestingElementOptimizer(RIS_System,
                                                                                                verbose=False,
                                                                                                plot=False, )

    # Update RIS_System Channel Matrices to reflect the devices consuming power during charging
    # RIS_System.H[A_h_idx, :] = RIS_System.H[A_h_idx, :] - RIS_System.E_RIS_TH / RIS_System.numAntennasAP
    # RIS_System.G[:, A_h_idx] = RIS_System.G[:, A_h_idx] - RIS_System.E_RIS_TH / RIS_System.numAntennasPS
    # RIS_System.g_r[:, A_h_idx] = RIS_System.g_r[:, A_h_idx] - RIS_System.E_RIS_TH / RIS_System.numIoTs
    # RIS_System.h_r[A_h_idx, :] = RIS_System.h_r[A_h_idx, :] - RIS_System.E_RIS_TH / RIS_System.numIoTs

    # Recalculate Phi and Psi Matrices (they're deterministic upon Channel Matrices)
    RIS_System.reflection_Phi()
    RIS_System.reflection_Psi()

    w_theta = np.random.uniform(0, 2 * np.pi, (RIS_System.numAntennasAP, 1))
    w_initial = np.exp(1j * w_theta)

    system_values = dict()
    system_values['omega'] = w_initial

    system_values = {
        'omega': w_initial,
        'values': []
    }

    optimization_iterations = opt_iterations
    for i in range(optimization_iterations):
        if debug:
            output_string = f"Opt Loop: {i},"
            print(output_string)



        """
            Sub-Problem 1:
            Given a fixed omega, maximize objective function.
            Return v and v_r.
        """

        solve_dict_p1 = Optimize_Solve_Sub_Problem_1(RIS_System,
                                                     input_values=system_values,
                                                     n_iterations=5, solver='MOSEK', debug=debug)

        """
        Sub-Problem 2:
        Given optimized v from previous objective, optimize objective.
        Return omega.
        """

        solve_dict_p2 = Optimize_Solve_Sub_Problem_2(RIS_System=RIS_System,
                                                     input_values=solve_dict_p1,
                                                     n_iterations=5, solver='MOSEK', debug=debug)

        # print('\tSubproblem 2 values:', solve_dict_p2['value'])  # , '\noptimized omega:', solve_dict_p_2['omega'])

        system_values['omega'] = solve_dict_p2['omega']
        # print('omega: ', system_values['omega'])

        try:
            output = Equation_Three(system_parameters=RIS_System,
                                    p1_values=solve_dict_p1,
                                    p2_values=solve_dict_p2)
            #if verbose: print(output_string, '\tvalue: ', output)
            system_values['values'].append(output)

        except Exception:
            if debug: print(output_string, '\tArbitrary Error.')
            break

        if len(system_values['values']) > 4 and np.all(np.diff(system_values['values'][-3:]) <= 0):
            if debug: print(output_string, '\tmonotonically decreasing 3 times.')
            break

    if debug: print(output_string, '\tFinal Value:', system_values['values'][-1])

    return system_values['values'][-1]



if __name__ == '__main__':

    avg_runtime = 0
    channel_iterations = 10
    optimization_loop_iterations = 3

    linear_power_list = [0.0001, 5, 10, 15, 17, 20, 25, 30, 40, 50, 55]
    noise_power = -100
    iot_devices = 64
    ris_elements = [4, 25, 64, 144, 256]
    folder_path = os.path.dirname(os.path.abspath(__file__))



    for num_ris in ris_elements:

        subfolder = f"ris_elements\\{iot_devices}_iot\\ris_{num_ris}\\"
        sub_path = os.path.join(folder_path, subfolder)
        if not os.path.isdir(sub_path):
            os.mkdir(sub_path)



        for linear_power in linear_power_list:

            '''Checking if the file exists'''

            name = f'{linear_power}dBm.csv'
            #print(f'\n{name}')

            save_path = os.path.join(folder_path, subfolder, name)
            print('output file name: ', save_path)
            # jjz = np.loadtxt(save_path, delimiter=',')
            # np.savetxt(save_path, jz, delimiter=",")

            if os.path.isfile(save_path):
                '''load the file'''
                print('file exists')
                optimized_channel_values = np.loadtxt(save_path, delimiter=',')
                rate_sum_array = optimized_channel_values.flatten()
                rate_sum_array = rate_sum_array[~np.isnan(rate_sum_array)]
                if len(rate_sum_array)>0:
                    print(f'Channel Capacity Stats: '
                          f'\nData points: {len(rate_sum_array)} '
                          f'\nMean: {rate_sum_array.mean()}'
                          f'\nMedian: {np.median(rate_sum_array)}')

            else:
                '''create the file'''
                print(f'file doesnt exist'
                      f'\ncreating file at: {save_path}')


                empty_csv = pd.DataFrame(list())
                empty_csv.to_csv(save_path)
                optimized_channel_values = []
                rate_sum_array = np.array(optimized_channel_values)

            counter = 0
            try:

                while len(rate_sum_array) < channel_iterations:
                    try:

                        counter += 1

                        start_time = time.perf_counter()

                        output_string = f'\niot_{iot_devices}\\ris_{num_ris}\\{name} Channel: {counter}'
                        print(output_string)

                        '''Generate random channel values and optimize capacity.'''
                        RIS_System = RISNomaSystem(numRISelements=num_ris,
                                                   numIoTs=iot_devices,
                                                   PowerStation_dBm=linear_power,
                                                   noisePower_dB=noise_power,
                                                   verbose=False,
                                                   plot=False)

                        '''optimize new ris system channel and store final value'''
                        optimized_value = optimizeSystem(RIS_System,
                                                         opt_iterations=optimization_loop_iterations,
                                                         debug=True, verbose=False)
                        rate_sum_array = np.append(rate_sum_array, optimized_value)
                        rate_sum_array = rate_sum_array[~np.isnan(rate_sum_array)]


                        end_time = time.perf_counter()
                        runtime = end_time - start_time

                        '''
                            running average:
                            u_n = ((n-1) * u_n_1 + x_n) / n
                        '''
                        samples_remaining = channel_iterations - len(rate_sum_array)
                        avg_runtime = (counter * avg_runtime + runtime) / (counter + 1)
                        time_to_end = avg_runtime * samples_remaining / 60 / 60
                        print(f'\toptimized value: {optimized_value}'
                              f'\toptimization time: {runtime} seconds'
                              f'\n\taverage time: {avg_runtime} seconds'
                              f'\n\tsamples left: {samples_remaining}'
                              f'\n\ttime left to {channel_iterations} samples: {time_to_end} hours')

                    except (ValueError, IndexError) as inst:
                        print(f'\n{inst} error encountered. skip to next channel configuration'
                              # f'\n{optimized_channel_values}'
                              )
                        pass

                    np.savetxt(fname=save_path,
                               X=rate_sum_array,
                               fmt='%f',
                               delimiter=",")

            except KeyboardInterrupt:
                print(f'\nExiting Early. saving values to {save_path}')
                np.savetxt(fname=save_path,
                           X=rate_sum_array,
                           fmt='%f',
                           delimiter=",")
                break

            print(f'saving values to {save_path}'
                  # f'\n{optimized_channel_values}'
                  )
            np.savetxt(fname=save_path,
                       X=rate_sum_array,
                       fmt='%f',
                       delimiter=",")

            rate_sum_array = np.array(optimized_channel_values).flatten()
            print(f'Channel Capacity Stats: '
                  f'\nData points: {len(rate_sum_array)} '
                  f'\nMean: {rate_sum_array.mean()}'
                  f'\nMedian: {np.median(rate_sum_array)}')

            '''except KeyboardInterrupt:
                print(f'\nyou hit the emergency brakes'
                      f'\nsaving to file: {save_path}')
                np.savetxt(fname=save_path,
                           X=optimized_channel_values,
                           fmt='%f',
                           delimiter=",")'''




        """
        '''Generate random channel values and optimize capacity.'''
        RIS_System = RISNomaSystem(numRISelements=144,
                                   numIoTs=8,
                                   PowerStation_dBm=100,
                                   noisePower_dB=-30,
                                   verbose=False,
                                   plot=False)
    
        '''optimize new ris system channel and store final value'''
        optimized_value = optimizeSystem(RIS_System, opt_iterations=10, verbose=True)
        print(optimized_value)
        """