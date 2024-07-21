
import cvxpy as cvx
import numpy as np
from numpy.random import rand

from RIS_Channel_Generation import RISNomaSystem

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D


def discreteColorMap(ris_element_square):
    '''
    This function creates a discrete color map of the Harvesting and Reflecting RIS elements
    :param ris_element_square: square binary matrix representing RIS elements
    :return: discrete color map of RIS surface
    '''
    n = ris_element_square.shape[0]
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    c = ax.pcolor(ris_element_square, edgecolors='k',cmap=colors.ListedColormap(['red', 'green']),linewidths=2)
    ax.set_title('RIS Elements')
    dummy_elements = [Line2D([0], [0], color='green', lw=4),Line2D([0], [0], color='red', lw=4)]
    ax.legend(dummy_elements, ['Harvesting', 'Reflecting'], loc='lower right')
    ax.set_ylim(n, 0)
    ax.set_xlim(0, n)
    fig.tight_layout()
    plt.show()

def HarvestingElementOptimizer(RIS_System, verbose=False, plot=True):
    """
    This is a modified knapsack problem.

    Knapsack Maximizes the utility of set of items, given their weight:

    item_utility, item_weight = [u_1,u_2,...,u_n], [w_1,w_2,...,w_n]

    item_selection = [s_1,s_2,...,s_n] boolean (1 or 0)

    weight_max = M

    weight_constraint = item_weight @ item_selection <= weight_max

    Maximize (item_utility @ item_selection), given [weight_constraint]


    This is instead minimizing the E_Ah (power consumed by RIS and IoTs) by selecting channels whose power values
    sum as close to E_RIS_TH (power requirement for system) as possible.

    E_Ah is further minimized by reducing tau_0 (power harvesting time constant).

    tau_0 * E_Ah_star >= E_RIS_TH, tau_0 = 1

    tau_0_star >= E_RIS_TH / E_Ah_star

    :param G: channel between PS and RIS. Must be same length as number of RIS elements.
    :param E_RIS_TH: threshold energy to power the RIS.
    :param eta_RIS: power efficiency of the RIS.
    :param PowerStation_lin: the supplied power from the Power Station.
    :return:
    """
    G = RIS_System.G
    eta_RIS = RIS_System.eta_RIS
    PowerStation_lin = RIS_System.PowerStation_lin
    E_RIS_TH = RIS_System.E_RIS_TH

    # Part I. Minimize(E_Ah)
    channel_power = np.sum(np.abs(G), axis=0) ** 2
    total_ris_elements = len(channel_power)
    ah_constraints = []

    # The boolean variable we are solving for
    ah_selection = cvx.Variable(shape=(channel_power.shape[0], 1), boolean=True, name='Ah_selection')

    # tau_0 can be very small, we'll start with 1
    # tau_0 = 1
    # E_Ah = tau_0 * eta_ris * Po * G_column_sum @ ah_selection
    E_Ah_sum = 1 * eta_RIS * PowerStation_lin * cvx.scalar_product(channel_power, ah_selection)
    ah_constraints += [E_Ah_sum >= E_RIS_TH]
    Ah_knapsack = cvx.Problem(cvx.Minimize(E_Ah_sum), ah_constraints)

    # Solving for E_Ah using a Mixed Integer Solver
    Ah_knapsack.solve(solver=cvx.GLPK_MI, verbose=verbose)

    # Part II. Minimize(tau_0)
    tau_0 = cvx.Variable(name="tau_0")
    E_Ah = Ah_knapsack.value
    tau_objective = cvx.Minimize(tau_0)
    tau_constraints = [tau_0 * E_Ah >= E_RIS_TH]
    tau_problem = cvx.Problem(tau_objective, tau_constraints)
    tau_problem.solve(verbose=verbose)
    tau_0_star = tau_problem.value
    tau_1 = 1 - tau_0_star



    A_h = np.sum(ah_selection.value).astype(int)
    A_r = total_ris_elements - A_h
    A_h_elements = np.array(ah_selection.value).flatten()
    n = A_h_elements.shape[0]

    # Prints out the RIS elements in a square shape.
    # If RIS elements cannot fit in n x n square space, pad with NaN values. Display ONLY.
    nn_ceil = np.ceil(np.sqrt(n)).astype(int)
    ris_element_square = np.pad(A_h_elements, pad_width=(0, nn_ceil ** 2 - n),mode='constant',constant_values=(np.nan,)).reshape(nn_ceil, nn_ceil)

    # Indices of Harvesting and Reflecting Elements
    A_h_idx = np.argwhere(A_h_elements == 1.).flatten()
    A_r_idx = np.argwhere(A_h_elements == 0.).flatten()

    if verbose:
        print('\nindividual channels to select from:', len(channel_power))
        print('total available power:', np.sum(channel_power))
        print('eta_RIS:', eta_RIS)
        print('E_RIS_TH:', E_RIS_TH)
        print('PowerStation_lin:', PowerStation_lin)

        print(f'\nMinimize(E_Ah)\ns.t.: tau_0 * E_Ah >= E_RIS_TH,\n\tE_Ah = RIS_elements @ Selection_boolean, \n\ttau_0 = 1, E_RIS_TH = {E_RIS_TH}')
        print(f'Optimal E_Ah: {Ah_knapsack.value}')

        print(f'\nMinimize(tau_0) \ns.t.: tau_0 * E_Ah >= E_RIS_TH, \n\tE_RIS_TH={E_RIS_TH}, E_Ah = {Ah_knapsack.value}')
        print(f'Optimal tau_0: {tau_0_star:.6f} tau_1 = {tau_1:.6f}')

        print(f'\nOptimal E_Ah >= E_RIS_TH: \n\t{tau_0_star * Ah_knapsack.value} >= {E_RIS_TH}')

        print(f'\nRIS Elements: (N = {total_ris_elements}), (Ah = {A_h}), (Ar = {A_r}).\n1: Harvesting {A_h_idx},\n0: Reflecting {A_r_idx}, \nnan: empty\nRIS Elements:\n{ris_element_square}' )

    if plot:
        discreteColorMap(ris_element_square)

    return tau_0_star, tau_1, A_h, A_r, A_h_idx, A_r_idx


if __name__ == '__main__':

    # Initialize the RIS system
    RIS_System = RISNomaSystem(
        numRISelements=200,
        E_RIS_TH=1e-6
    )

    tau_0, tau_1, A_h, A_r, A_h_idx, A_r_idx = HarvestingElementOptimizer(RIS_System,
                                                                          verbose=True,
                                                                          plot=True)

    # Update RIS_System values to reflect the devices charging
    RIS_System.tau_0 = tau_0
    RIS_System.tau_1 = tau_1

    # Updating these values will alter the shapes of the Channel Matrices.
    # Should they remain the same shape?
    #RIS_System.A_h = A_h
    #RIS_System.A_r = A_r

    # Update RIS_System Channel Matrices to reflect the devices charging
    RIS_System.H[A_h_idx, :] = RIS_System.H[A_h_idx, :] - RIS_System.E_RIS_TH / RIS_System.numAntennasAP
    RIS_System.G[:, A_h_idx] = RIS_System.G[:, A_h_idx] - RIS_System.E_RIS_TH / RIS_System.numAntennasPS
    RIS_System.g_r[:, A_h_idx] = RIS_System.g_r[:, A_h_idx] - RIS_System.E_RIS_TH / RIS_System.numIoTs
    RIS_System.h_r[A_h_idx, :] = RIS_System.h_r[A_h_idx, :] - RIS_System.E_RIS_TH / RIS_System.numIoTs

    # Recalculate Phi and Psi Matrices (they're deterministic upon Channel Matrices)
    RIS_System.reflection_Phi()
    RIS_System.reflection_Psi()

    '''
        This is where you would perform the rest of the problem: Subproblem 1 and Subproblem 2.
    '''




