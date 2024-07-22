import numpy as np
import matplotlib.pyplot as plt
import os
import time

if __name__ == "__main__":

    noise_power = 100

    cwd_path = os.path.dirname(os.path.abspath(__file__))
    print(cwd_path)
    folder_path = os.path.join(cwd_path, f"{noise_power}dBm_noise")
    print(folder_path)

    linear_power_list = [0.0001, 0.001, 0.1, 1, 2, 5, 10, 15, 17, 20, 25, 30, 40, 50, 55]
    del_list = []

    axis_means = []

    fig, ax = plt.subplots()
    # ax.set_xscale("log", nonpositive='clip')
    for i, lin_p in enumerate(linear_power_list):
        name = f'{lin_p}dBm.csv'
        save_path = os.path.join(folder_path, name)

        if not os.path.isfile(save_path):
            # remove from x axis
            del_list.append(i)

            continue
        # print(save_path)

        optimized_channel_values = np.loadtxt(save_path, delimiter=',')
        rate_sum_array = optimized_channel_values.flatten()
        rate_sum_array = rate_sum_array[~np.isnan(rate_sum_array)]

        axis_means.append(np.mean(rate_sum_array))

        n = len(rate_sum_array)
        x = np.ones(n) * lin_p
        ax.scatter(x, rate_sum_array, s=16)

    linear_power_list = np.delete(linear_power_list, del_list)
    print(len(linear_power_list))
    ax.plot(linear_power_list, axis_means, ls='-', c='red')
    plt.xlabel("Power (dBm)")
    plt.ylabel("Bitrate (b/s)")
    plt.title(f"noise_dBm=-{noise_power}")

    # plt.boxplot(x,vert=True)

    fig_name = f"noise_{noise_power}dBm_RIS.png"
    fig_save_path = os.path.join(folder_path, fig_name)
    print(fig_save_path)
    plt.savefig(fig_save_path, format="png")
    plt.show()
