import numpy as np
from numpy import sqrt, pi, sin, cos, multiply, linspace, ones, zeros, \
    log10, transpose, exp, arange, concatenate, abs, sort, \
    sum, array, argsort, diag, matrix, argwhere, ndarray
from numpy.random import rand, randn, seed
import matplotlib.pyplot as plt


# Path loss Line-of-Sight (LOS) function
def pathloss_LOS(d):
    # d=d/1000
    # loss=89.5 + 16.9*log10(d)
    # loss=38.46 + 20*log10(d)
    loss = 35.6 + 22 * log10(d)
    return loss


# Path loss non-Line-of-Sight (NLOS) function
def pathloss_NLOS(d):
    # d=d/1000
    # loss=147.4 +43.3*log10(d)
    # loss=max(15.3 +37.6*log10(d), pathloss_LOS( d ))+20
    # loss=max(2.7 +42.8*log10(d), pathloss_LOS( d ))+20
    loss = 32.6 + 36.7 * log10(d)
    return loss


# Uniform Linear Array (ULA) transpose function
def ULA_fun(phi, N):
    # h=exp( 1j * pi * sin(phi) .* (0:N-1)')
    h = np.exp(1j * pi * sin(phi) * transpose([arange(0, N)]))
    return h

# Generate System Values
def defaultSystemParameters(eta_RIS=0.8,
                            numAntennasPS=10,numAntennasAP=10,numRISelements=144,
                            numIoTs=4,
                            PowerStation_dBm=30,
                            noisePower_dB=-100,
                            PS_loc=[-10.000001, 0.000001],AP_loc=[10.000001, 0.000001],
                            PS_angle=rand(1, 1),AP_angle=rand(1, 1),RIS_angle=rand(1, 1),
                            eb=10,
                            center=[0, 0],radius=4,):
    """
    Generates System Parameters to be used with the RIS assisted communication system

    :return:
    """
    eb1, eb2 = ricianChannel(eb)
    return_obj: dict = dict(eta_RIS=eta_RIS,
                      numAntennasPS=numAntennasPS,
                      numAntennasAP=numAntennasAP,
                      numRISelements=numRISelements,
                      numIoTs=numIoTs,
                      PowerStation_lin=PowerStation_lin(PowerStation_dBm, eta_RIS),
                      noise_power=noise_power(noisePower_dB),
                      PS_loc=PS_loc,
                      AP_loc=AP_loc,
                      PS_angle=PS_angle,
                      AP_angle=AP_angle,
                      RIS_angle=RIS_angle,
                      eb1=eb1,
                      eb2=eb2,
                      IoT_loc=generateIoTLocations(center, radius, numIoTs),
                      ULA_RIS=ULA_fun(RIS_angle, numRISelements),
                      ULA_AP=ULA_fun(AP_angle, numAntennasAP),
                      ULA_PS=ULA_fun(PS_angle, numAntennasPS),
                            )
    return return_obj


# Base Station Power = Power at transmitter (Ptx)
def PowerStation_lin(power_station_dBm, eta_RIS):
    PowerStation_dB = power_station_dBm - 30
    PowerStation_lin = 10 ** (PowerStation_dB / 10)  # P_o to linear
    print(f'\nPowerStation_lin: {PowerStation_lin}')
    print(f'Po*etaRIS: {PowerStation_lin * eta_RIS}')
    return PowerStation_lin


# noise power (sigma)
def noise_power(noisePower_dB):
    noise_power = 10 ** (noisePower_dB / 10)
    print(f'\nnoise_power: {noise_power}')
    return noise_power


# channel generation Rician factor
# https://www.mathworks.com/help/comm/ug/fading-channels.html
def ricianChannel(eb):
    eb2 = 1 / (1 + eb)
    eb1 = 1 - eb2
    # strength of LoS
    eb1 = sqrt(eb1)
    # strength of NLoS
    eb2 = sqrt(eb2)
    return eb1, eb2


# Generate IoT locations within radius
def generateIoTLocations(center, radius, numIoTs):
    angle = 2 * pi * rand(numIoTs, 1)
    r = radius * sqrt(rand(numIoTs, 1))
    x = r * cos(angle) + center[0]
    y = r * sin(angle) + center[1]
    IoT_loc = np.block([x, y])
    return IoT_loc


# Plotting function
def plotMap(IoT_loc, iot_center, iot_radius, RIS_loc, PS_loc, AP_loc):
    # plot IoTs and circle
    plt.scatter(IoT_loc[:, 0], IoT_loc[:, 1], label='IoTs')
    circle = plt.Circle(iot_center, iot_radius, color='b', fill=False)
    plt.gca().add_patch(circle)
    # plot RIS, PS, and AP
    plt.plot(RIS_loc[0], RIS_loc[1], marker='D', color='purple', label='RIS')
    plt.plot(PS_loc[0], PS_loc[1], marker='+', color='red', label='Power Station')
    plt.plot(AP_loc[0], AP_loc[1], marker='*', color='green', label='Access Point')
    plt.title('Geographical Locations')

    # PS Diagram Arrows pyplot.arrow(x, y, dx, dy) draws an arrow (x,y) ---> (x+dx, y+dy)
    head_width = 3
    len_reduct = 0.02
    annot_relative = 4

    # PS --> RIS
    plt.arrow(PS_loc[0], PS_loc[1],
              (1 - len_reduct) * (RIS_loc[0] - PS_loc[0]),
              (1 - len_reduct) * (RIS_loc[1] - PS_loc[1]),
              length_includes_head=True,
              head_width=head_width,
              color='cornflowerblue')
    plt.annotate("G", color="cornflowerblue", fontsize=15,
                 xy=(PS_loc[0] + 0.5 * (RIS_loc[0] - PS_loc[0]), PS_loc[1] + 0.5 * (RIS_loc[1] - PS_loc[1])),
                 xytext=(-annot_relative, annot_relative),
                 textcoords='offset points')

    # PS --> IoTs
    dist_PS_iot_radius = 1 - iot_radius / np.linalg.norm(np.array(iot_center) - np.array(PS_loc))
    plt.arrow(PS_loc[0], PS_loc[1],
              (1 - len_reduct) * (dist_PS_iot_radius) * (iot_center[0] - PS_loc[0]),
              (1 - len_reduct) * (dist_PS_iot_radius) * (iot_center[1] - PS_loc[1]),
              length_includes_head=True,
              head_width=head_width,
              color='cornflowerblue')
    plt.annotate("g_d", color="cornflowerblue", fontsize=15,
                 xy=(PS_loc[0] + 0.5 * (iot_center[0] - PS_loc[0]), PS_loc[1] + 0.5 * (iot_center[1] - PS_loc[1])),
                 xytext=(-annot_relative, annot_relative),
                 textcoords='offset points')

    # RIS --> IoTs
    dist_RIS_IoTs_iot_radius = iot_radius / np.linalg.norm(np.array(iot_center) - np.array(RIS_loc))
    plt.arrow(RIS_loc[0] - 1, RIS_loc[1],
              (1 - len_reduct) * (1 - dist_RIS_IoTs_iot_radius) * (iot_center[0] - RIS_loc[0]) - 1,
              (1 - len_reduct) * (1 - dist_RIS_IoTs_iot_radius) * (iot_center[1] - RIS_loc[1]),
              length_includes_head=True,
              head_width=head_width,
              color='cornflowerblue')
    plt.annotate("g_r", color="cornflowerblue", fontsize=15,
                 xy=(RIS_loc[0] + 0.5 * (iot_center[0] - RIS_loc[0]), RIS_loc[1] + 0.5 * (iot_center[1] - RIS_loc[1])),
                 xytext=(-8 * annot_relative, 0),
                 textcoords='offset points')

    # IoTs --> RIS
    theta_IoTs_RIS = np.arctan((RIS_loc[1] - iot_center[1]) / (RIS_loc[0] - iot_center[0]))
    x_IoTs_rad_RIS = -iot_radius * np.cos(theta_IoTs_RIS)
    y_IoTs_rad_RIS = -iot_radius * np.sin(theta_IoTs_RIS)
    plt.arrow(iot_center[0] + x_IoTs_rad_RIS + 1,
              iot_center[1] + y_IoTs_rad_RIS,
              (1 - len_reduct) * (1 - dist_RIS_IoTs_iot_radius) * (RIS_loc[0] - iot_center[0]) + 1,
              (1 - len_reduct) * (1 - dist_RIS_IoTs_iot_radius) * (RIS_loc[1] - iot_center[1]),
              length_includes_head=True,
              head_width=head_width,
              color='brown')
    plt.annotate("h_r", color="brown", fontsize=15,
                 xy=(iot_center[0] + 0.5 * (RIS_loc[0] - iot_center[0]),
                     iot_center[1] + 0.5 * (RIS_loc[1] - iot_center[1])),
                 xytext=(2 * annot_relative, 0),
                 textcoords='offset points')

    # IoTs --> AP
    dist_IoTs_radius_AP = iot_radius / np.linalg.norm(np.array(iot_center) - np.array(AP_loc))
    theta_IoTs_AP = np.arctan((AP_loc[1] - iot_center[1]) / (AP_loc[0] - iot_center[0]))
    x_IoTs_rad_AP = iot_radius * np.cos(theta_IoTs_AP)
    y_IoTs_rad_AP = iot_radius * np.sin(theta_IoTs_AP)
    plt.arrow(iot_center[0] + x_IoTs_rad_AP, iot_center[1] + y_IoTs_rad_AP,
              (1 - len_reduct) * (1 - dist_IoTs_radius_AP) * (AP_loc[0] - iot_center[0]),
              (1 - len_reduct) * (1 - dist_IoTs_radius_AP) * (AP_loc[1] - iot_center[1]),
              length_includes_head=True,
              head_width=head_width,
              color='brown')
    plt.annotate("h_d", color="brown", fontsize=15,
                 xy=(
                     iot_center[0] + 0.5 * (AP_loc[0] - iot_center[0]),
                     iot_center[1] + 0.5 * (AP_loc[1] - iot_center[1])),
                 xytext=(-annot_relative, annot_relative),
                 textcoords='offset points')

    # RIS --> AP
    plt.arrow(RIS_loc[0], RIS_loc[1],
              (1 - len_reduct) * (AP_loc[0] - RIS_loc[0]),
              (1 - len_reduct) * (AP_loc[1] - RIS_loc[1]),
              length_includes_head=True,
              head_width=head_width,
              color='brown')
    plt.annotate("H", color="brown", fontsize=15,
                 xy=(RIS_loc[0] + 0.5 * (AP_loc[0] - RIS_loc[0]), RIS_loc[1] + 0.5 * (AP_loc[1] - RIS_loc[1])),
                 xytext=(-annot_relative, annot_relative),
                 textcoords='offset points')

    plt.title('Geographical Locations')
    plt.axis('equal')
    plt.legend()
    plt.grid()
    plt.show()


# Populating channels related to RIS, AP, and PS, and IoTs
# LOS link: RIS to AP (H channel matrix value)
def ch_RIS_to_AP(numRISelements, numAntennasAP, RIS_loc, AP_loc, eb1, eb2, ULA_RIS, ULA_AP):
    _ch_RIS_to_AP = zeros((numRISelements, numAntennasAP),
                          dtype=complex)  # ch RIS--> AP    [Matrix] (constant. complex)
    _dist_RIS_to_AP = sqrt((RIS_loc[0] - AP_loc[0]) ** 2 + (RIS_loc[1] - AP_loc[1]) ** 2)  # [scalar] (constant. real)
    _path_RIS_to_AP = 10 ** (-pathloss_LOS(_dist_RIS_to_AP) / 10)  # [scalar] (constant. real)
    _ch_RIS_to_AP_NLoS = (1 / sqrt(2)) * (
            randn(numRISelements, 1) + 1j * randn(numRISelements, 1))  # [Matrix] (constant. complex)
    _ch_RIS_to_AP = sqrt(_path_RIS_to_AP) * (
            eb1 * ULA_RIS @ ULA_AP.T + eb2 * _ch_RIS_to_AP_NLoS)  # [Matrix] (constant. complex)
    _H = _ch_RIS_to_AP
    print(f"H  : {_H.shape}")
    assert _H.shape == (numRISelements, numAntennasAP), f"H  : {_H.shape} should be ({numRISelements, numAntennasAP})"
    return _H


# LOS link: PS to RIS (G channel matrix)
def ch_PS_to_RIS(numAntennasPS, numRISelements, PS_loc, RIS_loc, eb1, eb2, ULA_RIS, ULA_PS):
    ch_PS_to_RIS = zeros((numAntennasPS, numRISelements),
                         dtype=complex)  # G ch  PS --> RIS [Matrix] (constant. complex) channel PS to RIS
    dist_PS_to_RIS = sqrt((PS_loc[0] - RIS_loc[0]) ** 2 + (PS_loc[1] - RIS_loc[1]) ** 2)  # [scalar] (constant. real)
    pathloss_PS_to_RIS = 10 ** (-pathloss_LOS(dist_PS_to_RIS) / 10)  # [scalar] (constant. real)
    ch_PS_to_RIS_NLoS = (1 / sqrt(2)) * (
            randn(numRISelements, 1) + 1j * randn(numRISelements, 1))  # [scalar] (constant. complex)
    ch_PS_to_RIS = (sqrt(pathloss_PS_to_RIS) * (eb1 * ULA_RIS @ (
        ULA_PS).T + eb2 * ch_PS_to_RIS_NLoS))  # [Matrix] (constant. complex) channel from PS to RIS
    G = ch_PS_to_RIS.T
    print(f"G  : {G.shape}")
    assert G.shape == (numAntennasPS, numRISelements), f"G  : {G.shape} should be ({numAntennasPS, numRISelements})"
    return G


# NLoS link: PS to IoTs g_d,k
def ch_PS_to_IoTs(numIoTs, numAntennasPS, PS_loc, IoT_loc):
    dist_PS_to_IoTs = zeros(
        (1, numIoTs))  # PS to IoTS                                       [vector] (constant. real) PS to IoTS
    pathloss_PS_to_IoTs = zeros((1,
                                 numIoTs))  # pathlos  between PS to IoTS                  [vector] (constant. real) pathloss line of sight PS to IoTS
    ch_PS_to_IoTs = zeros((numIoTs, numAntennasPS),
                          dtype=complex)  # channel  PS --> IoTS   [Matrix] (constant. complex) channel PS to IOTs
    ch_PS_to_IoTs_LoS = zeros((numAntennasPS,
                               numIoTs))  # LoS channel between PS --> IoTS   (unused.)[Matrix]  line of sight channel PS to IOTs
    ch_PS_to_IoTs_NLoS = zeros((numAntennasPS,
                                numIoTs))  # NLoS channel between PS --> IoTS  (unused.)[Matrix]  non-line of sight channel PS to IOTs
    dist_PS_to_IoTs[0, :] = sqrt((PS_loc[0] - IoT_loc[:, 0]) ** 2 + (PS_loc[1] - IoT_loc[:, 1]) ** 2)
    pathloss_PS_to_IoTs[0, :] = 10 ** (-pathloss_NLOS(dist_PS_to_IoTs[0, :]) / 10)
    ch_PS_to_IoTs = sqrt(pathloss_PS_to_IoTs[0, :]).reshape(4, 1) * (1 / sqrt(2)) * (
            rand(1, numAntennasPS) + 1j * rand(1, numAntennasPS))
    g_d = ch_PS_to_IoTs
    print(f"g_d: {g_d.shape}")
    assert g_d.shape == (numIoTs, numAntennasPS), f"g_d: {g_d.shape} should be ({numIoTs, numAntennasPS})"
    return g_d


# NLOS link: IoT_i to AP h_d,k
def ch_IoTs_to_AP(numAntennasAP, numIoTs, AP_loc, IoT_loc):
    dist_IoTs_to_AP = zeros(
        (1, numIoTs))  # IOTs to AP                                   [vector] (constant. real) distance IOTs to AP
    ch_IoTs_to_AP = zeros((numAntennasAP, numIoTs), dtype=complex)  # IoTs --> AP           [Matrix] (constant. complex)
    dist_IoTs_to_AP = sqrt((IoT_loc[:, 0] - AP_loc[0]) ** 2 + (IoT_loc[:, 1] - AP_loc[1]) ** 2)
    pathloss_IoTs_to_AP = 10 ** (-pathloss_LOS(dist_IoTs_to_AP) / 10)
    ch_IoTs_to_AP = (
            sqrt(pathloss_IoTs_to_AP) * (1 / sqrt(2)) * (randn(numAntennasAP, 1) + 1j * randn(numAntennasAP, 1)))
    #                         Use A[:,i] = b[:,0] when assigning vector b to entire coumn of matrix A
    # ch_IoTs_to_AP_LoS = sqrt(path_IoT1_to_AP)*(eb1.*ULA_fun(RIS_angle ,N)*ULA_fun(AP_angle ,M)'+eb2.*ch_IoT1_to_AP_NLoS)
    h_d = ch_IoTs_to_AP
    print(f"h_d: {h_d.shape}")
    assert h_d.shape == (numAntennasAP, numIoTs), f"h_d: {h_d.shape} should be ({numAntennasAP, numIoTs})"
    return h_d


# g_r,k channel from RIS to IoTs
def ch_RIS_to_IoTs(numRISelements, numIoTs, RIS_loc, IoT_loc, eb1, eb2, ULA_RIS):
    dist_RIS_to_IoTs = zeros((1, numIoTs))  # distance from RIS to individual IOTs          [vector] (constant. real)
    pathloss_RIS_to_IoTs = zeros((1, numIoTs))  # pathloss of RIS to individual IOTs        [vector] (constant. real)
    ch_RIS_to_IoTs_NLoS = zeros((numRISelements, numIoTs),
                                dtype=complex)  # NLoS ch RIS-->IoTs [Matrix] (constant. complex)
    ch_RIS_to_IoTs = zeros((numRISelements, numIoTs),
                           dtype=complex)  # ch RIS-->IoTs        [Matrix] (constant. complex)
    dist_RIS_to_IoTs = sqrt(
        (RIS_loc[0] - IoT_loc[:, 0]) ** 2 + (
                RIS_loc[1] - IoT_loc[:, 1]) ** 2)  # calculates distance from RIS to individual IOTs
    pathloss_RIS_to_IoTs = 10 ** (-pathloss_LOS(dist_RIS_to_IoTs) / 10)
    ch_RIS_to_IoTs_NLoS = (
            sqrt(pathloss_RIS_to_IoTs) * (1 / sqrt(2)) * (randn(numRISelements, 1) + 1j * randn(numRISelements, 1)))
    ch_RIS_to_IoTs = (eb1 * ULA_RIS + eb2 * ch_RIS_to_IoTs_NLoS)  # [:, 0] #channel from RIS to IoTs
    g_r = ch_RIS_to_IoTs.T
    print(f"g_r: {g_r.shape}")
    assert g_r.shape == (numIoTs, numRISelements), f"g_r: {g_r.shape} should be ({numIoTs, numRISelements})"
    return g_r


# LOS link: IoTs to RIS (h_r,k channel from RIS to IoT_i) Ricing fading channel
def ch_IoTs_to_RIS(numRISelements, numIoTs, RIS_loc, IoT_loc, eb1, eb2, ULA_RIS):
    dist_RIS_to_IoTs = zeros((1, numIoTs))  # distance from RIS to individual IOTs          [vector] (constant. real)
    pathloss_RIS_to_IoTs = zeros((1, numIoTs))  # pathloss of RIS to individual IOTs        [vector] (constant. real)
    ch_IoTs_to_RIS = zeros((numRISelements, numIoTs),
                           dtype=complex)  # ch IoTs-->RIS         [Matrix] (constant. complex)
    ch_IoTs_to_RIS_NLoS = zeros((numRISelements, numIoTs),
                                dtype=complex)  # ch IoTs-->RIS   [Matrix] (constant. complex)
    dist_RIS_to_IoTs = sqrt((RIS_loc[0] - IoT_loc[:, 0]) ** 2 + (RIS_loc[1] - IoT_loc[:, 1]) ** 2)
    pathloss_RIS_to_IoTs = 10 ** (-pathloss_LOS(dist_RIS_to_IoTs) / 10)
    ch_IoTs_to_RIS_NLoS = ((1 / sqrt(2)) * (randn(numRISelements, 1) + 1j * randn(numRISelements, 1)))
    ch_IoTs_to_RIS = (sqrt(pathloss_RIS_to_IoTs) * (eb1 * ULA_RIS + eb2 * ch_IoTs_to_RIS_NLoS))
    h_r = ch_IoTs_to_RIS
    print(f"h_r: {h_r.shape}")
    assert h_r.shape == (numRISelements, numIoTs), f"h_r: {h_r.shape} should be ({numRISelements, numIoTs})"
    return h_r
