from typing import Dict, Any, List

import numpy as np
from numpy import sqrt, pi, sin, cos, multiply, linspace, ones, zeros, \
    log10, transpose, exp, arange, concatenate, abs, sort, \
    sum, array, argsort, diag, matrix, argwhere, ndarray
from numpy.random import rand, randn, seed
import matplotlib.pyplot as plt


class RISNomaSystem:
    """
    Class object to hold System Parameters and Channel Values
    """

    def __init__(self,
                 eta_RIS=0.8,
                 numAntennasPS=10,
                 numAntennasAP=10,
                 numRISelements=144,
                 numIoTs=4,
                 PowerStation_dBm=30,
                 noisePower_dB=-100,
                 E_RIS_TH=1e-6,
                 PS_loc=[-10.000001, 0.000001],
                 AP_loc=[10.000001, 0.000001],
                 RIS_loc=[-2, 6],
                 PS_angle=rand(1, 1),
                 AP_angle=rand(1, 1),
                 RIS_angle=rand(1, 1),
                 eb=10,
                 center=[0, 0],
                 radius=4,
                 verbose=True, plot=True):

        self.eta_RIS = eta_RIS
        self.numAntennasPS = numAntennasPS
        self.numAntennasAP = numAntennasAP
        self.numRISelements = numRISelements
        self.numIoTs = numIoTs
        self.PowerStation_dBm = PowerStation_dBm
        self.PowerStation_lin = self.PowerStation_lin(self.PowerStation_dBm)
        self.noisePower_dB = noisePower_dB
        self.noise_power = self.noise_power(self.noisePower_dB)
        self.E_RIS_TH = E_RIS_TH
        self.PS_loc = PS_loc
        self.AP_loc = AP_loc
        self.RIS_loc = RIS_loc
        self.PS_angle = PS_angle
        self.AP_angle = AP_angle
        self.RIS_angle = RIS_angle
        self.eb = eb
        self.eb1, self.eb2 = self.ricianChannel(eb)
        self.center = center
        self.radius = radius

        self.tau_0 = 1
        self.tau_1 = 1

        self.IoT_loc = self.generateIoTLocations()
        self.ULA_RIS = self.ULA_fun(RIS_angle, numRISelements)
        self.ULA_AP = self.ULA_fun(AP_angle, numAntennasAP)
        self.ULA_PS = self.ULA_fun(PS_angle, numAntennasPS)

        self.H = self.ch_RIS_to_AP()  # LOS link: H
        self.G = self.ch_PS_to_RIS()  # LOS link: G
        self.g_d = self.ch_PS_to_IoTs()  # NLoS link: g_d
        self.h_d = self.ch_IoTs_to_AP()  # NLOS link: h_d
        self.g_r = self.ch_RIS_to_IoTs()  # g_r
        self.h_r = self.ch_IoTs_to_RIS()  # LOS link: h_r   Ricing fading channel

        self.Phi = self.reflection_Phi()
        self.Psi = self.reflection_Psi()

        self.A_r = numRISelements  # Default Values. Will change when running Optimize_RIS_Elements.py
        self.A_h = numRISelements

        if verbose:
            self.printSystemParams()
        if plot:
            self.plotMap()

    def printSystemParams(self):
        print('RIS NOMA Wireless Power/Transmission System Parameters:')
        print(f'\nPowerStation_lin: {self.PowerStation_lin}')
        print(f'Po*eta_RIS: {self.PowerStation_lin * self.eta_RIS}')
        print(f'noise_power: {self.noise_power}')
        print('eta_RIS: ', self.eta_RIS)
        print('\nnumAntennasPS: ', self.numAntennasPS)
        print('numAntennasAP: ', self.numAntennasAP)
        print('numRISelements: ', self.numRISelements)
        print('numIoTs: ', self.numIoTs)
        print('\nPS_angle: ', self.PS_angle)
        print('AP_angle: ', self.AP_angle)
        print('RIS_angle: ', self.RIS_angle)
        print('\nRician channel:\neb1:', self.eb1, '\neb2', self.eb2)
        print('\nIoT locations:\n', self.IoT_loc)

    def pathloss_LOS(self, d):
        '''Path loss Line-of-Sight (LOS) function'''
        # d=d/1000
        # loss=89.5 + 16.9*log10(d)
        # loss=38.46 + 20*log10(d)
        loss = 35.6 + 22 * log10(d)
        return loss

    def pathloss_NLOS(self, d):
        '''Path loss non-Line-of-Sight (NLOS) function'''
        # d=d/1000
        # loss=147.4 +43.3*log10(d)
        # loss=max(15.3 +37.6*log10(d), pathloss_LOS( d ))+20
        # loss=max(2.7 +42.8*log10(d), pathloss_LOS( d ))+20
        loss = 32.6 + 36.7 * log10(d)
        return loss

    def ULA_fun(self, phi, n):
        """
        Uniform Linear Array (ULA) transpose function
        """
        # h=exp( 1j * pi * sin(phi) .* (0:N-1)')
        h = np.exp(1j * pi * sin(phi) * transpose([arange(0, n)]))
        return h

    def PowerStation_lin(self, power_station_dBm):
        """
        Base Station Power = Power at transmitter (Ptx)
        """

        PowerStation_lin = 10 ** ((power_station_dBm - 30) / 10)  # dBm to linear
        return PowerStation_lin

    def noise_power(self, noisePower_dB):
        """
        noise power (sigma)
        """
        noise_power = 10 ** (noisePower_dB / 10)
        return noise_power

    def ricianChannel(self, eb):
        '''
        channel generation Rician factor
        https://www.mathworks.com/help/comm/ug/fading-channels.html
        '''
        eb2 = 1 / (1 + eb)
        eb1 = 1 - eb2
        # strength of LoS
        eb1 = sqrt(eb1)
        # strength of NLoS
        eb2 = sqrt(eb2)
        self.eb1, self.eb2 = eb1, eb2
        return self.eb1, self.eb2

    def generateIoTLocations(self):
        '''
        Generate IoT locations within radius
        '''
        angle = 2 * pi * rand(self.numIoTs, 1)
        r = self.radius * sqrt(rand(self.numIoTs, 1))
        x = r * cos(angle) + self.center[0]
        y = r * sin(angle) + self.center[1]
        IoT_loc = np.block([x, y])
        return IoT_loc

    def plotMap(self):
        '''
        Plotting function
        :param IoT_loc:
        :param iot_center:
        :param iot_radius:
        :param RIS_loc:
        :param PS_loc:
        :param AP_loc:
        :return:
        '''
        IoT_loc = self.IoT_loc
        iot_center = self.center
        iot_radius = self.radius
        RIS_loc = self.RIS_loc
        PS_loc = self.PS_loc
        AP_loc = self.AP_loc

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
        head_width = 0.3
        len_reduct = 0.02
        annot_relative = 4

        # PS --> RIS
        plt.arrow(PS_loc[0], PS_loc[1],(1 - len_reduct) * (RIS_loc[0] - PS_loc[0]),(1 - len_reduct) * (RIS_loc[1] - PS_loc[1]),length_includes_head=True,head_width=head_width,color='cornflowerblue')

        plt.annotate("G", color="cornflowerblue", fontsize=15,xy=(PS_loc[0] + 0.5 * (RIS_loc[0] - PS_loc[0]), PS_loc[1] + 0.5 * (RIS_loc[1] - PS_loc[1])),xytext=(-annot_relative, annot_relative),textcoords='offset points')

        # PS --> IoTs
        dist_PS_iot_radius = 1 - iot_radius / np.linalg.norm(np.array(iot_center) - np.array(PS_loc))
        plt.arrow(PS_loc[0], PS_loc[1],(1 - len_reduct) * (dist_PS_iot_radius) * (iot_center[0] - PS_loc[0]),(1 - len_reduct) * (dist_PS_iot_radius) * (iot_center[1] - PS_loc[1]),length_includes_head=True,head_width=head_width,color='cornflowerblue')
        plt.annotate("g_d", color="cornflowerblue", fontsize=15,xy=(PS_loc[0] + 0.5 * (iot_center[0] - PS_loc[0]), PS_loc[1] + 0.5 * (iot_center[1] - PS_loc[1])),xytext=(-annot_relative, annot_relative),textcoords='offset points')

        # RIS --> IoTs
        dist_RIS_IoTs_iot_radius = iot_radius / np.linalg.norm(np.array(iot_center) - np.array(RIS_loc))
        plt.arrow(x=RIS_loc[0],
                  y=RIS_loc[1],
                  dx=(1 - len_reduct) * (iot_center[0] - RIS_loc[0]),
                  dy=(1 - len_reduct) * (iot_center[1] - RIS_loc[1]),
                  length_includes_head=True,head_width=head_width,color='cornflowerblue')
        plt.annotate("g_r", color="cornflowerblue", fontsize=15,xy=(RIS_loc[0] + 0.5 * (iot_center[0] - RIS_loc[0]), RIS_loc[1] + 0.5 * (iot_center[1] - RIS_loc[1])),xytext=(-8 * annot_relative, 0),textcoords='offset points')

        # IoTs --> RIS

        plt.arrow(x=iot_center[0],
                  y=iot_center[1],
                  dx=(1 - len_reduct) * (RIS_loc[0] - iot_center[0]),
                  dy=(1 - len_reduct) * (RIS_loc[1] - iot_center[1]),
                  length_includes_head=True, head_width=head_width, color='brown')

        plt.annotate("h_r", color="brown", fontsize=15,xy=(iot_center[0] + 0.5 * (RIS_loc[0] - iot_center[0]),    iot_center[1] + 0.5 * (RIS_loc[1] - iot_center[1])),xytext=(2 * annot_relative, 0),textcoords='offset points')

        # IoTs --> AP
        dist_IoTs_radius_AP = iot_radius / np.linalg.norm(np.array(iot_center) - np.array(AP_loc))
        theta_IoTs_AP = np.arctan((AP_loc[1] - iot_center[1]) / (AP_loc[0] - iot_center[0]))
        x_IoTs_rad_AP = iot_radius * np.cos(theta_IoTs_AP)
        y_IoTs_rad_AP = iot_radius * np.sin(theta_IoTs_AP)
        plt.arrow(iot_center[0] + x_IoTs_rad_AP, iot_center[1] + y_IoTs_rad_AP,(1 - len_reduct) * (1 - dist_IoTs_radius_AP) * (AP_loc[0] - iot_center[0]),(1 - len_reduct) * (1 - dist_IoTs_radius_AP) * (AP_loc[1] - iot_center[1]),length_includes_head=True,head_width=head_width,color='brown')
        plt.annotate("h_d", color="brown", fontsize=15,xy=(    iot_center[0] + 0.5 * (AP_loc[0] - iot_center[0]),    iot_center[1] + 0.5 * (AP_loc[1] - iot_center[1])),xytext=(-annot_relative, annot_relative),textcoords='offset points')

        # RIS --> AP
        plt.arrow(RIS_loc[0], RIS_loc[1],(1 - len_reduct) * (AP_loc[0] - RIS_loc[0]),(1 - len_reduct) * (AP_loc[1] - RIS_loc[1]),length_includes_head=True,head_width=head_width,color='brown')
        plt.annotate("H", color="brown", fontsize=15,xy=(RIS_loc[0] + 0.5 * (AP_loc[0] - RIS_loc[0]), RIS_loc[1] + 0.5 * (AP_loc[1] - RIS_loc[1])),xytext=(-annot_relative, annot_relative),textcoords='offset points')

        plt.title('Geographical Locations')
        plt.axis('equal')
        plt.legend()
        plt.grid()
        plt.show()

    def ch_RIS_to_AP(self):
        '''
        LOS link: RIS to AP (H channel matrix value)

        shape: (numRISelements, numAntennasAP)

        :param numRISelements:
        :param numAntennasAP:
        :param RIS_loc:
        :param AP_loc:
        :param eb1:
        :param eb2:
        :param ULA_RIS:
        :param ULA_AP:
        :return:
        '''
        _ch_RIS_to_AP = zeros((self.numRISelements, self.numAntennasAP),dtype=complex)  # ch RIS--> AP    [Matrix] (constant. complex)
        _dist_RIS_to_AP = sqrt((self.RIS_loc[0] - self.AP_loc[0]) ** 2 + (self.RIS_loc[1] - self.AP_loc[1]) ** 2)  # [scalar] (constant. real)
        _path_RIS_to_AP = 10 ** (-self.pathloss_LOS(_dist_RIS_to_AP) / 10)  # [scalar] (constant. real)
        _ch_RIS_to_AP_NLoS = (1 / sqrt(2)) * (randn(self.numRISelements, 1) + 1j * randn(self.numRISelements, 1))  # [Matrix] (constant. complex)
        _ch_RIS_to_AP = sqrt(_path_RIS_to_AP) * (self.eb1 * self.ULA_RIS @ np.transpose(self.ULA_AP) + self.eb2 * _ch_RIS_to_AP_NLoS)  # [Matrix] (constant. complex)

        _H = _ch_RIS_to_AP
        #print(f"H  : {_H.shape}")
        assert _H.shape == (self.numRISelements, self.numAntennasAP), f"H  : {_H.shape} should be ({self.numRISelements, self.numAntennasAP})"
        return _H

    def ch_PS_to_RIS(self):
        '''
        LOS link: PS to RIS (G channel matrix)

        shape: [numAntennasPS, numRISelements]

        :param numAntennasPS:
        :param numRISelements:
        :param PS_loc:
        :param RIS_loc:
        :param eb1:
        :param eb2:
        :param ULA_RIS:
        :param ULA_PS:
        :return:
        '''
        ch_PS_to_RIS = zeros((self.numAntennasPS, self.numRISelements),
                             dtype=complex)  # G ch  PS --> RIS [Matrix] (constant. complex) channel PS to RIS
        dist_PS_to_RIS = sqrt(
            (self.PS_loc[0] - self.RIS_loc[0]) ** 2 + (self.PS_loc[1] - self.RIS_loc[1]) ** 2)  # [scalar] (constant. real)
        pathloss_PS_to_RIS = 10 ** (-self.pathloss_LOS(dist_PS_to_RIS) / 10)  # [scalar] (constant. real)
        ch_PS_to_RIS_NLoS = (1 / sqrt(2)) * (
                randn(self.numRISelements, 1) + 1j * randn(self.numRISelements, 1))  # [scalar] (constant. complex)
        ch_PS_to_RIS = (sqrt(pathloss_PS_to_RIS) * (self.eb1 * self.ULA_RIS @ np.transpose(self.ULA_PS) + self.eb2 * ch_PS_to_RIS_NLoS))  # [Matrix] (constant. complex) channel from PS to RIS
        G = ch_PS_to_RIS.T
        #print(f"G  : {G.shape}")
        assert G.shape == (self.numAntennasPS, self.numRISelements), f"G  : {G.shape} should be ({self.numAntennasPS, self.numRISelements})"
        return G

    def ch_PS_to_IoTs(self):
        '''
        g_d NLoS link: PS to IoTs

        shape: (numIoTs, numAntennasPS)

        :param numIoTs:
        :param numAntennasPS:
        :param PS_loc:
        :param IoT_loc:
        :return:
        '''
        dist_PS_to_IoTs = zeros(
            (1, self.numIoTs))  # PS to IoTS                                       [vector] (constant. real) PS to IoTS
        pathloss_PS_to_IoTs = zeros((1,
                                     self.numIoTs))  # pathlos  between PS to IoTS                  [vector] (constant. real) pathloss line of sight PS to IoTS
        ch_PS_to_IoTs = zeros((self.numIoTs, self.numAntennasPS),
                              dtype=complex)  # channel  PS --> IoTS   [Matrix] (constant. complex) channel PS to IOTs
        ch_PS_to_IoTs_LoS = zeros((self.numAntennasPS,
                                   self.numIoTs))  # LoS channel between PS --> IoTS   (unused.)[Matrix]  line of sight channel PS to IOTs
        ch_PS_to_IoTs_NLoS = zeros((self.numAntennasPS,
                                    self.numIoTs))  # NLoS channel between PS --> IoTS  (unused.)[Matrix]  non-line of sight channel PS to IOTs
        dist_PS_to_IoTs[0, :] = sqrt((self.PS_loc[0] - self.IoT_loc[:, 0]) ** 2 + (self.PS_loc[1] - self.IoT_loc[:, 1]) ** 2)
        pathloss_PS_to_IoTs[0, :] = 10 ** (-self.pathloss_NLOS(dist_PS_to_IoTs[0, :]) / 10)
        ch_PS_to_IoTs = sqrt(pathloss_PS_to_IoTs[0, :]).reshape(self.numIoTs, 1) * (1 / sqrt(2)) * (
                rand(1, self.numAntennasPS) + 1j * rand(1, self.numAntennasPS))
        g_d = ch_PS_to_IoTs
        #print(f"g_d: {g_d.shape}")
        assert g_d.shape == (self.numIoTs, self.numAntennasPS), f"g_d: {g_d.shape} should be ({self.numIoTs, self.numAntennasPS})"
        return g_d

    def ch_IoTs_to_AP(self):
        '''
        NLOS link: IoT_i to AP h_d,k
        :param numIoTs:
        :param AP_loc:
        :param IoT_loc:
        :return:
        '''
        dist_IoTs_to_AP = zeros(
            (1, self.numIoTs))  # IOTs to AP                                   [vector] (constant. real) distance IOTs to AP
        ch_IoTs_to_AP = zeros((self.numAntennasAP, self.numIoTs),
                              dtype=complex)  # IoTs --> AP           [Matrix] (constant. complex)
        dist_IoTs_to_AP = sqrt((self.IoT_loc[:, 0] - self.AP_loc[0]) ** 2 + (self.IoT_loc[:, 1] - self.AP_loc[1]) ** 2)
        pathloss_IoTs_to_AP = 10 ** (-self.pathloss_LOS(dist_IoTs_to_AP) / 10)
        ch_IoTs_to_AP = (
                sqrt(pathloss_IoTs_to_AP) * (1 / sqrt(2)) * (randn(self.numAntennasAP, 1) + 1j * randn(self.numAntennasAP, 1)))
        #                         Use A[:,i] = b[:,0] when assigning vector b to entire coumn of matrix A
        # ch_IoTs_to_AP_LoS = sqrt(path_IoT1_to_AP)*(eb1.*ULA_fun(RIS_angle ,N)*ULA_fun(AP_angle ,M)'+eb2.*ch_IoT1_to_AP_NLoS)
        h_d = ch_IoTs_to_AP
        #print(f"h_d: {h_d.shape}")
        assert h_d.shape == (self.numAntennasAP, self.numIoTs), f"h_d: {h_d.shape} should be ({self.numAntennasAP, self.numIoTs})"
        return h_d

    def ch_RIS_to_IoTs(self):
        '''
        g_r,k channel from RIS to IoTs
        :param numIoTs:
        :param RIS_loc:
        :param IoT_loc:
        :param eb1:
        :param eb2:
        :param ULA_RIS:
        :return:
        '''
        dist_RIS_to_IoTs = zeros((1, self.numIoTs))  # distance from RIS to individual IOTs          [vector] (constant. real)
        pathloss_RIS_to_IoTs = zeros((1, self.numIoTs))  # pathloss of RIS to individual IOTs        [vector] (constant. real)
        ch_RIS_to_IoTs_NLoS = zeros((self.numRISelements, self.numIoTs),dtype=complex)  # NLoS ch RIS-->IoTs [Matrix] (constant. complex)
        ch_RIS_to_IoTs = zeros((self.numRISelements, self.numIoTs),dtype=complex)  # ch RIS-->IoTs        [Matrix] (constant. complex)
        dist_RIS_to_IoTs = sqrt((self.RIS_loc[0] - self.IoT_loc[:, 0]) ** 2 + (self.RIS_loc[1] - self.IoT_loc[:, 1]) ** 2)  # calculates distance from RIS to individual IOTs
        pathloss_RIS_to_IoTs = 10 ** (-self.pathloss_LOS(dist_RIS_to_IoTs) / 10)
        ch_RIS_to_IoTs_NLoS = (sqrt(pathloss_RIS_to_IoTs) * (1 / sqrt(2)) * (randn(self.numRISelements, 1) + 1j * randn(self.numRISelements, 1)))
        ch_RIS_to_IoTs = (self.eb1 * self.ULA_RIS + self.eb2 * ch_RIS_to_IoTs_NLoS)  # [:, 0] #channel from RIS to IoTs
        g_r = ch_RIS_to_IoTs.T
        #print(f"g_r: {g_r.shape}")
        assert g_r.shape == (self.numIoTs, self.numRISelements), f"g_r: {g_r.shape} should be ({self.numIoTs, self.numRISelements})"
        return g_r

    def ch_IoTs_to_RIS(self):
        """
        LOS link: IoTs to RIS (h_r,k channel from RIS to IoT_i) Rician fading channel

        shape: (numRISelements, numIoTs)

        :param numIoTs:
        :param RIS_loc:
        :param IoT_loc:
        :param eb1:
        :param eb2:
        :param ULA_RIS:
        :return:
        """
        dist_RIS_to_IoTs = zeros(
            (1, self.numIoTs))  # distance from RIS to individual IOTs          [vector] (constant. real)
        pathloss_RIS_to_IoTs = zeros(
            (1, self.numIoTs))  # pathloss of RIS to individual IOTs        [vector] (constant. real)
        ch_IoTs_to_RIS = zeros((self.numRISelements, self.numIoTs),
                               dtype=complex)  # ch IoTs-->RIS         [Matrix] (constant. complex)
        ch_IoTs_to_RIS_NLoS = zeros((self.numRISelements, self.numIoTs),
                                    dtype=complex)  # ch IoTs-->RIS   [Matrix] (constant. complex)
        dist_RIS_to_IoTs = sqrt((self.RIS_loc[0] - self.IoT_loc[:, 0]) ** 2 + (self.RIS_loc[1] - self.IoT_loc[:, 1]) ** 2)
        pathloss_RIS_to_IoTs = 10 ** (-self.pathloss_LOS(dist_RIS_to_IoTs) / 10)
        ch_IoTs_to_RIS_NLoS = ((1 / sqrt(2)) * (randn(self.numRISelements, 1) + 1j * randn(self.numRISelements, 1)))
        ch_IoTs_to_RIS = (sqrt(pathloss_RIS_to_IoTs) * (self.eb1 * self.ULA_RIS + self.eb2 * ch_IoTs_to_RIS_NLoS))
        h_r = ch_IoTs_to_RIS
        #print(f"h_r: {h_r.shape}")
        assert h_r.shape == (self.numRISelements, self.numIoTs), f"h_r: {h_r.shape} should be ({self.numRISelements, self.numIoTs})"
        self.h_r = h_r
        return h_r

    def reflection_Phi(self):
        """
        np.diag(h_r[:, i].T) @ H

        shape: phi_i = h_r_diag[numRISelements, numRISelements] x H[numRISelements, numAntennasAP] = [numRISelements, numAntennasAP]

        :param i:
        :param h_r:
        :param H:
        :return:
        """
        #Phi = np.zeros(shape=(self.numIoTs, self.numRISelements, self.numAntennasAP), dtype=complex)


        Phi = np.dstack([ np.diag(self.h_r[:, i].conj().T) @ self.H for i in range(self.numIoTs)])
        #print(f"Phi: {Phi.shape}")
        assert Phi.shape == (self.numRISelements, self.numAntennasAP, self.numIoTs), f"Phi: {Phi.shape} should be ({self.numRISelements, self.numAntennasAP, self.numIoTs})"
        self.Phi = Phi
        return self.Phi

    def reflection_Psi(self):
        """
        np.diag(g_d[i, :]) @ G.conv().T  shape(Ar, Mp)

        shape: psi_i = g_d_diag[numAntennasPS, numAntennasPS] x G_T[numRISelements, numAntennasPS] = [numRISelements, numAntennasPS]

        :param i:
        :param g_r:
        :param G:
        :return:
        """

        G_H = self.G.conj()

        Psi = np.dstack([np.diag(self.g_d[i, :]) @ G_H for i in range(self.numIoTs)])
        #print(f"Psi: {Psi.shape}")
        assert Psi.shape == (self.numAntennasPS, self.numRISelements, self.numIoTs), f"Psi: {Psi.shape} should be ({self.numAntennasPS, self.numRISelements,self.numIoTs})"
        self.Psi = Psi
        return self.Psi

    def regenerateRISChannels(self):
        """
        Generate and populate channels related to RIS, AP, and PS, and IoTs
        H:   ch_RIS_to_AP() -   Channel from RIS to AP
        h_d: ch_IoTs_to_AP() -  Channel from IoTs to AP
        h_r: ch_IoTs_to_RIS() - Channel from IoTs to RIS
        G:   ch_PS_to_RIS() -   Channel from PS to RIS
        g_d: ch_PS_to_IoTs() -  Channel from PS to IoTs
        g_r: ch_RIS_to_IoTs() - Channel from RIS to IoTs

        :return:
        """
        # LOS link: H
        self.H = self.ch_RIS_to_AP()
        # LOS link: G
        self.G = self.ch_PS_to_RIS()
        # NLoS link: g_d
        self.g_d = self.ch_PS_to_IoTs()
        # NLOS link: h_d
        self.h_d = self.ch_IoTs_to_AP()
        # g_r
        self.g_r = self.ch_RIS_to_IoTs()
        # LOS link: h_r   Ricing fading channel
        self.h_r = self.ch_IoTs_to_RIS()
        self.Phi = self.reflection_Phi()
        self.Psi = self.reflection_Psi()
        return self.H, self.h_d, self.h_r, self.G, self.g_d, self.g_r, self.Phi, self.Psi

    def getChannelValues(self):
        '''
            Helper function for getting necessary values for the experimental tensorflow graph code.
        :return:
        '''

        return_dict = {
            #'H': self.H,
            #'G': self.G,
            'h_d': self.h_d,
            #'h_r': self.h_r,
            'g_d': self.g_d,
            #'g_r': self.g_r,
            'Phi': self.Phi,
            'Psi': self.Psi,
            'tau_1': self.tau_1,
            'numIoTs': self.numIoTs,
            'sigma': self.noise_power,
            'eta_tau0_Po': self.eta_RIS * self.tau_0 * self.PowerStation_lin
        }
        return return_dict



if __name__ == "__main__":

    RIS_System = RISNomaSystem(numIoTs=16,
                               plot=True,
                               verbose=True)



