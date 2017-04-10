__author__ = 'xlibb'

import numpy as np
import wolff
import originalH


class geteffectiveH:
    """
    This is the class for the original Hamiltonian getting effective Hamiltonian
    """
    def __init__(self,Nx,Ny):
        """

        :return:
        """
        self.Nx=Nx
        self.Ny=Ny


    def fit_originalHsample(self,temperature,K,mcsetN):
        """
        Fitting n parameters using the linear regression
        :return Jn:
        """
        energy,correlations=originalH.energyVsCorrelation(temperature,K,self.Nx,self.Ny,mcsetN)
        Jlist=np.linalg.lstsq(correlations,energy)[0]
        return Jlist


    def iterativeWolff(self,temperaturelist,K,mcsetN):
        """
        Generate spin sample chain with Wolff algorithm
        :return:
        """
        hightemperature=temperaturelist[0]
        Jlist=self.fit_originalHsample(hightemperature,K,mcsetN)
        for temperature in temperaturelist[1:]:
            energy,correlations=wolff.energyVsCorrelation(temperature,Jlist[0],self.Nx,self.Ny,mcsetN)
            Jlist=np.linalg.lstsq(correlations,energy)[0]

        return Jlist


def train(K,mcsetN,Nx,Ny):
    """
    Train the model from high temperature to critical temperature using the reinforcement
    :param temperaturelist:
    :param K: four body interaction term
    :param mcsetN: number of Markov chain set
    :param Nx: size on the x direction
    :param Ny: size on the y direction
    :return: final Jlist
    """
    temperaturelist=np.linspace(5,2.49,50)
    trainer=geteffectiveH(Nx,Ny)
    Jlist=trainer.iterativeWolff(temperaturelist,K,mcsetN)
    print(Jlist)


train(0.2,mcsetN=10,Nx=10,Ny=10)






