__author__ = 'xlibb'

import numpy as np
import wolff
import originalH
import pickle
import Ising2D
import matplotlib.pyplot as plt


class geteffectiveH:
    """
    This is the class for the original Hamiltonian getting effective Hamiltonian
    """
    def __init__(self,Nx,Ny,Jo,Ko,nth=1):
        """

        :param nth: fit to the nth nearest neighbor
        :return:
        """
        self.Nx=Nx
        self.Ny=Ny
        self.Jo=Jo
        self.Ko=Ko
        self.nth=nth

    def fit_originalHsample(self,temperature,mcsetN):
        """
        Fitting n parameters using the linear regression
        :return Jn:
        """
        energy,correlations=originalH.energyVsCorrelation(temperature,self.Jo,self.Ko,self.Nx,self.Ny,mcsetN)
        A=np.hstack([correlations[:,0:self.nth],np.ones((len(energy),1))])
        Jlist,erro,_,_=np.linalg.lstsq(A,energy)
        return Jlist,erro

    def iterativeWolff(self,temperaturelist,mcsetN):
        """
        Generate spin sample chain with Wolff algorithm
        :return:
        """
        hightemperature=temperaturelist[0]

        Jlist,erro=self.fit_originalHsample(hightemperature,mcsetN)
        Jdata=np.zeros((len(temperaturelist),self.nth+1))
        errodata=np.zeros(len(temperaturelist))
        Jdata[0]=Jlist
        errodata[0]=erro

        for (idx,temperature) in enumerate(temperaturelist[1:]):
            spinsample_chain=wolff.spinsample_chain(temperature,abs(Jlist[0]),self.Nx,self.Ny,mcsetN)
            energy,correlations=originalH.energyVsCorrelation(temperature,self.Jo,self.Ko,self.Nx,self.Ny,mcsetN,spinsample_chain)
            # energy,correlations=wolff.energyVsCorrelation(temperature,Jlist[0],self.Nx,self.Ny,mcsetN)
            A=np.hstack([correlations[:,0:self.nth],np.ones((len(correlations),1))])
            Jlist,erro,_,_=np.linalg.lstsq(A,energy)
            Jdata[idx+1]=Jlist
            errodata[idx+1]=erro

        with open("data/Jdata.txt","wb") as f:
            pickle.dump(Jdata,f)
        with open("data/errodata.txt","wb") as f:
            pickle.dump(errodata,f)
        return Jlist

def loaddata(filename):
    with open("data/"+filename,"rb") as f:
        return pickle.load(f)

def train(J,K,Nx,Ny,temperaturehigh,temperaturelow,mcsetN,nth=1):
    """
    Train the model from high temperature to critical temperature using the reinforcement
    :param temperaturelist:
    :param K: four body interaction term
    :param mcsetN: number of Markov chain set
    :param Nx: size on the x direction
    :param Ny: size on the y direction
    :return: final Jlist
    """
    temperaturelist=np.linspace(temperaturehigh,temperaturelow,10)
    trainer=geteffectiveH(Nx,Ny,Jo=J,Ko=K,nth=nth)
    Jlist=trainer.iterativeWolff(temperaturelist,mcsetN)
    effectiveH=Ising2D.Ising2D(nxspins=Nx,nyspins=Ny,J=Jlist[0])
    return Jlist,effectiveH


def get_trainingdata():
    Jlist=loaddata("Jdata.txt")
    error=loaddata("errodata.txt")
    print(Jlist[:,0])
    plt.figure()
    plt.plot(Jlist[:,0])
    plt.show()
    plt.figure()
    plt.plot(error)
    plt.show()


if __name__=="__main__":
    # train(J=1,K=0.2,
    #       temperaturehigh=2.49+3,temperaturelow=2.49,
    #       mcsetN=100,Nx=10,Ny=10)

    get_trainingdata()





