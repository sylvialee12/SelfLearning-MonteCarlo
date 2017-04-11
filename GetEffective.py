__author__ = 'xlibb'

import numpy as np
import wolff
import originalH
import pickle
import matplotlib.pyplot as plt


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
        Jlist,erro,_,_=np.linalg.lstsq(correlations,energy)
        return Jlist,erro

    def iterativeWolff(self,temperaturelist,K,mcsetN):
        """
        Generate spin sample chain with Wolff algorithm
        :return:
        """
        hightemperature=temperaturelist[0]

        Jlist,erro=self.fit_originalHsample(hightemperature,K,mcsetN)
        Jdata=np.zeros((len(temperaturelist),3))
        errodata=np.zeros(len(temperaturelist))
        Jdata[0]=Jlist
        errodata[0]=erro

        for (idx,temperature) in enumerate(temperaturelist[1:]):
            spinsample_chain=wolff.spinsample_chain(temperature,abs(Jlist[0]),self.Nx,self.Ny,mcsetN)
            energy,correlations=originalH.energyVsCorrelation(temperature,K,self.Nx,self.Ny,mcsetN,spinsample_chain)
            # energy,correlations=wolff.energyVsCorrelation(temperature,Jlist[0],self.Nx,self.Ny,mcsetN)
            Jlist,erro,_,_=np.linalg.lstsq(correlations,energy)
            Jdata[idx+1]=Jlist
            errodata[idx+1]=erro

        with open("data/Jdata2.txt","wb") as f:
            pickle.dump(Jdata,f)
        with open("data/errodata2.txt","wb") as f:
            pickle.dump(errodata,f)
        return Jlist

def loaddata(filename):
    with open("data/"+filename,"rb") as f:
        return pickle.load(f)

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
    temperaturelist=np.linspace(10,2.49,40)
    trainer=geteffectiveH(Nx,Ny)
    Jlist=trainer.iterativeWolff(temperaturelist,K,mcsetN)
    print(Jlist)

def get_trainingdata():
    Jlist=loaddata("Jdata2.txt")
    error=loaddata("errodata2.txt")
    print(Jlist[:,0])
    plt.figure()
    plt.plot(Jlist[:,0])
    plt.plot(Jlist[:,1]*20)
    plt.show()
    plt.figure()
    plt.plot(error)
    plt.show()



# train(0.2,mcsetN=100,Nx=10,Ny=10)
get_trainingdata()





