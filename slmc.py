__author__ = 'xlibb'

import GetEffective
import wolff
import originalH
import numpy as np
import pickle
import matplotlib.pyplot as plt
import Ising2D

class slmc(wolff.wolff):

    """
    This is the class implementing the self-learning Monte Carlo in Paper:Phys.Rev.B 95,041101.
    It inherits the wolff class in the wolff.py module, which implements the famous Wolff cluster
    update algorithm. The GetEffective.py provides the effective Hamiltonian. The originalH.py module
    implements the original Hamiltonian with four body interaction.

    """
    def __init__(self,temperature,Hamiltonian,init_sample,Hreal):
        """
        The constructor of slmc. It inherits the wolff class.
        :param temperature: critical temperature
        :param Hamiltonian: 2D Ising model object, it contains the information of Jeff, system size
        :param init_sample: initial sampling set to generate Markov Chain set
        :param Hreal: the real Hamiltonian object
        """
        super().__init__(temperature,Hamiltonian,init_sample,nsteps=2000)
        self.Hreal = Hreal

    def real_flip_cluster(self,cluster,spin_sample):
        """
        Find the energy difference before and after flipping the whole cluster
        :param cluster: cluster formed by Wolff algorithm
        :return:
        """
        nearest_neighbor = []
        spin_sample_new=self.flip_cluster(cluster,spin_sample)
        for site in cluster:
            neighbor,neighborbond = self.bond_neighbor(site[0],site[1],spin_sample)
            nearest_neighbor += neighbor
        effective_neighbor = list(filter(lambda x:x not in cluster,nearest_neighbor))
        delta_energy_effetive = 2*self.J*sum([spin_sample[cluster[0]]*spin_sample[site]
                                            for site in effective_neighbor])
        energy0=self.Hreal.measure_energy_singlesample(spin_sample)*self.Nx*self.Ny
        energy1=self.Hreal.measure_energy_singlesample(spin_sample_new)*self.Nx*self.Ny
        rn=np.random.rand()
        if np.exp(-1/self.temperature*((energy1-energy0)-delta_energy_effetive))>rn:
            return spin_sample_new
        else:
            return spin_sample

    def one_whole_step(self,spin_sample):
        """
        A full Markov chain step
        :param spin_sample:
        :return:new spin sample
        """
        spin_sample_new = []
        for spin_i in spin_sample:
            index_x = np.random.randint(0,self.Nx)
            index_y = np.random.randint(0,self.Ny)
            cluster = []
            self.single_cluster(index_x,index_y,spin_i,cluster)
            spin_sample_new.append(self.real_flip_cluster(cluster,spin_i))
        return np.array(spin_sample_new)



def main(J,K,Nx,Ny,temperature,mcsetN,Jeff=None):
    """

    :return:
    """
    Hreal=originalH.originalH(J,K,Nx,Ny)
    if Jeff==None:
        Jlist,effetiveH=GetEffective.train(J,K,Nx,Ny,temperature+3,temperature,mcsetN=mcsetN)
        print(effetiveH.J)
    else:
        effetiveH=Ising2D.Ising2D(nxspins=Nx,nyspins=Ny,J=Jeff)
        print(effetiveH.J)
    initsampleset=2*np.random.binomial(1,p=0.5,size=(mcsetN,Nx,Ny))-1
    simulator=slmc(temperature,effetiveH,initsampleset,Hreal)
    spin_chain1=simulator.markov_chain_sample(k=1)
    spin_chain2=Hreal.spin_chain_generation(temperature,initsampleset,k=Nx*Ny)
    energy1_list=np.array([Hreal.measure_energy(spin_sampleset) for spin_sampleset in spin_chain1])
    energy2_list=np.array([Hreal.measure_energy(spin_sampleset) for spin_sampleset in spin_chain2])
    tau_correlation1=originalH.time_correlation(spin_chain1)
    tau_correlation2=originalH.time_correlation(spin_chain2)

    with open("data/energy1.txt",'wb') as f:
        pickle.dump(energy1_list,f)
    with open("data/energy2.txt","wb") as f:
        pickle.dump(energy2_list,f)
    with open("data/correlation1.txt","wb") as f:
        pickle.dump(tau_correlation1,f)
    with open("data/correlation2.txt","wb") as f:
        pickle.dump(tau_correlation2,f)


def loaddata(filename):
    with open(filename,"rb") as f:
        return pickle.load(f)

def visulizedata():
    energy1=loaddata("data/energy1.txt")
    energy2=loaddata("data/energy2.txt")

    tau_correlation1=loaddata("data/correlation1.txt")
    tau_correlation2=loaddata("data/correlation2.txt")
    plt.figure("Energy")
    plt.plot(energy1[0:100],"b--",label="SLMC")
    plt.plot(energy2[0:100],"g-",label="Local update")
    plt.legend(loc='upper right', shadow=True)
    plt.savefig("data/energy2.png")
    plt.figure("AutoCorrelation")
    plt.plot(tau_correlation1[0:100],"b--",label="SLMC")
    plt.plot(tau_correlation2[0:100],"g-",label="Local update")
    plt.legend(loc='upper right', shadow=True)
    plt.savefig("data/Autocorrealation2.png")
    plt.show()

main(J=1,
     K=0.2,
     Nx=10,
     Ny=10,
     temperature=2.493,
     mcsetN=100,
     Jeff=1.12
     )

visulizedata()