__author__ = 'xlibb'

import numpy as np
import Ising2D
import matplotlib.pyplot as plt
import time

def functimer(func):
    def wrapper(*args,**kwargs):
        t0=time.time()
        result=func(*args,**kwargs)
        print("Running %s : %f s"%(func.__name__,time.time()-t0))
        return result
    return wrapper

class swendsenwang:
    """
    This is a demo for Swendsen Wang algorithm for 2D Ising model.
    """
    def __init__(self,Hamiltonian,nsteps=1000):
        """
        :return:
        """
        self.pbc=Hamiltonian.pbc
        self.J=Hamiltonian.J
        self.nsteps=nsteps


    def bond_given_config(self,temperature,spin_sample):
        """

        :param Ising:
        :return:
        """
        factor=(1-np.exp(-2.0/temperature*self.J))
        if self.pbc==1:
            upshift_sample=np.array(list(map(lambda x:np.concatenate((x[1:,:],np.expand_dims(x[0,:],axis=0)),axis=0),spin_sample)))
            leftshift_sample=np.array(list(map(lambda x:np.concatenate((x[:,1:],np.expand_dims(x[:,0],axis=1)),axis=1),spin_sample)))
            bond_vertical=((upshift_sample*spin_sample+1)//2)*(np.random.binomial(1,p=factor,size=spin_sample.shape))
            bond_horizontal=((leftshift_sample*spin_sample+1)//2)*(np.random.binomial(1,p=factor,size=spin_sample.shape))
        else:
            pass
        return bond_vertical,bond_horizontal


    def cluster_from_bond(self,bond_vertical,bond_horizontal):
        """
        Form cluster from the bond matrices.
        :param bond_vertical:vertical bond matrix, (i,j) meaning the bond connecting (i,j) and (i+1,j).
        :param bond_horizontal:
        :return:
        """
        labels = np.arange(np.prod(bond_vertical.shape)).reshape(bond_vertical.shape)
        while True:
            labelsnew = self.proper_label(labels,bond_vertical,bond_horizontal)
            if np.array_equal(labelsnew,labels):
                break
            else:
                labels = labelsnew
        cluster_set = np.unique(labels)
        cluster_dict = {}
        for ith in cluster_set:
            cluster_dict[ith] = np.argwhere(labels==ith)
        return cluster_dict


    def proper_label(self,labels,bond_vertical,bond_horizontal):
        """
        Quick-Union algorithm implementation
        A recursive method to find a cluster, start from sitex, sitey and go through their neighbors.
        Then go through the neighbors of the neighbors.
        :param labels: labels matrix for every site.
        :param bond_vertical: vertical bond matrix, (i,j) meaning the bond connecting (i,j) and (i+1,j).
        :param bond_horizontal: horizontal bond matrix, (i,j) meaning the bond connecting (i,j) and (i,j+1).
        :param cluster_ith: the ith cluster to label
        :return: updated labels
        """
        labelsnew=labels.copy()
        Nx,Ny=bond_vertical.shape
        if self.pbc==1:
            for i in range(Nx):
                for j in range(Ny):
                    if bond_vertical[i,j]==1:
                        labelsnew[(i+1)%Nx,j],labelsnew[i,j]=min(labelsnew[(i+1)%Nx,j],labelsnew[i,j]),\
                                                             min(labelsnew[(i+1)%Nx,j],labelsnew[i,j])
                    if bond_horizontal[i,j]==1:
                        labelsnew[i,(j+1)%Ny],labelsnew[i,j]=min(labelsnew[i,(j+1)%Ny],labelsnew[i,j]),\
                                                             min(labelsnew[i,(j+1)%Ny],labelsnew[i,j])
        else:
            pass
        return labelsnew


    def flip_cluster(self,cluster_dict,spin_config):
        """
        Assign all the spins in a cluster randomly to be \pm 1
        :param bond_sample:
        :return:
        """
        spin_confignew=spin_config.copy()
        for key,value in cluster_dict.items():
            a=1 if np.random.rand()>0.5 else -1
            for value_i in value:
                spin_confignew[value_i]=a

        return spin_confignew



    def markov_chain_onestep(self,temperature,spin_sample):
        """
        :param spin_sample:
        :return:
        """
        bond_vertical,bond_horizontal=self.bond_given_config(temperature,spin_sample)
        cluster_dict_sample=[self.cluster_from_bond(bond_vertical_i,bond_horizontal_i)
                             for bond_vertical_i,bond_horizontal_i in zip(bond_vertical,bond_horizontal)]
        spinnew_sample=np.array([self.flip_cluster(cluster_dict_i,spin_sample_i)
                        for cluster_dict_i,spin_sample_i in zip(cluster_dict_sample,spin_sample)])
        return spinnew_sample


    def annealing(self,hightemp,lowtemp,spin_sample,k=100):
        """
        Annealing algorithm with Gibbs sampling
        :param hightemp: high temperature
        :param lowtemp: low temperature
        :param spin_sample: initial sampling
        :return:
        """
        spin_sample_chain = [spin_sample]
        temperaturelist = np.linspace(hightemp,lowtemp,self.nsteps)
        for temperature in temperaturelist:
            for i in range(k):
                spin_sample_chain.append(self.markov_chain_onestep(temperature,spin_sample_chain[-1]))
        return spin_sample_chain


    def gibbs_sampling(self,temperature,spin_sample):
        """
        Gibbs sampling from spin_sample to final spin_sample
        :param sample:
        :param kstep:
        :return:
        """
        spin_sample_chain=[spin_sample]
        for i in range(self.nsteps):
            # if i%100==0:
            #     print(i)
            spin_sample_chain.append(self.markov_chain_onestep(temperature,spin_sample_chain[i]))
        return np.array(spin_sample_chain)



    def measure_energy(self,Hamiltonian,spin_sample_chain):
        """
        :return:
        """
        energy = []
        energy_error = []
        for spin_sample in spin_sample_chain:
            upshift_sample = np.array(list(map(lambda x:np.concatenate((x[1:,:],Hamiltonian.J*np.expand_dims(x[0,:],axis=0)),axis=0),spin_sample)))
            leftshift_sample = np.array(list(map(lambda x:np.concatenate((x[:,1:],Hamiltonian.J*np.expand_dims(x[:,0],axis=1)),axis=1),spin_sample)))
            energy.append(np.mean(-Hamiltonian.J*(upshift_sample*spin_sample+leftshift_sample*spin_sample)))
            energy_error.append(np.std(np.mean(-Hamiltonian.J*(upshift_sample*spin_sample+leftshift_sample*spin_sample),axis=(1,2))))

        return energy,energy_error


    def measure_magnetization(self,spin_sample_chain):
        """
        :param spin_sample_chain:
        :return:
        """
        magnetization=[]
        for spin_sample in spin_sample_chain:
            magnetization.append(np.mean(np.abs(np.mean(spin_sample,axis=(1,2)))))

        return magnetization

    def time_correlation(self):
        pass

@functimer
def main():
    Hamiltonian=Ising2D.Ising2D(nxspins=5,nyspins=5,J=1,hfield=0)
    MC_Ising2D=swendsenwang(Hamiltonian,nsteps=1000)
    spin_sample=(2*np.random.binomial(1,p=0.5,size=(50,5,5))-1)
    temperature = 2.26
    spin_sample_chain=MC_Ising2D.gibbs_sampling(temperature,spin_sample)

    # hightemp,lowtemp=5,3
    # spin_sample_chain=MC_Ising2D.annealing(hightemp,lowtemp,spin_sample,k=100)
    energy,energy_erro=MC_Ising2D.measure_energy(Hamiltonian,spin_sample_chain)
    magnetization=MC_Ising2D.measure_magnetization(spin_sample_chain)
    plt.figure("Energy")
    plt.plot(energy)
    plt.show()
    plt.figure("Magnetization")
    plt.plot(magnetization)
    plt.show()

if __name__=="__main__":
    main()





