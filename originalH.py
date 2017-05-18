__author__ = 'xlibb'
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import stats


def functimer(func):

    def wrapper(*args,**kwargs):
        t0=time.time()
        result=func(*args,**kwargs)
        print("Running %s : %f s"%(func.__name__,time.time()-t0))
        return result
    return wrapper


class originalH:
    """
    This is a class describing the properties of the original Hamiltonian
    """
    def __init__(self,J,K,Nx,Ny,pbc=1):
        """
        Constructor for original Hamiltonian: H=-J\sum_{i,j} s_i*s_j-K\sum_{neighbors} s_j s_j s_k s_l
        :param K: Four body interaction term exchange energy
        :param Nx: Number of sites along x direction
        :param Ny: Number of sites along y direction
        :param pbc: Boundary condition, if periodic, pbc=1, open pbc=0
        :return:
        """
        self.J=J
        self.K=K
        self.Nx=Nx
        self.Ny=Ny
        self.pbc=pbc


    def measure_energy(self,spin_sampleset):
        """
        Measure energy for whole spin sample set
        :param spin_sampleset: classical spin configuration
        :return: energy get from single spin sample
        """
        left_shift1_sampleset=np.array([np.concatenate((spin_sample[1:,:],
                                                        self.pbc*np.expand_dims(spin_sample[0,:],axis=0)),axis=0)
                                        for spin_sample in spin_sampleset])
        up_shift1_sampleset=np.array([np.concatenate((spin_sample[:,1:],
                                                      self.pbc*np.expand_dims(spin_sample[:,0],axis=1)),axis=1)
                                      for spin_sample in spin_sampleset])
        upleft_shift_sampleset=np.array([np.concatenate((left_shift1_sample[:,1:],
                                                         self.pbc*np.expand_dims(left_shift1_sample[:,0],axis=1)),axis=1)
                                      for left_shift1_sample in left_shift1_sampleset])

        energy_nearest=-self.J*np.mean(left_shift1_sampleset*spin_sampleset+up_shift1_sampleset*spin_sampleset)
        energy_fourbody=-self.K*np.mean(left_shift1_sampleset*spin_sampleset*up_shift1_sampleset*upleft_shift_sampleset)

        return energy_nearest+energy_fourbody

    def measure_energy_singlesample(self,spin_sample):
        """

        :param spin_sample:
        :return:
        """
        left_shift1_sample=np.concatenate((spin_sample[1:,:],
                                                        self.pbc*np.expand_dims(spin_sample[0,:],axis=0)),axis=0)
        up_shift1_sample=np.concatenate((spin_sample[:,1:],
                                                      self.pbc*np.expand_dims(spin_sample[:,0],axis=1)),axis=1)
        upleft_shift_sample=np.concatenate((left_shift1_sample[:,1:],
                                                         self.pbc*np.expand_dims(left_shift1_sample[:,0],axis=1)),axis=1)

        energy_nearest=-self.J*np.mean(left_shift1_sample*spin_sample+up_shift1_sample*spin_sample)
        energy_fourbody=-self.K*np.mean(left_shift1_sample*spin_sample*up_shift1_sample*upleft_shift_sample)
        return energy_nearest+energy_fourbody

    def correlation_k(self,k,spin_sampleset):
        """
        Measure spin correlation between site(i) and site(i+k), which is <s_i s_i+k>
        :param spin_sample:
        :return:
        """
        left_shiftk_sampleset=np.array([np.concatenate((spin_sample[k:,:],self.pbc*spin_sample[0:k,:]),axis=0)
                                        for spin_sample in spin_sampleset])
        up_shiftk_sampleset=np.array([np.concatenate((spin_sample[:,k:],self.pbc*spin_sample[:,0:k]),axis=1)
                                      for spin_sample in spin_sampleset])
        correlation=np.mean(left_shiftk_sampleset*spin_sampleset+up_shiftk_sampleset*spin_sampleset)
        return correlation

    def correlation_2(self,spin_sampleset):
        """
        Measure spin correlation between the site i, j and its second nearest neighbor
        :param spin_sample:
        :return:
        """
        left_shift_sampleset=np.array([np.concatenate((spin_sample[1:,:],self.pbc*spin_sample[0:1,:]),axis=0)
                                       for spin_sample in spin_sampleset])
        up_leftshift_sampleset=np.array([np.concatenate((left_shift_sample[:,1:],self.pbc*left_shift_sample[:,0:1]),axis=1)
                                         for left_shift_sample in left_shift_sampleset])
        dn_leftshift_sampleset=np.array([np.concatenate((left_shift_sample[:,-1:],left_shift_sample[:,:-1]),axis=1)
                                         for left_shift_sample in left_shift_sampleset])
        correlation=np.mean(spin_sampleset*up_leftshift_sampleset+spin_sampleset*dn_leftshift_sampleset)
        return correlation


    def local_update_onestep(self,temperature,spin_sample):
        """
        :param temperature:
        :param spin_sample:
        :return:
        """
        new_spin_sample=spin_sample.copy()
        site_x,site_y=np.random.randint(self.Nx),np.random.randint(self.Ny)
        if self.pbc==1:
            neighbors=[((site_x+1)%self.Nx,site_y),
                       (site_x,(site_y+1)%self.Ny),
                       ((site_x-1+self.Nx)%self.Nx,site_y),
                       (site_x,(site_y-1+self.Ny)%self.Ny)]
            next_neighbors=[((site_x+1)%self.Nx,(site_y+1)%self.Ny),
                       ((site_x+1)%self.Nx,(site_y-1+self.Ny)%self.Ny),
                       ((site_x-1+self.Nx)%self.Nx,(site_y-1+self.Ny)%self.Ny),
                       ((site_x+1)%self.Nx,(site_y-1+self.Ny)%self.Ny)]

        else:
            pass

        delta_energy_1=2*self.J*sum([spin_sample[site_x,site_y]*spin_sample[site] for site in neighbors])
        delta_energy_2=2*self.K*sum([spin_sample[site_x,site_y]*spin_sample[neighbors[i]]*\
                                     spin_sample[neighbors[(i+1)%4]]*spin_sample[next_neighbors[i]] for i in range(4)])

        probability=np.exp(-1.0/temperature*(delta_energy_1+delta_energy_2))
        if probability>1:
            new_spin_sample[site_x,site_y]=-new_spin_sample[site_x,site_y]
        else:
            random=np.random.rand()
            if probability>random:
                new_spin_sample[site_x,site_y]=-new_spin_sample[site_x,site_y]
        return new_spin_sample

    @functimer
    def spin_chain_generation(self,temperature,init_sampleset,nstep=2000,k=1):
        """

        :return:
        """
        spin_chain=[]
        spin_sampleset=init_sampleset
        for i in range(nstep):
            spin_chain.append(spin_sampleset)
            for j in range(k):
                new_spin_sampleset=np.array([self.local_update_onestep(temperature,spin_sample)
                                             for spin_sample in spin_sampleset])
                spin_sampleset=new_spin_sampleset

        return spin_chain


def time_correlation(spin_chainset):
    """
    A method to visualize time correlation of the Markov Chain
    """
    # initsampleset=spin_chainset[0]
    # timecorrelation=[]
    # for sampleset in spin_chainset:
    #     timecorrelation.append(np.mean(initsampleset*sampleset)-np.mean(initsampleset)*np.mean(sampleset))
    # return timecorrelation

    mz = np.abs(np.mean(spin_chainset,axis=(2,3)))
    autocorrelation = [np.mean(np.sum(mz[:-j]*mz[j:],axis=0)/(len(mz)-j))
                       -np.mean(np.mean(mz,axis=0)**2) for j in range(1,100)]
    return autocorrelation


@functimer
def energyVsCorrelation(temperature,J,K,Nx,Ny,mcsetN,spin_chain=None):
    """

    :param temperature:
    :param K:
    :return:
    """
    if spin_chain==None:
        HMC=originalH(J=J,K=K,Nx=Nx,Ny=Ny,pbc=1)
        init_sampleset=(2*np.random.binomial(1,0.5,size=(mcsetN,Nx,Ny))-1)
        spin_chain=HMC.spin_chain_generation(temperature=temperature,init_sampleset=init_sampleset,nstep=1000,k=10)
    else:
        HMC=originalH(J=J,K=K,Nx=Nx,Ny=Ny,pbc=1)
    data_count_init=100
    energy=np.array([HMC.measure_energy(spin_sample) for spin_sample in spin_chain[data_count_init:]])
    correlation1=np.array([HMC.correlation_k(1,spin_sample) for spin_sample in spin_chain[data_count_init:]])
    correlation2=np.array([HMC.correlation_2(spin_sample) for spin_sample in spin_chain[data_count_init:]])
    correlation3=np.array([HMC.correlation_k(2,spin_sample) for spin_sample in spin_chain[data_count_init:]])
    correlations=np.array([correlation1,correlation2,correlation3])
    return energy,correlations.transpose()



if __name__=="__main__":
    energy,correlations=energyVsCorrelation(3,K=0.2,Nx=10,Ny=10,mcsetN=150)
    A=np.hstack([correlations[:,0:1],np.ones((len(correlations),1))])
    print(np.linalg.lstsq(A,energy))
    # print(stats.linregress(A,energy))
    plt.figure()
    plt.plot(energy)
    plt.show()
