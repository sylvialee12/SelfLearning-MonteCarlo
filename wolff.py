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

class wolff:

    def __init__(self,temperature,Hamiltonian,init_sample,nsteps=1000):
        """

        :param temperature:
        :param Hamiltonian:
        :return:
        """
        self.Nx,self.Ny=Hamiltonian.Nx,Hamiltonian.Ny
        self.init_sample=init_sample
        self.temperature=temperature
        self.nsteps=nsteps
        self.pbc=Hamiltonian.pbc
        self.J=Hamiltonian.J
        pass

    def bond_neighbor(self,index_x,index_y,spin_sample):
        """
        Construct bond around site(index_x, index_y)
        :param index_x:
        :param index_y:
        :return:
        """
        factor=1-np.exp(-2.0/self.temperature*self.J)
        neighborsites=[((index_x+1)%self.Nx,index_y),
                       ((index_x+self.Nx-1)%self.Nx,index_y),
                       (index_x,(index_y+1)%self.Ny),
                       (index_x,(index_y-1+self.Ny)%self.Ny)]
        bond_neighbor=[(spin_sample[site]*spin_sample[index_x,index_y]+1)//2*(np.random.rand()<factor) for site in neighborsites]
        return neighborsites,bond_neighbor



    def single_cluster(self,index_x,index_y,spin_sample,cluster):
        """
        This is the first step for Wolff algorithm, starting from a random point, form cluster
        with its neighbor with the probability of p=(1-exp(-2 beta J))
        :param spin_sample: spin sample
        :return:
        """
        # go above to add neighbor into the cluster
        if (index_x,index_y) not in cluster:
            cluster.append((index_x,index_y))
            if self.pbc==1:

                neighborsites,bond_neighbor=self.bond_neighbor(index_x,index_y,spin_sample)
                effective_neighborsite=[neighborsites[x] for x in range(4) if neighborsites[x] not in cluster and bond_neighbor[x]==1]
                if effective_neighborsite==[]:
                    return None
                else:
                    for site in effective_neighborsite:
                        self.single_cluster(site[0],site[1],spin_sample,cluster)


    def flip_cluster(self,cluster,spin_sample):
        """
        :param cluster:
        :return:
        """
        spin_sample_new=spin_sample.copy()
        for site in cluster:
            spin_sample_new[site]=-1*spin_sample_new[site]

        return spin_sample_new


    def one_whole_step(self,spin_sample):
        """
        :return:
        """
        index_x=np.random.randint(0,self.Nx)
        index_y=np.random.randint(0,self.Ny)
        spin_sample_new=[]
        for spin_i in spin_sample:
            cluster=[]
            self.single_cluster(index_x,index_y,spin_i,cluster)
            spin_sample_new.append(self.flip_cluster(cluster,spin_i))
        return np.array(spin_sample_new)


    def markov_chain_sample(self,k=1):
        """
        Generate a spin sample chain every k step
        :param k:
        :return:
        """
        spin_sample_chain=[]
        spin_sample=self.init_sample.copy()
        for i in range(self.nsteps):
            spin_sample_chain.append(spin_sample)
            for j in range(k):
                spin_sample=self.one_whole_step(spin_sample)
        return spin_sample_chain


    def measure_energy(self,spin_sample_chain):
        """
        :return:
        """
        energy=[]
        for spin_sample in spin_sample_chain:
            upshift_sample=np.array(list(map(lambda x:np.concatenate((x[1:,:],self.pbc*np.expand_dims(x[0,:],axis=0)),axis=0),spin_sample)))
            leftshift_sample=np.array(list(map(lambda x:np.concatenate((x[:,1:],self.pbc*np.expand_dims(x[:,0],axis=1)),axis=1),spin_sample)))
            energy.append(np.mean(-self.J*(upshift_sample*spin_sample+leftshift_sample*spin_sample)))
        return energy

    def correlation_k(self,k,spin_sampleset):
        """
        Measure spin correlation between site(i) and site(i+k), which is <s_i s_i+k>
        :param spin_sample:
        :return:
        """
        left_shiftk_sampleset=np.array([np.concatenate((spin_sample[k:,:],self.pbc*spin_sample[0:k,:]),axis=0) for spin_sample in spin_sampleset])
        up_shiftk_sampleset=np.array([np.concatenate((spin_sample[:,k:],self.pbc*spin_sample[:,0:k]),axis=1) for spin_sample in spin_sampleset])
        correlation=np.mean(left_shiftk_sampleset*spin_sampleset+up_shiftk_sampleset*spin_sampleset)
        return correlation

    def correlation_2(self,spin_sampleset):
        """
        Measure spin correlation between the site i, j and its second nearest neighbor
        :param spin_sample:
        :return:
        """
        left_shift_sampleset=np.array([np.concatenate((spin_sample[1:,:],self.pbc*spin_sample[0:1,:]),axis=0) for spin_sample in spin_sampleset])
        up_leftshift_sampleset=np.array([np.concatenate((left_shift_sample[:,1:],self.pbc*left_shift_sample[:,0:1]),axis=1) for left_shift_sample in left_shift_sampleset])
        dn_leftshift_sampleset=np.array([np.concatenate((left_shift_sample[:,-1:],left_shift_sample[:,:-1]),axis=1) for left_shift_sample in left_shift_sampleset])
        correlation=np.mean(spin_sampleset*up_leftshift_sampleset+spin_sampleset*dn_leftshift_sampleset)
        return correlation


    def measure_magnetization(self,spin_sample_chain):
        """
        :param spin_sample_chain:
        :return:
        """
        magnetization=[]
        for spin_sample in spin_sample_chain:
            magnetization.append(np.mean(np.abs(np.mean(spin_sample,axis=(1,2)))))

        return magnetization

    def time_correlation(self,spin_sample_chain):
        """

        :param spin_sample_chain:
        :return:
        """
        Nt=len(spin_sample_chain)
        pass

def energyVsCorrelation(temperature,J,Nx,Ny,mcsetN):
    """

    :param temperature:
    :param Nx:
    :param Ny:
    :return:
    """
    Hamiltonian=Ising2D.Ising2D(nxspins=Nx,nyspins=Ny,J=J)
    spin_sampleset=(2*np.random.binomial(1,p=0.5,size=(mcsetN,10,10))-1)
    MC_Wolff=wolff(temperature,Hamiltonian=Hamiltonian,init_sample=spin_sampleset)
    data_count_init=100
    spin_chain=MC_Wolff.markov_chain_sample(k=Nx*Ny)
    energy=MC_Wolff.measure_energy(spin_chain[data_count_init:])
    correlation1=np.array([MC_Wolff.correlation_k(1,spin_sample) for spin_sample in spin_chain[data_count_init:]])
    correlation2=np.array([MC_Wolff.correlation_2(spin_sample) for spin_sample in spin_chain[data_count_init:]])
    correlation3=np.array([MC_Wolff.correlation_k(2,spin_sample) for spin_sample in spin_chain[data_count_init:]])
    correlations=np.array([correlation1,correlation2,correlation3])
    return energy,correlations.transpose()

def spinsample_chain(temperature,J,Nx,Ny,mcsetN):
    """

    :param temperature:
    :param Nx:
    :param Ny:
    :return:
    """
    Hamiltonian=Ising2D.Ising2D(nxspins=Nx,nyspins=Ny,J=J)
    spin_sampleset=(2*np.random.binomial(1,p=0.5,size=(mcsetN,10,10))-1)
    MC_Wolff=wolff(temperature,Hamiltonian=Hamiltonian,init_sample=spin_sampleset)
    data_count_init=100
    spin_chain=MC_Wolff.markov_chain_sample(k=Nx*Ny)
    return spin_chain



@functimer
def main():
    Hamiltonian=Ising2D.Ising2D(nxspins=10,nyspins=10,J=1,hfield=0)
    spin_sample=(2*np.random.binomial(1,p=0.5,size=(1,10,10))-1)
    MC_Ising2D=wolff(temperature=0.5,Hamiltonian=Hamiltonian,init_sample=spin_sample)
    spin_sample_chain=MC_Ising2D.markov_chain_sample()
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















