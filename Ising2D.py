__author__ = 'xlibb'

class Ising2D:
    """
    Ising 2D system, giving the systems size, transverse field strength, periodic boundary condition
    """
    def __init__(self,nxspins,nyspins,J,hfield=0,pbc=1):
        """

        """
        self.Nx=nxspins
        self.Ny=nyspins
        self.J=J
        self.hfield=hfield
        self.pbc=pbc

    def findcon(self,config):
        """
        Finding the nonzero configuration for given configuration,
        which satisfies <s'|H|s> is nonzero
        """
        pass