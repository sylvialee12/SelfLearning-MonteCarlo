# MonteCarloIntro

This is a introduction project to simulate spin models(2D Ising model, four body interacting Ising model) with self learning Monte Carlo method. 

For the details of the method, please refer to the paper: Phys.Rev.B 90, 041101(R)

1. swendsenwang.py is the implementation of Swendsen Wang algorithm for 2D spin model. Here we provide it as a reference. The self learning Monte Carlo Method does not include it.

2. Wolff.py module is the implementation of Wolff cluster update algorithm for 2D spin model. We apply this update for proposing global update in the effectiveHamiltonian.

3. Ising2D.py module provides the constructor of 2D square lattice.

4. GetEffectiveH.py is the file constructing effective Hamiltonian for given original Hamiltonian  

5. originalH.py module provides the constructor of original Hamiltonian with four body interaction. It also concludes a simple local update Monte Carlo method.

6. slmc.py is the main module of the whole project. To use it, one may refer to the file itself.

The data folder is the basic results for our testing model(J=1,K=0.2)

