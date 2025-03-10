# Electronic Hamiltonian

## Introduction

The (electronic) Hamiltonian is the core mathematical description of molecular systems. Application of first principles from physics allow for the construction of an energy function (the Hamiltonian) which determines all the properties of the system. In this post I want to give a brief introduction to the electronic Hamiltonian, the main approaches and some related algorithms. A molecular system consists of atoms and electrons. From quantum physics it is known that all matter is subject to the laws of quantum mechanics. However, with increasing scale, the laws of quantum mechanics converge to the laws of classical physics. Generally, speaking protons and neutrons (i.e. nucleons) are considered particles. They have a mass thousand times greater than that of an electron. The atomic properties are mostly determined by the electrons. Being fermions, electrons obey the Pauli exclusion principle which has some interesting consequences. Without going into detail, no two electrons can occupy the same quantum state. Due to the stron quantum nature of electrons, they can not be considered point particles, nor waves. Instead, the behavious of a quantum particle is described by a wave function, which is physically interpretable only as its quadratic absolute value. The quadratic absolute value of the wave function is the probability density of the particle. 

## Building the Energy function

It is clear that the nucleons and electrons of a molecule contribute to its energy. From elementary physics, the energy of a system is the sum of the kinetic and potential energy. Positive and negative charges attract each other, while equal charges repel each other, contributing to a potential energy. The respective forces are also called coulomb forces.They depend on the distance between the charges. Nucleons and electrons can be in a state of movement, contributing to kinetic energy. An important approximation we can make is that the kinetic energy of the nucleons is much smaller than that of the electrons. In fact, if we consider a time-scale sensible for the electrons, the nucleons can be considered stationary. This is called the Born-Oppenheimer approximation. Additionally, the mass of the nucleons is much greater than that of the electrons. + Let us write the energy terms of the electronic Hamiltonian: 

The total energy is 

$$ H = T + V $$ 

where T is the kinetic energy and V is the potential energy. The kinetic energy of the electrons is given by 

$$ T_e = - \sum_i \frac{h}{2 m_i} \nabla_{r_i}^2 $$ 

and, for the nucleons 

$$ T_n = - \sum_i \frac{h}{2 M_i} \nabla_{R_i}^2 $$ 

The potential energy due to electron-electron interactions is given by 

$$ V_{ee} = - \sum_i\sum_{j \neq i} \frac{e^2}{4 \pi \epsilon_0} \sum_{j \neq i} \frac{1}{|r_i - r_j|}. $$ 

The potential energy due to nucelus-nucelus interactionss is given by 

$$ V_{nn} = - \sum_i\sum_{j \neq i} \frac{Z_i Z_j e^2}{4 \pi \epsilon_0} \sum_{j \neq i} \frac{1}{|R_i - R_j|}. $$ 

The potential energy due to nucleon-electron interactionss is given by 

$$ V_{en} = - \sum_i\sum_{j \neq i} \frac{Z_i e^2}{4 \pi \epsilon_0} \sum_{j \neq i} \frac{1}{|R_i - r_j|}. $$ 

The total energy is written as \\(H = T_e + T_n + V_{ee} + V_{nn} + V_{en}\\). By the Born-Oppenheimer approximation, the kinetic energy of the nucleons is neglected so that \\(H = T_e + V_{ee} + V_{nn} + V_{en}\\). Thus, the electronic Hamiltonian is given by 

$$ H_e = - \sum_i \frac{h}{2 m_i} \nabla_{r_i}^2 - \sum_i\sum_{j \neq i} \frac{e^2}{4 \pi \epsilon_0} \sum_{j \neq i} \frac{1}{|r_i - r_j|} - \sum_i\sum_{j \neq i} \frac{Z_i Z_j e^2}{4 \pi \epsilon_0} \sum_{j \neq i} \frac{1}{|R_i - R_j|} - \sum_i\sum_{j \neq i} \frac{Z_i e^2}{4 \pi \epsilon_0} \sum_{j \neq i} \frac{1}{|R_i - r_j|}. $$ 

Note, that the nucleus positions are now fixed parameters and so are their distances and therefore also the arising coulomb forces. Considering the Schrödinger equation, the evolution of an (eigen-)quantum state is described by 

$$ H_e(r,R) \Phi(r,R) = E(r,R) \Phi(r,R) $$ 

where \\(\Phi\\) is the wave function and E is an eigenvalue associated with the Hamiltonian and eigen-state \Phi. The function \\(E(\cdot,R) \\) is called the potential energy surface. It depends only on \\(R\\). It is obtained by varying the nucleus positions and recomputing the eigenstate problem. To obtain the actual quantum evolution, the potential energy is added again and 

$$ [ T_n + E(R)] \Phi(R) = E\Phi(R) $$ 

is solved. \\(H = H_e + T_n \\) is called the molecular Hamiltonian. 

## Computational Complexities

Note, that the electronic Hamiltonian is a many-body problem. Already in classical phyiscs, many-body problems are hard to solve and generally subject to chaotic complexity. In quantum physics, the situation is even more complex due to quantum mechanical superposition, state collapse and entanglement. The electon-electron interactions require a computational complexity of \\(O(N_e^2)\\) where \\(N_e\\) is the number of electrons. The electron-nucleus interactions require a computational complexity of \\(O(N_e N_n)\\) where \\(N_n\\) is the number of nucleons. The nucleus-nucleus interactions require a computational complexity of \\(O(N_n^2)\\). The total computational complexity is \\(O(N_e^2 + N_e N_n + N_n^2)\\). This makes the electronic Hamiltonian a hard problem to solve. Given a set of nuclei and electrons it is hard to determine the actual state of the system from first principles (i.e. by energy minimization).   
  
The only nobel prize in computation was rewarded for the Kohn–Sham approximation which showed that the probabiliy density of the electrons can be approximated by function which depends only on the three position coordinates. This is a huge simplification from (\N + n \\) parameters to only three parameters. It paved the way for the widely used density functional theory (DFT). As the name already suggests, density functional theory ignores the wave function and directly computes the probability density.   
  
Another widely used method are Hartree-Fock methods which is based on variatonal methods. It considers a set of basis functions for the electron orbitals and parameterizes the wave function as a Slater determinant. This parameterization is anti-symmetric and fulfills the Pauli exclusion principle. By using variational methods, the energy is minimized and the wave function can obtained.   
  
A molecular system is not necessarily at its minimum-energy state. The presented methods, DFT and Hartree-Fock, both consider the ground state of the system. Extensions for excited states are possible but more complex. In this blog, I will attempt to describe density functional theory and the Hartree-Fock methods in more detail. I will present some algorithms and point towards exciting research which uses machine learning to solve the electronic Hamiltonian.   
  
This field is among the most exciting scientific endevours. It combines many disciplines, the new and weird properties of quantum-physics, computational challenges and information theory, bio-chemistry and material sciences. The electronic Hamiltonian is key to understanding the properties of molecules and materials and potentially open the doors to new medicines, new materials and a deeper understanding of life itself. 

