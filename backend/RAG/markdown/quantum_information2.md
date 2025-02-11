# Quantum Information Summary with Explanations

## State Representation

In quantum mechanics, the state of a system is represented by a vector in a complex Hilbert space. This is commonly denoted as \\( |\Psi\rangle \\) (a "ket" in Dirac notation).

\\[ |\Psi\rangle = \sum_{k=0}^{d-1} \psi_k |k\rangle \\]

Here: 

  * \\( \\{ |k\rangle \\} \\): An orthonormal basis of the Hilbert space.
  * \\( \psi_k \\): Complex coefficients representing the probability amplitudes of each basis state.



Similarly, another state \\( |\Phi\rangle \\) is given by:

\\[ |\Phi\rangle = \sum_{k=0}^{d-1} \phi_k |k\rangle \\]

The overlap (inner product) between these states determines how "similar" they are:

\\[ P_\Phi = |\langle \Phi | \Psi \rangle|^2 \\]

\\( P_\Phi \\) represents the probability of finding the system in state \\( |\Phi\rangle \\) if it is currently in state \\( |\Psi\rangle \\).

## Projectors and Operators

A projector is an operator that "projects" a vector onto a subspace. For \\( |\Psi\rangle \\), the projector is:

\\[ \Pi = |\Psi\rangle \langle \Psi| \\]

Projectors satisfy \\( \Pi^2 = \Pi \\), which means they are idempotent. They are used to describe measurements that correspond to specific outcomes.

A general linear operator \\( \hat{O} \\) that transforms \\( |\Psi\rangle \\) into \\( |\Phi\rangle \\) is defined as:

\\[ \hat{O} = |\Phi\rangle \langle \Psi| \\]

## Measurement & Observables

In quantum mechanics, observables correspond to Hermitian operators. Their eigenvalues represent possible measurement outcomes, and their eigenvectors represent the states associated with those outcomes.

The spectral decomposition of an observable \\( \hat{O} \\) is:

\\[ \hat{O} = \sum_{k=0}^{d-1} \lambda_k |k\rangle \langle k| \\]

Here: 

  * \\( \lambda_k \\): The eigenvalues (measurement outcomes).
  * \\( |k\rangle \\): The eigenvectors (states corresponding to those outcomes).



If a system is in state \\( |i\rangle \\), applying \\( \hat{O} \\) gives:

\\[ \hat{O} |i\rangle = \lambda_i |i\rangle \\]

The expectation value \\( \langle \hat{O} \rangle \\) represents the average value of measurements of \\( \hat{O} \\) and is computed as:

\\[ \langle \hat{O} \rangle = \sum_{k=0}^{d-1} \lambda_k P_{k,\Phi} \\]

or equivalently:

\\[ \langle \hat{O} \rangle = \langle \Psi | \hat{O} | \Psi \rangle \\]

## Observables for Qubits

Qubits are the fundamental units of quantum information. Common observables for qubits include the Pauli matrices:

\\[ \sigma_x = \begin{pmatrix} 0 & 1 \\\ 1 & 0 \end{pmatrix}, \quad \sigma_y = \begin{pmatrix} 0 & -i \\\ i & 0 \end{pmatrix}, \quad \sigma_z = \begin{pmatrix} 1 & 0 \\\ 0 & -1 \end{pmatrix} \\]

These matrices represent measurements along the \\( x \\)-, \\( y \\)-, and \\( z \\)-axes, respectively.

The eigenvalues and eigenvectors of these matrices are:

\\[ \sigma_x: \\{1, |0\rangle; -1, |1\rangle\\}, \quad \sigma_y: \\{1, |i\rangle; -1, |-i\rangle\\}, \quad \sigma_z: \\{1, |+\rangle; -1, |-\rangle\\} \\]

These operators are mutually exclusive, meaning you cannot measure along more than one axis simultaneously. For example, to measure \\( \\{|0\rangle, |1\rangle\\} \\), use \\( \sigma_z \\).

## Pure, Mixed, and Separable States

A quantum state can be pure, mixed, or separable. Here's how these terms are defined:

### Pure States

A pure state is described by a single vector \\( |\Psi\rangle \\) in the Hilbert space. It cannot be written as a mixture of other states. A pure state satisfies:

\\[ \rho = |\Psi\rangle \langle \Psi|, \quad \rho^2 = \rho \\]

### Mixed States

A mixed state represents a statistical ensemble of pure states. It is described by a density matrix \\( \rho \\) that satisfies:

\\[ \rho = \sum_k p_k |\psi_k\rangle \langle \psi_k|, \quad \text{where } \sum_k p_k = 1 \\]

The purity of a state is determined by \\( \text{Tr}(\rho^2) \\):

  * \\( \text{Tr}(\rho^2) = 1 \\): Pure state.
  * \\( \text{Tr}(\rho^2) < 1 \\): Mixed state.



### Separable States

A state is separable if it can be written as a tensor product of states from subsystems. For example, a separable state for two systems \\( A \\) and \\( B \\) is:

\\[ \rho_{A,B} = \sum_x p_x \rho_{A,x} \otimes \rho_{B,x} \\]

When this condition is not met, the state is entangled.

## Entangled States

An entangled state cannot be separated into independent subsystems. For example, the Bell states are maximally entangled states:

\\[ |\Phi^+\rangle = \frac{1}{\sqrt{2}} (|00\rangle + |11\rangle), \quad |\Psi^-\rangle = \frac{1}{\sqrt{2}} (|01\rangle - |10\rangle) \\]

Entanglement is a fundamental feature of quantum mechanics and enables phenomena such as quantum teleportation and superdense coding.

## Hermitian Operators and Mixed States

Hermitian operators are used to describe observables and density matrices in quantum mechanics. A general Hermitian operator can be written as:

\\[ \hat{H} = \frac{1}{2} (r_0 \hat{1} + \vec{r} \cdot \vec{\sigma}) \\]

Here: 

  * \\( r_0 \\): Scalar term.
  * \\( \vec{r} \\): Bloch vector representing the state.
  * \\( \vec{\sigma} \\): Vector of Pauli matrices.



The length of \\( \vec{r} \\) determines whether the state is pure (\\( r = 1 \\)) or mixed (\\( r < 1 \\)).

## Werner State

The Werner state is a mixture of a pure entangled state and the maximally mixed state:

\\[ \rho_{AB} = (1-p)|\Psi\rangle \langle \Psi| + p \frac{\hat{I}}{4} \\]

The Werner state becomes separable when \\( p \geq \frac{2}{3} \\).

## Quantum Measurements

Measurements in quantum mechanics are described using operators. The probability of obtaining a particular measurement outcome is given by:

\\[ P_k = \text{Tr}(\hat{M}_k^\dagger \hat{M}_k \rho) \\]

After the measurement, the state collapses to:

\\[ \rho \to \frac{\hat{M}_k \rho \hat{M}_k^\dagger}{P_k} \\]

A special class of measurements, called Positive Operator-Valued Measures (POVMs), generalizes the concept of projective measurements:

\\[ F_k = \hat{M}_k^\dagger \hat{M}_k \\]

## Quantum Information & Entropy

### Shannon Entropy

Shannon entropy measures the uncertainty of a probability distribution. For a message with symbols \\( k \\) occurring with probabilities \\( p_k \\):

\\[ H = -\sum_{k} p_k \log_2(p_k) \\]

This represents the minimum number of bits required to encode the message on average.

### Von Neumann Entropy

Von Neumann entropy extends Shannon entropy to quantum states. For a density matrix \\( \rho \\):

\\[ S(\rho) = -\text{Tr}(\rho \log \rho) \\]

If \\( \rho \\) is diagonal in the eigenbasis with eigenvalues \\( \lambda_j \\):

\\[ S(\rho) = -\sum_j \lambda_j \log \lambda_j \\]

Properties of Von Neumann entropy:

  * \\( S(\rho) = 0 \\) for pure states.
  * \\( S(\rho) > 0 \\) for mixed states.
  * \\( S(\rho) = \log_2(N) \\) for maximally mixed states with dimension \\( N \\).
  * It satisfies the subadditivity property: \\( S(\rho_{AB}) \leq S(\rho_A) + S(\rho_B) \\).



## Quantum Fourier Transform

The Quantum Fourier Transform (QFT) is a quantum algorithm that maps a quantum state \\( |x\rangle \\) to its Fourier-transformed version:

\\[ \text{QFT}|x\rangle = \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} e^{2\pi i \frac{xk}{N}} |k\rangle \\]

Applications of QFT include solving problems like period finding and phase estimation. It can be implemented using Hadamard and controlled phase-shift gates.

## Shor's Algorithm

### Classical Reduction

Shor's algorithm is used for integer factorization. It combines classical and quantum steps:

  1. Choose \\( a \\) such that \\( 1 < a < N \\).
  2. Compute \\( \text{gcd}(a, N) \\). If \\( \text{gcd}(a, N) > 1 \\), you have found a factor of \\( N \\).
  3. Define \\( f(x) = a^x \mod N \\). Find the period \\( r \\) of \\( f(x) \\) using a quantum computer.
  4. If \\( r \\) is even, compute \\( \text{gcd}(a^{r/2} - 1, N) \\) and \\( \text{gcd}(a^{r/2} + 1, N) \\) to get factors of \\( N \\).



### Quantum Period Finding

The quantum step of Shor's algorithm uses the QFT to find the period \\( r \\) of \\( f(x) \\). By preparing a superposition and applying the QFT, the period can be extracted efficiently.

## Grover's Algorithm

Grover's algorithm is a quantum search algorithm that finds a marked item in an unsorted database of \\( N \\) items using \\( O(\sqrt{N}) \\) steps. The key operations are:

  * An oracle \\( \hat{O} \\) that flips the phase of the marked state.
  * The Grover operator \\( \hat{G} = 2|\Psi\rangle\langle \Psi| - \hat{I} \\), which amplifies the amplitude of the marked state.



Starting with a superposition of all states, repeated applications of \\( \hat{O} \\) and \\( \hat{G} \\) converge to the marked state.

## Phase Estimation

The phase estimation algorithm is a cornerstone of quantum computing, used to estimate the eigenphase \\( \Phi \\) of a unitary operator \\( \hat{U} \\). If:

\\[ \hat{U}|u\rangle = e^{2\pi i \Phi}|u\rangle, \\]

then \\( \Phi \\) can be determined as follows:

  1. Prepare an input superposition state using Hadamard gates:

\\[ |x\rangle = \frac{1}{\sqrt{N}}\sum_{x=0}^{N-1}|x\rangle. \\]

  2. Apply controlled applications of \\( \hat{U} \\) to encode the phase information.
  3. Perform the Quantum Fourier Transform to extract \\( \Phi \\) from the encoded state.



The output state is a superposition, and measuring it provides information about \\( \Phi \\).

## Quantum Teleportation

Quantum teleportation allows the transfer of a quantum state \\( |\Psi\rangle = \alpha|0\rangle + \beta|1\rangle \\) from one location to another using entanglement and classical communication. The process involves:

  1. Entangling two qubits \\( B \\) and \\( C \\), shared between the sender and receiver:

\\[ \frac{1}{\sqrt{2}}(|0\rangle_B|1\rangle_C - |1\rangle_B|0\rangle_C). \\]

  2. Combining the state to be teleported with the entangled pair:

\\[ (\alpha|0\rangle_A + \beta|1\rangle_A) \otimes \frac{1}{\sqrt{2}}(|0\rangle_B|1\rangle_C - |1\rangle_B|0\rangle_C). \\]

  3. Measuring the combined system in the Bell basis and sending the classical result to the receiver.
  4. The receiver applies a unitary transformation based on the classical result to recover \\( |\Psi\rangle \\).



## Krauss Operators and Non-Isolated Systems

Krauss operators describe the evolution of quantum states in open systems. For a system \\( A \\) interacting with an environment \\( E \\), the state is described by:

\\[ \rho_A = \text{Tr}_E[|\Phi\rangle\langle\Phi|_{AE}]. \\]

Under evolution, the state of \\( A \\) changes as:

\\[ \rho_A \to \sum_k \hat{E}_k \rho_A \hat{E}_k^\dagger, \\]

where \\( \hat{E}_k \\) are Krauss operators satisfying:

\\[ \sum_k \hat{E}_k^\dagger \hat{E}_k = \hat{I}. \\]

This ensures the evolution is trace-preserving and physically valid.

## Generalized Measurement and POVMs

Generalized measurements go beyond projective measurements and use Positive Operator-Valued Measures (POVMs) to describe outcomes:

\\[ F_k = \hat{M}_k^\dagger \hat{M}_k, \\]

where \\( \hat{M}_k \\) are the measurement operators. The probability of outcome \\( k \\) is:

\\[ P_k = \text{Tr}(\hat{M}_k^\dagger \hat{M}_k \rho). \\]

The post-measurement state is:

\\[ \rho \to \frac{\hat{M}_k \rho \hat{M}_k^\dagger}{P_k}. \\]

## Applications

  * **Shor's Algorithm:** Efficient factorization of integers, breaking RSA encryption.
  * **Grover's Algorithm:** Fast quantum search in unsorted databases.
  * **Quantum Teleportation:** Secure state transfer over distances.
  * **Phase Estimation:** Used in algorithms like quantum simulation and factoring.



