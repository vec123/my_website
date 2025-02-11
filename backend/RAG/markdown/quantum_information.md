# Quantum Information Summary

## State Representation

\\[ |\Psi\rangle = \sum_{k=0}^{d-1} \psi_k |k\rangle \\]

\\[ |\Phi\rangle = \sum_{k=0}^{d-1} \phi_k |k\rangle \\]

\\[ P_\Phi = |\langle \Phi | \Psi \rangle|^2 \\]

\\[ \Pi = |\Psi\rangle \langle \Psi| \quad (\Pi^2 = \Pi) \\]

\\[ \hat{O} = |\Phi\rangle \langle \Psi| \\]

## Measurement & Observables

\\[ \hat{O} = \sum_{k=0}^{d-1} \lambda_k |k\rangle \langle k| \\]

\\[ \hat{O} |i\rangle = \sum_{k=0}^{d-1} \lambda_k |k\rangle \langle k| |i\rangle = \lambda_k \langle k | i \rangle |i\rangle = \lambda_{O,i} |i\rangle \\]

\\[ \langle \hat{O} \rangle = \sum_{k=0}^{d-1} \lambda_k P_{k,\Phi} = \langle \Psi | \lambda | \Psi \rangle \\]

## Observables for Qubits

\\[ \sigma_x = \begin{pmatrix} 0 & 1 \\\ 1 & 0 \end{pmatrix}, \quad \sigma_y = \begin{pmatrix} 0 & -i \\\ i & 0 \end{pmatrix}, \quad \sigma_z = \begin{pmatrix} 1 & 0 \\\ 0 & -1 \end{pmatrix} \\]

Eigenvalues and eigenvectors:

\\[ \sigma_x: \\{1, |0\rangle; -1, |1\rangle\\}, \quad \sigma_y: \\{1, |i\rangle; -1, |-i\rangle\\}, \quad \sigma_z: \\{1, |+\rangle; -1, |-\rangle\\} \\]

The three operators are mutually exclusive. To measure \\( \\{|0\rangle, |1\rangle\\} \\), use \\( \sigma_x \\); the others return random outputs.

## Pure, Mixed & Separable States

For pure states:

\\[ |\Psi\rangle \text{ is pure (can be mixed, can be entangled)}. \\]

\\[ \langle \hat{O} \rangle = \sum_{k=1}^{d-1} p_k \langle \psi_k|\hat{O}|\psi_k\rangle = \text{Tr}\big[\psi_k |\Psi\rangle \langle \Psi| \hat{O}\big] \\]

Trace properties:

\\[ \text{Tr}(\hat{O}) = \sum_{k=0}^{d-1} \langle k|\hat{O}|k\rangle, \quad \hat{O} = \sum_k p_k (|\psi_k\rangle \langle \psi_k|) \\]

Mixed state ensemble:

\\[ \rho = \sum_k p_k (|\psi_k\rangle \langle \psi_k|) \\]

Probability:

\\[ P_{sk} = \langle \Psi|\rho|\Psi\rangle \\]

Pure state criteria:

\\[ \rho^2 = \rho \quad (\text{pure}), \quad \text{Tr}(\rho^2) < 1 \quad (\text{mixed}) \\]

\\[ \rho = \sum_k p_k |\psi_k\rangle \langle \psi_k| \quad (\text{mixed}), \quad \text{Tr}(\rho) = 1 \\]

Entangled states:

\\[ \rho_{A,B} \neq \sum_{x=0}^N p_x \rho_{A,x} \otimes \rho_{B,x} \\]

This indicates the state is entangled (not separable). Note that:

  * An entangled state can be mixed or pure.
  * A mixed state can be separable or entangled.
  * A pure state can be entangled but cannot be mixed.



## Generic Hermitian Operators

\\[ \hat{H} = \frac{1}{2}(r_0 \hat{1} + \vec{r} \cdot \vec{\sigma}) \\]

Properties:

  * \\( r = 1 \\): pure state
  * \\( r < 1 \\): mixed state



## Mixed States as Convex Combinations

\\[ \rho = \sum_{x=0}^N p_x \rho_x, \quad \sum_{x=0}^N p_x = 1 \\]

## Multiple Quantum Systems and Non-Classical Correlations

For separable states:

\\[ |\Psi\rangle_{AB} = \sum_{j=0} \psi_{j,k} |j\rangle_A \otimes |k\rangle_B \\]

For entangled states:

\\[ |\Psi\rangle_{AB} \neq \sum_{j=0} \psi_{j,k} |j\rangle_A \otimes |k\rangle_B \\]

Examples (Bell states):

\\[ |\Phi^+\rangle_{AB} = \frac{1}{\sqrt{2}}(|01\rangle_{AB} + |10\rangle_{AB}) \\]

\\[ |\Psi^-\rangle_{AB} = \frac{1}{\sqrt{2}}(|00\rangle_{AB} - |11\rangle_{AB}) \\]

## Instantaneous Collapse

\\[ \rho = \frac{1}{2}|0\rangle\langle0| + \frac{1}{2}|1\rangle\langle1| \\]

Measurement in system \\( A \\) results in instantaneous collapse in system \\( B \\):

\\[ |0\rangle \to |0\rangle_A |0\rangle_B, \quad |1\rangle \to |1\rangle_A |0\rangle_B \\]

For \\( |\Psi^-\rangle \\):

\\[ \rho_B = \text{Tr}_A[|\Psi\rangle\langle\Psi|] = \frac{1}{2}\hat{I}_B \\]

This indicates a completely mixed state in \\( B \\).

## Entanglement for Mixed States

Mixed separable state:

\\[ \rho_{AB} = \sum_{n=1}^N p_n \rho_A^n \otimes \rho_B^n \\]

Mixed entangled state:

\\[ \rho_{AB} \neq \sum_{n=1}^N p_n \sigma_A^n \otimes \sigma_B^n \\]

Partial trace for separability:

\\[ \rho_{AB}^{T_A} = \sum_{n=1}^N p_n \rho_A^{nT} \otimes \rho_B^n \geq 0 \\]

If \\( \text{Tr}(\rho_{AB}^{T_A}) < 0 \\), the state is entangled.

## Werner State

\\[ \rho_{AB} = (1-p)|\Psi\rangle\langle\Psi| + p\frac{\hat{I}}{4} \\]

Matrix form:

\\[ \rho_{AB} = \begin{pmatrix} p & 0 & 0 & 2p-2 \\\ 0 & 2-p & 0 & 0 \\\ 0 & 0 & 2-p & 0 \\\ 2p-2 & 0 & 0 & p \end{pmatrix} \\]

The state becomes separable for \\( p \geq \frac{2}{3} \\).

## Evolution and Krauss Operators

Schrödinger equation:

\\[ i\hbar\frac{d}{dt}|\Psi(t)\rangle = \hat{H}|\Psi(t)\rangle, \quad \hat{H} = \hat{H}^\dagger \\]

Time evolution operator:

\\[ |\Psi(t)\rangle = e^{-\frac{i}{\hbar}\hat{H}t}|\Psi(0)\rangle \\]

Krauss operators for non-isolated systems:

\\[ \rho_A = \text{Tr}_E[|\Phi\rangle\langle\Phi|_{AE}] \\]

\\[ \rho_A \to \sum_k \hat{E}_k \rho_A \hat{E}_k^\dagger, \quad \sum_k \hat{E}_k^\dagger \hat{E}_k = \hat{I}_A \\]

## Generalized Measurement

Probability of outcome \\( k \\):

\\[ P_k = \text{Tr}(\hat{M}_k^\dagger \hat{M}_k \rho) \\]

Post-measurement state:

\\[ \rho \to \frac{\hat{M}_k \rho \hat{M}_k^\dagger}{P_k} \\]

POVM (positive operator-valued measure):

\\[ F_k = \hat{M}_k^\dagger \hat{M}_k \\]

## Quantum Information & Shannon Theory

### Number of Required Bits

Shannon entropy for a message of \\( k \\) symbols:

\\[ H = -\sum_{k} p_k \log_2(p_k) \\]

### Von Neumann Entropy

\\[ S(\rho) = -\text{Tr}(\rho \log(\rho)) \\]

For pure states:

\\[ S(\rho) = 0 \\]

For mixed states:

\\[ S(\rho) = -\sum_k p_k \log_2(p_k) \\]

Maximal entropy for a maximally mixed state:

\\[ S(\rho) = \log_2(N) \\]

## Quantum Fourier Transform

\\[ \text{QFT}|x\rangle = \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} e^{2\pi i \frac{xk}{N}} |k\rangle \\]

The QFT circuit uses Hadamard gates and controlled phase-shift gates.

## Shor's Algorithm

### Classical Reduction

Steps for factoring \\( N \\):

  1. Choose \\( a \\) such that \\( 1 < a < N \\).
  2. Compute \\( \text{gcd}(a, N) \\).
  3. Find the period \\( r \\) of \\( f(x) = a^x \mod N \\).



### Quantum Period Finding

Uses the Quantum Fourier Transform to find the period \\( r \\).

## Grover's Algorithm

Initial state:

\\[ |\Phi\rangle = \sin\left(\frac{\theta}{2}\right)|\alpha\rangle + \cos\left(\frac{\theta}{2}\right)|\beta\rangle \\]

The Grover operator:

\\[ G = 2|\Phi\rangle\langle \Phi| - \hat{I} \\]

## Phase Estimation

\\[ \hat{O}|u\rangle = e^{2\pi i \Phi} |u\rangle \\]

Uses the Quantum Fourier Transform to estimate the phase \\( \Phi \\).

## Quantum Teleportation

Initial state:

\\[ (\alpha|0\rangle_A + \beta|1\rangle_A) \otimes \frac{1}{\sqrt{2}}(|0\rangle_B|1\rangle_C - |1\rangle_B|0\rangle_C) \\]

Perform a Bell-basis measurement and apply the appropriate unitary transformation to recover the state.

