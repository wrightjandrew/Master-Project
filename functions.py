import scipy as sp
import numpy as np

def maxEntangledState(nqubits):
    """
    Creates the maximally entangled state of nqubits of a system + nqubits of a reference system.
    """
    V = np.zeros(2**(2*nqubits))
    for i in range(2**nqubits):
        v = np.zeros(2**nqubits)
        v[i] = 1
        V += np.kron(v,v) 
    
    return V/np.linalg.norm(V)

def Fidelity(state1, state2):
    """
    Fidelity between two states.
    """
    return np.abs(np.conj(state1)@state2)**2

def UJFidelity(state1, state2):
    """
    Uhlmann-Josza fidelity between two states.
    """
    rho = np.outer(state1, state1.conj())
    sigma = np.outer(state2, state2.conj())
    return np.trace(sp.linalg.sqrtm(sp.linalg.sqrtm(rho)@sigma@sp.linalg.sqrtm(rho)))**2

def TFD(beta, ham, state):
    """
    Returns the Thermofield double state at inverse temperature beta.
    """
    tfd = sp.linalg.expm(-beta*ham/2)@state
    normalization = np.sqrt(np.conj(tfd)@tfd)
    return 1/normalization*tfd

def commutator(A,B):
    """
    Commutator of two matrices A and B.
    """
    return A@B - B@A

def DBI(iters, H, step, state):
    """
    DBI iterations using the commutator with the Hamiltonian.
    """
    rho = np.outer(state, state.conj())
    newState = np.empty((iters+1,len(state)), dtype=complex)
    newState[0] = state
    for i in range(iters):
        rho = np.outer(newState[i], newState[i].conj())
        comm = commutator(rho, H)
        newState[i+1] = sp.linalg.expm(step*comm)@newState[i]
        newState[i+1] = newState[i+1]/np.sqrt(np.conj(newState[i+1])@newState[i+1])
    return newState

def reflectionOperator(state, step):
    """
    Reflection operator for the DBQITE algorithm.
    """
    rho = np.outer(state, state.conj())
    return sp.linalg.expm(1j*np.sqrt(step)*rho)

def unitaryRecursion(ham, refOperator, step):
    """
    Unitary recursion for the DBQITE algorithm.
    """
    U = sp.linalg.expm(1j*np.sqrt(step)*ham) @ refOperator @ sp.linalg.expm(-1j*np.sqrt(step)*ham)
    return U

def DBQITE(iters, H, step, state):
    """
    DBQITE algorithm.
    """
    ref = reflectionOperator(state, step)
    newState = np.empty((iters+1,len(state)), dtype=complex)
    newState[0,:] = state
    for i in range(iters):
        ref = reflectionOperator(newState[i,:], step)
        U = unitaryRecursion(H, ref, step)
        newState[i+1,:] = U@newState[i,:]
        newState[i+1,:] = newState[i+1,:]/np.sqrt(np.conj(newState[i+1,:])@newState[i+1,:])
    return newState

def DBQITE_thirdOrder(iters, H, step, state):
    """
    DBQITE 3rd order algorithm.
    """
    phi = 0.5*(np.sqrt(5)-1)
    newState = np.empty((iters+1,len(state)), dtype=complex)
    newState[0,:] = state
    for i in range(iters):
        rho = np.outer(newState[i,:], newState[i,:].conj())
        ref1 = sp.linalg.expm(1j*phi*np.sqrt(step)*rho)
        ref2 = sp.linalg.expm(-1j*(phi+1)*np.sqrt(step)*rho)
        U = sp.linalg.expm(1j*phi*np.sqrt(step)*H) @ ref1 @ sp.linalg.expm(-1j*np.sqrt(step)*H) @ ref2 @ sp.linalg.expm(1j*(1-phi)*np.sqrt(step)*H)
        newState[i+1,:] = U@newState[i,:]
        newState[i+1,:] = newState[i+1,:]/np.sqrt(np.conj(newState[i+1,:])@newState[i+1,:])
    return newState


def thermalStatePrepComparison(beta, H, nqubits, method, step = 1e-2):
    """
    Prepares the TFD state at temperature beta and compares it with the final state obtained by the DBI or DBQITE algorithm.
    """
    initState = maxEntangledState(nqubits)
    tfd = TFD(beta, H, initState)
    iters = int(beta/(2*step))
    if method == 'DBI':
        newState = DBI(iters, H, step, initState)
    elif method == 'DBQITE':
        newState = DBQITE(iters, H, step, initState)
    elif method == 'DBQITE_thirdOrder':
        newState = DBQITE_thirdOrder(iters, H, step, initState)
    fidelity = np.abs(UJFidelity(tfd, newState[-1,:]))

    return fidelity

def variance(H, state):
    """
    Variance of the Hamiltonian in the state.
    """
    E = np.conj(state)@H@state
    val = np.conj(state)@H@H@state
    return val - E**2

def skewness(H, V, state):
    """
    Skewness of the Hamiltonian in the state.
    """
    E = np.conj(state)@H@state
    val1 = np.conj(state)@H@H@H@state
    val2 = -3*E*np.conj(state)@H@H@state
    val3 = 2*E**3

    return (val1+val2+val3)/V**(3/2)

def optimalEnergyStep(H,  state):
    """
    Optimal step size at an iteration for minimizing the energy.
    """
    V = variance(H, state)
    S = skewness(H, V, state)
    alpha = np.arccos(1/(np.sqrt(1+0.25*S**2)))
    sOpt = (np.pi-2*alpha)/(4*np.sqrt(V))
    return sOpt

def optimalFidelityStep(H, state, lam0):
    """
    Optimal step size at an iteration for maximizing the fidelity.
    """
    E = np.conj(state)@H@state
    V = variance(H, state)
    delta = E-lam0
    theta = np.arcsin(1/np.sqrt(1+delta**2/V))
    sOpt = (np.pi/2 - theta)/np.sqrt(V)
    return sOpt

def optimalDBI(H, initState, refState, method = "DBI", scheduling = "Fidelity",iters = 20):
    fidelity = np.empty(iters+1)
    fidelity[0] = UJFidelity(refState, initState)
    state = initState
    E0 = sp.linalg.eigvalsh(H)[0]
    steps = np.empty(iters)
    for i in range(iters):
        if scheduling == "Fidelity":
            s = optimalFidelityStep(H, state, E0)
        elif scheduling == "Energy":
            s = optimalEnergyStep(H, state)
        steps[i] = s
        if method == "DBI":
            state = DBI(1,H,s,state)[-1,:]
        elif method == "DBQITE":
            state = DBQITE(1,H,s,state)[-1,:]
        elif method == "DBQITE_thirdOrder":
            state = DBQITE_thirdOrder(1,H,s,state)[-1,:]
        fidelity[i+1] = UJFidelity(refState, state)
        if fidelity[i+1] > 1 - 1e-3:
            fidelity[i+1:] = 1
            i = iters
    return fidelity, state, steps

def thermalStatePrepOptimal(beta, H, nqubits, method = "DBI", scheduling = "Energy"):
    """
    Prepares the TFD state at temperature beta and compares it with the final state obtained by the DBI or DBQITE algorithm.
    """
    initState = maxEntangledState(nqubits)
    tfd = TFD(beta, H, initState)
    totalStepping = 0
    iters = 0
    while totalStepping < beta/2:

        if scheduling == "Fidelity":
            s = optimalFidelityStep(H, initState, sp.linalg.eigvalsh(H)[0])
        elif scheduling == "Energy":
            s = optimalEnergyStep(H, initState)
        totalStepping += s
        # Use optimal time stepping until the it is bigger than beta/2 and then adjust the last step
        if totalStepping > beta/2:
            s = s - totalStepping + beta/2
        if method == "DBI":
            initState = DBI(1,H,s,initState)[-1,:]
        elif method == "DBQITE":
            initState = DBQITE(1,H,s,initState)[-1,:]
        elif method == "DBQITE_thirdOrder":
            initState = DBQITE_thirdOrder(1,H,s,initState)[-1,:]
        
        iters += 1

    fidelity = Fidelity(tfd, initState)

    return fidelity, iters

def defaultStep(H):
    """
    Step size guaranteeing decrease $s = \frac{\Delta}{12\|H\|^3}$.
    """
    eigs = sp.linalg.eigvalsh(H)
    delta = eigs[1]-eigs[0] 
    norm = eigs[-1]
    s = delta/(12*norm**3)
    return s