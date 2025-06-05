import scipy as sp
import numpy as np

def maxEntangledState(nqubits, eigenvectors, coeffs = None):
    """
    Creates the maximally entangled state of nqubits of a system + nqubits of a reference system.
    """
    V = np.zeros(2**(2*nqubits), dtype=complex)
    for i in range(2**nqubits):
        v = eigenvectors[:,i]
        if (coeffs is not None):
            v[i] *= coeffs[i]
        V += np.kron(v,v)
    
    return V/np.linalg.norm(V)

def Fidelity(state1, state2):
    """
    Fidelity between two pure states.
    """
    return np.abs(np.conj(state1)@state2)**2

def UJFidelity(rho, sigma):
    """
    Uhlmann-Josza fidelity between two states.
    """
    return np.real(np.trace(sp.linalg.sqrtm(sp.linalg.sqrtm(rho)@sigma@sp.linalg.sqrtm(rho)))**2)

def TFD(beta, ham, state):
    """
    Returns the Thermofield double state at inverse temperature beta.
    Uses sparse matrix exponential if ham is sparse.
    """
    if sp.sparse.issparse(ham):
        tfd = sp.sparse.linalg.expm_multiply(-beta * ham / 2, state)
    else:
        tfd = sp.linalg.expm(-beta * ham / 2) @ state
    
    normalization = np.sqrt(np.conj(tfd) @ tfd)
    return tfd / normalization

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

def DBIComplex(state, H, step, theta):
    rho = np.outer(state, state.conj())
    comm = commutator(rho, H)
    newState = sp.linalg.expm(1j*theta*rho) @ sp.linalg.expm(step*comm) @ state
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

def DBQITE(iters, H, step, state, trotterization = 1):
    """
    DBQITE algorithm.
    """
    ref = reflectionOperator(state, step)
    newState = np.empty((iters+1,len(state)), dtype=complex)
    newState[0,:] = state
    for i in range(iters):
        U = np.eye(len(state), dtype=complex)
        for j in range(trotterization):
            ref = reflectionOperator(newState[i,:], step/trotterization)
            U = unitaryRecursion(H, ref, step/trotterization) @ U
        newState[i+1,:] = U@newState[i,:]
        newState[i+1,:] = newState[i+1,:]/np.sqrt(np.conj(newState[i+1,:])@newState[i+1,:])
    return newState

def DBQITE_thirdOrder(iters, H, step, state, trotterization = 1):
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

def DBQITE_thirdOrderYudai(iters, H, step, state, trotterization = 1):
    """
    DBQITE 3rd order algorithm.
    """
    phi = (np.sqrt(5)+1)/2
    newState = np.empty((iters+1,len(state)), dtype=complex)
    newState[0,:] = state
    for i in range(iters):
        rho = np.outer(newState[i,:], newState[i,:].conj())
        ref1 = sp.linalg.expm(1j*(1+phi)/np.sqrt(phi)*np.sqrt(step)*rho)
        ref2 = sp.linalg.expm(-1j*(phi+1)/np.sqrt(phi)*np.sqrt(step)*rho)
        U = sp.linalg.expm(1j*np.sqrt(phi)*np.sqrt(step)*H) @ ref1 @ sp.linalg.expm(-1j*np.sqrt(step)*H/np.sqrt(phi)) @ ref2 @ sp.linalg.expm(-1j*np.sqrt(phi)*np.sqrt(step)*H)
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
    return np.real(val - E**2) 

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
    # """
    # Optimal step size at an iteration for minimizing the energy.
    # """
    # V = variance(H, state)
    # S = skewness(H, V, state)
    # alpha = np.arccos(1/(np.sqrt(1+0.25*S**2)))
    # sOpt = (np.pi-2*alpha)/(4*np.sqrt(V))
    # # alpha = np.arctan(-2/S)
    # # sOpt = -alpha/(2*np.sqrt(V))
    """
    Optimal step size at an iteration for minimizing the energy.
    """
    V = variance(H, state)
    S = skewness(H, V, state)
    # alpha = np.arccos(1/(np.sqrt(1+0.25*S**2)))
    # sOpt = (np.pi-2*alpha)/(4*np.sqrt(V))
    alpha = np.arctan(-2/S)
    if alpha > 0:
        alpha = -np.pi + alpha

    sOpt = -alpha/(2*np.sqrt(V))
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
    fidelity[0] = np.abs(refState.conj().T @ initState)**2
    state = initState
    E0 = sp.linalg.eigvalsh(H)[0]
    steps = np.empty(iters)
    energy = np.empty(iters+1)
    energy[0] = np.real(np.conj(state) @ H @ state)
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
        fidelity[i+1] =  np.abs(refState.conj().T @ state)**2
        energy[i+1] = np.real(np.conj(state) @ H @ state)
        if fidelity[i+1] > 1 - 1e-3:
            fidelity[i+1:] = 1
            i = iters
    return fidelity, state, steps, energy

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
        # Use optimal time stepping until the total time is bigger than beta/2 and then adjust the last step
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

def bestApproximatingStep(H, state, tau):
    E = np.conj(state)@H@state
    V = variance(H, state)
    numerator = 1-E*tau
    # numerator = (np.eye(len(state))-tau*H)@state
    # numerator = np.linalg.norm(numerator)
    denominator = np.sqrt((1-E*tau)**2 + V*tau**2)
    s = 1/np.sqrt(V)*np.arccos(numerator/denominator)

    return s

def thermalStatePrepBest(beta, H, nqubits, method ='DBI', K = 10):
    initState = tfd0(nqubits)
    tfd = TFD(beta, H, initState)
    for i in range(K):
        s = bestApproximatingStep(H, initState, beta/(2*K))
        if method == 'DBI':
            initState = DBI(1,H,s,initState)[-1]
        elif method == 'DBQITE':
            initState = DBQITE(1,H,s,initState)[-1]
        elif method == 'DBQITE_thirdOrder':
            initState = DBQITE_thirdOrder(1,H,s,initState)[-1]
    #fidelity = UJFidelity(tfd, initState)
    fidelity = Fidelity(tfd, initState)
    return fidelity

def thermalStatePrepKNumbers(beta, eps, H, nqubits, method ='DBI', K = 0):
    initState = tfd0(nqubits)

    tfd = TFD(beta, H, initState)
    fidelity = Fidelity(tfd, initState)
    while fidelity < 1-eps:
        K += 1
        initState = tfd0(nqubits)

        for i in range(K):
            s = bestApproximatingStep(H, initState, beta/(2*K))
            if method == 'DBI':
                initState = DBI(1,H,s,initState)[-1]
            elif method == 'DBQITE':
                initState = DBQITE(1,H,s,initState)[-1]
            elif method == 'DBQITE_thirdOrder':
                initState = DBQITE_thirdOrder(1,H,s,initState)[-1]
        
        #fidelity = UJFidelity(tfd, initState)
        fidelity = Fidelity(tfd, initState)
    return K

def c_kl (k,l):
    if l == 1:
        return 1
    else:
        return (-1)**(l-1)*(sp.special.comb(2*k+2,2*l)-sp.special.comb(2*k+1,2*l-1))
    
def d_kl (k,l):
    if l == 0:
        return 1
    elif (4*l == 2*k+2):
        return (-1)**(l-1)*(sp.special.comb(2*k+1,2*l))
    else:
        return (-1)**(l-1)*(sp.special.comb(2*k+2,2*l)+sp.special.comb(2*k+1,2*l+1))
    
def moment(H, state, k):
    """
    k-th moment of the Hamiltonian in the state.
    """
    E = np.conj(state)@H@state
    operator = np.eye(len(H), dtype=complex)
    for i in range(k):
        operator @= (H - E*np.eye(len(H)))
    val = np.conj(state)@operator@state
    return np.real(val)

def expectation(H, state,k):
    """
    k-th moment of the Hamiltonian in the state.
    """
    operator = np.eye(len(H), dtype=complex)
    for i in range(k):
        operator @= H
    val = np.conj(state)@operator@state
    return np.real(val)

def energyDiffApproximation(s, H, state, k):
    evenTerm = 0
    oddTerm = 0
    moments = np.empty(2*k+2+1)
    for i in range(len(moments)):
        moments[i] = moment(H, state, i)
    for i in range(k+1):
        coeffEven = (-s)**k/(sp.special.factorial(2*k))
        coeffOdd = (-1)**k *s**((2*k+1)/2)/(sp.special.factorial(2*k+1))
        for l in range(k):
            evenTerm += coeffEven*c_kl(k,l)*moments[2*k+1-2*l]*moments[2*l]
            oddTerm += coeffOdd*d_kl(k,l)*moments[2*k+2-2*l]*moments[2*l]
        oddTerm += coeffOdd*d_kl(k,k)*moments[2]*moments[2*k]

    return -2*(1-np.cos(np.sqrt(s)))*evenTerm - 2*np.sin(np.sqrt(s))*oddTerm
        
    
def energyDiffApproximation2(t, H, state, order):
    val = 0
    E = np.real(np.conj(state)@H@state)
    for k in range(order+1):
        for l in range(order+1):    
            for s in range(1,order+1):
                if (k+l+s <= order):
                    coeff = (-1)**l*(1j)**(k+l)*t**(k+l+s)/(sp.special.factorial(k)*sp.special.factorial(s))*expectation(H, state, l)
                    val += coeff*E/(sp.special.factorial(2*s))*expectation(H, state, k)
                    if (k+l+s %2 == 0):
                        val += coeff*(-1)**s*1j**s/(sp.special.factorial(s))*moment(H, state, k+1)

    return np.real(val)

def decomposeInEigenBasis(eigenvectors, state):
    n = len(eigenvectors)
    coeff = np.zeros(n, dtype = complex)
    for i in range(n):
        coeff[i] = state.conj().T @ eigenvectors[:,i]
    return coeff

def energy(H, state):
    if state.ndim == 1:
        return np.real(np.conj(state).T @ H @ state)
    elif state.ndim == 2: 
        return np.real(np.trace(state @ H))
    else:
        raise ValueError("State must be either a vector or a square matrix.")

def countOnes(n):
    binary = bin(n)
    count = 0
    for i in range(len(binary)):
        if binary[i] == '1':
            count += 1
    return count

def tfd0(n):
    tfd = np.zeros(2**(2*n))
    for i in range(2**n):
        state = np.zeros(2**n)
        state[i] = 1
        state2 = np.zeros(2**n)
        state2[2**n-1-i] = 1
        ones = countOnes(i)
        if ones%2 == 0:
            sign = 1
        else:
            sign = -1

        tfd += sign*np.kron(state, state2)
    return tfd/np.sqrt(2**n)    

def matrixPolynomialScheduling(state, H, coeff):

    E = energy(H, state)
    V = variance(H, state)
    # if E == 0:
    #     s = -1/ np.sqrt(V) * np.arccos( np.abs(E-coeff) / np.sqrt(V + np.abs(E-coeff)**2) )
    #     theta = np.angle ( (E-coeff) / np.abs(E-coeff) )
    # else:
    #     # s = -np.sign(E) / np.sqrt(V) * np.arccos( np.abs(E-coeff) / np.sqrt(V + np.abs(E-coeff)**2) )
    #     # theta = np.angle ( np.sign(E)*(E-coeff) / np.abs(E-coeff) )
    s = -1 / np.sqrt(V) * np.arccos( np.abs(E-coeff) / np.sqrt(V + np.abs(E-coeff)**2) )
    theta = np.angle ( (E-coeff) / np.abs(E-coeff) )
    return s, theta

def matrixPolynomialEvolution(initState, H, coeffs):

    steps = len(coeffs)
    state = initState.copy()
    I = np.eye(len(state))

    for i in range(steps):
        
        operator = (H-coeffs[i]*I)
        state = operator @ state

    state = state / np.linalg.norm(state)
    return state

def matrixPolynomialEvolutionDBI(initState, H, coeffs, method = "DBI"):

    steps = len(coeffs)
    state = initState.copy()

    for i in range(steps):
        coeff = coeffs[i]
        s, theta = matrixPolynomialScheduling(state, H, coeff)
        rho = np.outer(state, state.conj())
        W = commutator(rho, H)
        if method == "DBI":
            state = sp.linalg.expm(1j*theta*rho) @ sp.linalg.expm(s*W) @ state
        elif method == "DBQITE":
            state = sp.linalg.expm(-1j*theta*rho) @ sp.linalg.expm(1j*np.sqrt(np.abs(s))*H) @ sp.linalg.expm(1j*np.sqrt(np.abs(s))*rho) @ sp.linalg.expm(-1j*np.sqrt(np.abs(s))*H) @ state
    state = state / np.linalg.norm(state)
    return state

def truncatedExponentialPoly(N, s):
    return [s**k/ sp.special.factorial(k) for k in range(N + 1)]

def findRoots(N, s):
    coefficients = truncatedExponentialPoly(N, s)
    roots = np.roots(coefficients[::-1])  # highest degree first
    
    return roots

def thermalStatePrepMatrixPolynomial(H, nqubits, beta, order):
    """
    Prepares the TFD state at temperature beta and compares it with the final state obtained by the DBI or DBQITE algorithm.
    """
    initState = tfd0(nqubits)
    tfd = TFD(beta, H, initState)
    coeffs = findRoots(order, -beta/2)

    newState = matrixPolynomialEvolutionDBI(initState, H, coeffs)
    fidelity = Fidelity(tfd, newState)

    return fidelity