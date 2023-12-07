# utility functions for midterm2
# code adapted from: https://medium.com/eatpredlove/natural-cubic-splines-implementation-with-python-edf68feb57aa

import numpy as np

def cubic_spline(x, y):
    """Interpolate using knot-a-knot cubic spline method.
    """
    x = np.array(x)
    y = np.array(y)
    # check that data is sorted in increasing order
    if np.any(np.diff(x) < 0):
        idx = np.argsort(x)
        x = x[idx]
        y = y [idk]

    size = len(x)
    dx = np.diff(x)
    dy = np.diff(y)
    div_diff = dx / dy

    # get coefficient matrix
    A = np.zeros(shape = (size, size))
    A[0, 0] = -dx[1]
    A[0, 1] = dx[0] + dx[1]
    A[0, 2] = -dx[0]
    A[-1, -1] = -dx[-2]
    A[-1, -2] = dx[-2] + dx[-1]
    A[-1, -3] = -dx[-1]
    # set up RHS vector b
    rhs = np.zeros(size)
    rhs[1:-1] = 3 * np.diff(div_diff)
    # set boundary conditions
    rhs[0] = 0
    rhs[-1] = 0

    # set up diagonals
    for i in range(1, size - 1):
        A[i, i - 1] = dx[i - 1]
        A[i, i + 1] = dx[i]
        A[i, i] = 2 * (dx[i - 1] + dx[i])
    
    # print(f"Coefficient Matrix [A] =\n{A}\n")
    # print(f"RHS (b) =\n {b}\n")

    c = gauss_iter_solver(A, rhs)
    print(f"c (gauss-seidel) =\n {c}\n")
    c1 = np.linalg.solve(A, rhs)
    print(f"c (np.solve) =\n{c1}\n")

    # solve for remaining coefficients
    d = np.zeros(shape= (size-1, 1))
    b = np.zeros(shape = (size-1, 1))
    a = y[:-1]

    for i in range (0, len(d)):
        d[i] = (c[i+1] - c[i]) / (3 * dx[i])
        b[i] = (dy[i] / dx[i]) - c[i] * dx[i] - d[i] * dx[i] ** 2

    print(f"d coefficients (di) =\n {d}\n")
    print(f"b coefficients (bi) =\n {b}\n")
    print(f"a coefficients (ai) =\n {a}\n")


    return a, b, c, d


def gauss_iter_solver(A, b, x0 = None, tol = 1e-8, alg = 'seidel'):
    """ This function implements the iterative Gauss-Seidel Approach to solve linear systems of the form Ax = b.

    Parameters
    ----------
    A: array-like, shape(n, n)
       coefficient matrix of linear system

    b: array-like, shape (n, m) where m >= 1
       right-hand side of linear system

    x0: (optional) array-like, shape (n,) or (n,m)
        initial guesses to solve system

    tol: (optional) float
         stopping criterion

    alg: (optional) string flag
         lists algorithm, the two acceptable inputs are seidel or jacobi

    Returns
    -------
    numpy.array, shape (n,m)
        The solution vector

    Raises
    ------
    TypeError: 
        Checks that the alg flag is a string and contains either 'seidel' or 'jacobi'.

    ValueError:
        If coefficient matrix A is not 2D and square
        If rhs vector b is not 1D or 2D, or has a different number of rows than A
        If the initial guess x0 is not 1D or 2D, has a different shape than b, or has a different number of rows than A and b

    RuntimeWarning:
        If the system does not converge by 100 iterations to a specified error tolerance.
    """
    # check that coefficient matrix and constant vector are valid inputs
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    # check that the coefficient matrix is square
    n = len(A)
    ndim = len(A.shape)
    if ndim != 2:
        raise ValueError(f"A has {ndim} dimensions"
                         + ", should be 2")
    if A.shape[1] != n:
        raise ValueError(f"A has {n} rows and {A.shape[1]} cols"
                         + ", should be square")

    # check that the rhs vector is 1D or 2D
    ndimb = len(b.shape)
    if ndimb not in [1, 2]:
        raise ValueError(f"b has {ndimb} dimensions"
                         + ", should be 1D or 2D")
    # check that number of rhs rows matches number of rows in A
    if len(b) != n:
        raise ValueError(f"A has {n} rows, b has {len(b)} values"
                         + ", dimensions incompatible")
     # if b is 1D convert b to a 2D column vector
    if b.ndim == 1:
        b = np.reshape(b, (n, 1))
                        
    # check if the alg flag is either siedel or jacobi (case insensitive and ignores trailing/leading whitespaces)
    alg = alg.strip().lower()
    if alg not in ['seidel', 'jacobi']:
        raise TypeError(f"The algorithm flag ({alg}) contains a string other than 'seidel' or 'jacobi'")

    # check and initialise x0
    if x0 is None:
        # make column vector x0 using column vector b
        x0 = np.zeros_like(b)
    else:
        # make sure x0 is an np.array
        x0 = np.array(x0, dtype=float)
        if x0.ndim == 1:
            # reshape x0 to match dimensions of b and convert to column vector
            x0 = np.reshape(x0, (n,1))
        if x0.ndim not in [1,2]:
            raise ValueError(f"x0 has {x0.ndim}, should be 1D or 2D")
        # make sure x0 has the same number of rows as A and b
        if len(x0) != n:
            raise ValueError(f"x0 has {x0.shape[0]} rows, A and b have {n} rows, dimemsions incompatible")
    
    # set number of maximum iterations (solution must converge before this number is reached)
    maxit = 100
    # approximate relative error variable, ensures that loop will execute at least once
    eps_a = 2 * tol
    # set up A_d with main diagonal entries of coefficient matrix A and zeros elsewhere
    A_d = np.diag(np.diag(A))
    # inverse of A_d (note: this is computationally inexpensive because it only involves scalar inversion of each diagonal entry)
    A_d_inv = np.linalg.inv(A_d)
    # Determine normalized matrix A^*
    A_ = A - A_d
    A_star = A_d_inv @ A_
    # Determine normalized matrix B^*
    B_star = A_d_inv @ b
    
    # set iteration counter
    itr = 1
    while np.max(eps_a) > tol and itr < maxit:
        if alg == 'jacobi':
            x_old = np.array(x0)
            x0 = B_star - (A_star @ x_old)
        elif alg == 'seidel':
            x_old = np.array(x0)
            for i, j in enumerate(A):
                x0[i,:] = B_star[i:(i+1),:] - A_star[i:(i+1),:] @ x0
        # calculate error at each iteration
        num = x0 - x_old
        eps_a = np.linalg.norm(num) / np.linalg.norm(x0)
        itr += 1
        # system must converge over 100 iterations, if it does not, a runtime warning is raised
        if itr >= maxit:
            raise RuntimeWarning(f"The system did not converge over {itr} iterations. The approximate error ({np.max(eps_a)}) is greater than ({tol}) the specified error tolerance.")

    return x0