def iterate_complex(c: complex, max_iter: int = 100) -> int:
    """
    Iterates z = z^2 + c starting at z = 0 and returns the number of iterations
    before |z| > 2 or until max_iter is reached.

    Parameters
    ----------
    c : complex
        Complex number to iterate with.
    max_iter : int, optional
        Maximum number of iterations (default is 100).

    Returns
    -------
    int
        Number of iterations before divergence, or max_iter if bounded.
    """
    z = 0
    for i in range(max_iter):
        z = z*z + c
        if abs(z) > 2:
            return i
    return max_iter