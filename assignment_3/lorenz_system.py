def lorenz(t, W, sigma=10.0, r=28.0, b=8.0/3.0):
    """
    Computes the derivatives for the Lorenz system.

    Parameters
    ----------
    t : float
        Current time (not used directly but required by solver).
    W : array_like
        Current state [X, Y, Z].
    sigma : float
        Prandtl number.
    r : float
        Rayleigh number.
    b : float
        Dimensionless length scale.

    Returns
    -------
    dWdt : list of float
        Derivatives [dX/dt, dY/dt, dZ/dt].
    """
    X, Y, Z = W
    dX = -sigma * (X - Y)
    dY = r * X - Y - X * Z
    dZ = -b * Z + X * Y
    return [dX, dY, dZ]