import matplotlib.pyplot as plt

def plotComMotion(xs, us):
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    cx = [x[0] for x in xs]
    cy = [x[1] for x in xs]
    cz = [x[2] for x in xs]
    cxdot = [x[3] for x in xs]
    cydot = [x[4] for x in xs]
    czdot = [x[5] for x in xs]
    f_z = [u[0] for u in us]
    # u_x = [u[1] for u in us]
    # u_y = [u[2] for u in us]
    # u_z = [u[3] for u in us]

    plt.plot(cx)
    plt.show()
    plt.plot(cy)
    plt.show()
    plt.plot(cz)
    plt.show()

