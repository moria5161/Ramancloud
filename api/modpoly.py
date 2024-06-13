import numpy as np
from sklearn.linear_model import LinearRegression


def poly(x, degree):
    input_array_for_poly = np.array(x, dtype='object')
    X = np.transpose(np.vstack([input_array_for_poly**k for k in range(degree + 1)]))
    return np.linalg.qr(X)[0][:,1:]

def mod_poly(row, degree=2, repitition=100, gradient=0.001, index=0):
    criteria = np.inf
    ywork = np.array(row)
    yold = np.array(row)
    yorig = np.array(row)
    polx = poly(list(range(1, len(yorig) + 1)), degree)
    nrep = 0

    while (criteria >= gradient) and (nrep <= repitition):
        ypred = LinearRegression().fit(polx, yold).predict(polx)
        ywork = np.minimum(yorig, ypred)
        criteria = np.sum(np.abs((ywork - yold) / yold))
        yold = ywork
        nrep += 1

    corrected = yorig - ypred
    return (corrected, index)

def imod_poly(row, degree=2, repitition=100, gradient=0.001, index=0):
    lin = LinearRegression()
    yold = np.array(row)
    yorig = np.array(row)

    polx = poly(list(range(1, len(yorig) + 1)), degree)
    ypred = lin.fit(polx, yold).predict(polx)
    Previous_Dev = np.std(yorig - ypred)

    yold = yold[yorig <= (ypred + Previous_Dev)]
    polx_updated = polx[yorig <= (ypred + Previous_Dev)]
    ypred = ypred[yorig <= (ypred + Previous_Dev)]

    for i in range(2, repitition + 1):
        if i > 2:
            Previous_Dev = DEV
        ypred = lin.fit(polx_updated, yold).predict(polx_updated)
        DEV = np.std(yold - ypred)

        if np.abs((DEV - Previous_Dev) / DEV) < gradient:
            break
        else:
            for j in range(len(yold)):
                if yold[j] >= ypred[j] + DEV:
                    yold[j] = ypred[j] + DEV

    baseline = lin.predict(polx)
    corrected = yorig - baseline
    return (corrected, index)