import sklearn as scik
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

def stdEscal(data, test):
    media = data.mean()
    desviacion = data.std()

    rEscal = (data - media) / desviacion

    rTest = (test - media) / desviacion

    return rEscal, rTest

def robustEscal(data, test):
    mediana = data.median()
    ran_inter = data.quantile(0.75) - data.quantile(0.25)

    rEscal = (data - mediana) / ran_inter

    rTest = (test - mediana) / ran_inter

    return rEscal, rTest

def regLin(dX, dY, modo='', mxIt=10, alfa=0.01):
    if modo == 'OLS':
        lr = LinearRegression()
        lr.fit(dX, dY)
        return lr
    elif modo == 'SGD':
        #sgd = SGDRegressor(max_iter=mxIt, eta0=alfa, random_state=0)
        sgd = SGDRegressor(max_iter=mxIt, learning_rate='constant', eta0=alfa, random_state=0)
        sgd.fit(dX, dY)
        return sgd
    else:
        raise ValueError("Modo no reconocido. Use 'OLS' o 'SGD'.")
    
def regPol(dX, dY, modo='', mxIt=10, alfa=0.01, grado=2):
    poli = PolynomialFeatures(degree=grado, include_bias=False)
    trPoli = poli.fit_transform(dX)
    if modo == 'OLS':
        lr = LinearRegression()
        lr.fit(trPoli, dY)
        return lr, poli
    elif modo == 'SGD':
        #sgd = SGDRegressor(max_iter=mxIt, eta0=alfa, random_state=0)
        sgd = SGDRegressor(max_iter=mxIt, learning_rate='constant', eta0=alfa, random_state=0)
        sgd.fit(trPoli, dY)
        return sgd, poli
    else:
        raise ValueError("Modo no reconocido. Use 'OLS' o 'SGD'.")

def graficador(X, y, model, titulo, poli=None):
    if X.shape[1] != 1: return None

    plt.figure()

    X_plot = X.values.flatten()

    orden = X_plot.argsort()
    X_ordenado = X_plot[orden]
    y_ordenado = y.values[orden]

    if poli:
        X_poly = poli.transform(X)
        y_pred = model.predict(X_poly)[orden]
    else:
        y_pred = model.predict(X)[orden]

    plt.scatter(X_plot, y, color='blue', label='Datos de prueba')
    
    plt.plot(X_ordenado, y_pred, color='red', label='Modelo predicho')

    plt.xlabel("X_test")
    plt.ylabel("y / y_pred")
    plt.legend()
    plt.title(titulo)

    plt.show()

def pipeLine(data, t_test = 0.3, t_train = 0.7, escalador = 'std', modo = 'OLS', polinomial = False, grado = 2, maxIt = 10, alfa = 0.01):
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    trX, teX, trY, teY = scik.model_selection.train_test_split(x, y, test_size= t_test, train_size= t_train, shuffle=True, random_state=0)

    if escalador == 'std':
        tr_, te_ = stdEscal(trX, teX)
    elif escalador == 'robust':
        tr_, te_ = robustEscal(trX, teX)
    else:
        tr_, te_ = trX, teX

    if polinomial:
        modelo, poliT = regPol(tr_, trY, modo=modo, mxIt=maxIt, alfa=alfa, grado=grado)
        y_pred = modelo.predict(poliT.transform(te_))
        graficador(te_, teY, modelo, modo, poli=poliT)
    else:
        modelo = regLin(tr_, trY, modo=modo, mxIt=maxIt, alfa=alfa)
        y_pred = modelo.predict(te_)
        graficador(te_, teY, modelo, modo)

    mse = mean_squared_error(teY, y_pred)
    r2 = r2_score(teY, y_pred)

    print(f"|-Regresión {('polinomial de grado ' + str(grado)) if polinomial else 'lineal'} con {modo} {('y escalamiento '+ escalador) if (escalador == 'std' or escalador == 'robust') else ''} \n\t mse: {mse:.4e} \t r2: {r2:.4e}")

d1 = pd.read_csv("datos.csv")
d2 = pd.read_csv("cal_housing.csv")
mIt = 10000
al = 0.0000001

print("|"*50, "Dataset 1", "|"*50)
pipeLine(d1, escalador='n', modo='OLS', polinomial=False)
pipeLine(d1, escalador='n', modo='OLS', polinomial=True, grado=2)
pipeLine(d1, escalador='n', modo='OLS', polinomial=True, grado=3)
pipeLine(d1, escalador='n', modo='SGD', polinomial=False, maxIt=mIt, alfa=al)
pipeLine(d1, escalador='n', modo='SGD', polinomial=True, grado=2, maxIt=mIt, alfa=al)
pipeLine(d1, escalador='n', modo='SGD', polinomial=True, grado=3, maxIt=mIt, alfa=al)
print("="*50, "Dataset 2", "="*50)
pipeLine(d2, t_test=0.2, t_train=0.8, escalador='n', modo='OLS', polinomial=False)
pipeLine(d2, t_test=0.2, t_train=0.8, escalador='n', modo='OLS', polinomial=True, grado=2)
pipeLine(d2, t_test=0.2, t_train=0.8, escalador='std', modo='OLS', polinomial=True, grado=2)
pipeLine(d2, t_test=0.2, t_train=0.8, escalador='robust', modo='OLS', polinomial=True, grado=2)
pipeLine(d2, t_test=0.2, t_train=0.8, escalador='n', modo='OLS', polinomial=True, grado=3)
pipeLine(d2, t_test=0.2, t_train=0.8, escalador='std', modo='OLS', polinomial=True, grado=3)
pipeLine(d2, t_test=0.2, t_train=0.8, escalador='robust', modo='OLS', polinomial=True, grado=3)