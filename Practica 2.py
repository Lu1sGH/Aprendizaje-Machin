import sklearn as scik
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def bgd(data, alfa=0.01, maxIt=10, variables=None, mono=True):
    w = np.array([0 for a in range(len(data.columns))], dtype=float)
    a = 0
    todos_W = []

    if mono:
        x = np.array(data[variables[0]].to_list(), dtype=float)
        y = np.array(data[variables[1]].to_list(), dtype=float)
        
        n = len(y)
        while a < maxIt:
            w[1] = w[1] - 2*alfa*np.sum((w[1]*x-y)*x)
            print(f"Iteracion {a}: {w[1]}")
            todos_W.append(w[1].copy())
            a += 1
    else:
        x = np.array(data[variables[:-1]].to_numpy(), dtype=float).T
        y = np.array(data[variables[-1]].to_list(), dtype=float)
        
        while a < maxIt:
            for i in range(len(x)):
                w[i] = w[i] - 2*alfa*np.sum((w[i]*x[i]-y)*x[i])
            print(f"Iteracion {a}: {w[:-1]}")
            todos_W.append(w[:-1].copy())
            a += 1

    return w, todos_W

def error(test, w, variables, mono=True):
    if mono:
        x = np.array(test[variables[0]].to_list(), dtype=float)
        y = np.array(test[variables[1]].to_list(), dtype=float)
        err = 0

        for i, t in zip(x, y):
            err += abs((i*w)-t)
        return err
    else:
        x = np.array(test[variables[:-1]].to_numpy(), dtype=float)
        y = np.array(test[variables[-1]].to_list(), dtype=float)
        err = 0

        for i, t in zip(x, y):
            err += abs(np.sum(i * w)-t)
        return err
    
def rectas(w, test):
    plt.figure()
    x_test = np.array(test[test.columns[0]], dtype=float)
    y_real = np.array(test[test.columns[1]], dtype=float)

    for idx, wi in enumerate(w):
        # Generate line for this iteration
        x_line = np.linspace(min(x_test), max(x_test), 100)
        y_line = wi * x_line

        # Plot regression line
        plt.plot(x_line, y_line, alpha=0.6, label=f"Iteración {idx+1} (w={wi:.5f})")

        # Plot predicted points for this line
        y_pred = x_test * wi
        plt.scatter(x_test, y_pred, alpha=0.7, marker='o')

    # Plot real test data points
    plt.scatter(x_test, y_real, color='red', label="Datos reales", zorder=5)

    plt.xlabel("Terreno (m²)")
    plt.ylabel("Precio (MDP)")
    plt.title("Rectas de Regresión y Predicciones (Monovariable)")
    plt.legend()
    plt.show()

def error_plot(err):
    plt.figure()
    cont = 0
    for i in err:
        plt.plot(cont+1, i, marker='o')
        cont += 1
    plt.xlabel("Iteración")
    plt.ylabel("|y_pred - y_real|")
    plt.title("Errores de estimación")
    plt.show()

dataset1 = pd.read_csv("casas.csv")
dataset2 = pd.read_csv("Dataset_multivariable.csv")

train1, test1 = scik.model_selection.train_test_split(dataset1, test_size=0.3, train_size=0.7, shuffle=True, random_state=0)
train2, test2 = scik.model_selection.train_test_split(dataset2, test_size=0.3, train_size=0.7, shuffle=True, random_state=0)

#Errores de estimación
err1, err2 = [], []

#MONOVARIABLE
print("#"*25, "Monovariable", "#"*25)
w1, todos_W1 = bgd(train1, maxIt=4, alfa=0.00000007, variables=train1.columns, mono=True)
print("\nTest: \n", test1)
print("\nPredicción: \n")
acc = 0
for i in todos_W1:
    tem = []
    for j in test1[test1.columns[0]]:
        tem.append(j*i)
    print(f"Iteración {acc+1}:", tem)
    acc += 1
print("\nError de estimación:\n")
acc = 0
for i in todos_W1:
    print(f"Iteración {acc+1}:", error(test1, i, variables=test1.columns, mono=True))
    acc += 1

for i in todos_W1:
    err1.append(error(test1, i, variables=test1.columns, mono=True))

rectas(todos_W1, test1)
error_plot(err1)

#MULTIVARIABLE
print("\n","#"*25, "Multivariable", "#"*25)
w2, todos_W2 = bgd(train2, alfa=0.000006, maxIt=4, variables=train2.columns, mono=False)
print("\nTest: \n", test2)
print("\nPredicción: \n")
acc = 0
for i in todos_W2:
    tem = []
    for j in test2[train2.columns[:-1]].to_numpy():
        tem.append(np.sum(j*i))
    print(f"Iteración {acc+1}:", tem)
    acc += 1
print("\nError de estimación:\n")
acc = 0
for i in todos_W2:
    print(f"Iteración {acc+1}:", error(test2, i, variables=test2.columns, mono=False))
    acc += 1

for i in todos_W2:
    err2.append(error(test2, i, variables=test2.columns, mono=False))

error_plot(err2)