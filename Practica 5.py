import sklearn as scik
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt

def entrenarModeloFullData(dataX, dataY, pesos=None, nNeighbors=3):
    if pesos not in ['uniform', 'distance', None]: 
        raise ValueError("Pesos no reconocidos. Use 'uniform', 'distance' o None.")
    
    if pesos is None:
        model = KNeighborsClassifier(n_neighbors=nNeighbors, weights=None)
    else:
        model = KNeighborsClassifier(n_neighbors=nNeighbors, weights=pesos)

    model.fit(dataX, dataY)
    return model

def entrenarModelo(dataX, dataY, pesos=None, kFolds=3, nNeighbors=3):
    if pesos not in ['uniform', 'distance', None]: 
        raise ValueError("Pesos no reconocidos. Use 'uniform', 'distance' o None.")
    
    kf = scik.model_selection.KFold(n_splits=kFolds)
    accs = []

    for train_id, test_id in kf.split(dataX, dataY):
        X_train, X_test = dataX.iloc[train_id], dataX.iloc[test_id]
        y_train, y_test = dataY.iloc[train_id], dataY.iloc[test_id]

        if pesos is None:
            model = KNeighborsClassifier(n_neighbors=nNeighbors, weights=None)
        else:
            model = KNeighborsClassifier(n_neighbors=nNeighbors, weights=pesos)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = scik.metrics.accuracy_score(y_test, y_pred)
        accs.append(acc)
        
        if acc == max(accs):
            bestModel = model

    bAcc = max(accs)
    print(f"Accuracy para cada pliegue: {accs}.\nMejor accuracy: {bAcc:.4f}.")

    return bestModel, accs

def metricas(model, testX, testY):
    y_pred = model.predict(testX)
    accuracy = scik.metrics.accuracy_score(testY, y_pred)
    mConf = scik.metrics.confusion_matrix(testY, y_pred)
    mcplot = scik.metrics.ConfusionMatrixDisplay(confusion_matrix=mConf)
    mcplot.plot(cmap="BuGn")
    if 0 in model.classes_ or 1 in model.classes_:
        mcplot.ax_.set_xticklabels(['No Spam', 'Spam'], rotation=90)
        mcplot.ax_.set_yticklabels(['No Spam', 'Spam'])
    else:
        mcplot.ax_.set_xticklabels(model.classes_, rotation=90)
        mcplot.ax_.set_yticklabels(model.classes_)

    reporte = scik.metrics.classification_report(testY, y_pred)
    print(reporte)
    plt.show()

    return accuracy

def printReporte(nombre, pliegues, accuracies):
    print("≣"*120)
    print(f"Dataset \t | \t Vecinos  \t | \t Pesos\t \t | \t Pliegue \t | \t Accuracy")
    for i, accs in enumerate(accuracies):
        print(f"{nombre}\t \t | \t {1 if i == 0 else 10}\t \t | \t {'--------    ' if i == 0 else 'uniforme' if i == 1 else 'distancia'}", end="\t | \t")
        for j, acc in enumerate(accs):
            print(f"\t \t | \t \t \t | \t \t \t | \t{j+1} \t\t | \t {acc:.4f}") if j != 0 else print(f" {j+1} \t\t | \t {acc:.4f}")
        print("/"*70, f" Promedio \t | \t {sum(accs)/pliegues:.4f}", "/"*10)
    print("≣"*120)

dIris = pd.read_csv('iris 1.csv')
dEmail = pd.read_csv('emails 1.csv')

dIx = dIris.iloc[:, :-1]
dIy = dIris.iloc[:, -1]

dEx = dEmail.iloc[:, 1:-1]
dEy = dEmail.iloc[:, -1]

tIx, teIx, tIy, teIy = scik.model_selection.train_test_split(dIx, dIy, test_size=0.3, train_size=0.7, shuffle=True, random_state=0)
tEx, teEx, tEy, teEy = scik.model_selection.train_test_split(dEx, dEy, test_size=0.3, train_size=0.7, shuffle=True, random_state=0)

nKFolds = 3
accs = []
modelos = []
bestIris = ()
bestEmail = ()


print(*"-"*50, "Dataset Iris", "-"*50)
modelo, rAN = entrenarModelo(tIx, tIy, pesos=None, kFolds=nKFolds, nNeighbors=1)
acc = metricas(modelo, teIx, teIy)
accs.append(rAN)
modelos.append(modelo)

bestIris = (1, '------', acc)

modelo, rAN = entrenarModelo(tIx, tIy, pesos='uniform', kFolds=nKFolds, nNeighbors=10)
acc = metricas(modelo, teIx, teIy)
accs.append(rAN)
modelos.append(modelo)

if bestIris[2] <= acc:
        bestIris = (10, 'uniform', acc)

modelo, rAN = entrenarModelo(tIx, tIy, pesos='distance', kFolds=nKFolds, nNeighbors=10)
acc = metricas(modelo, teIx, teIy)
accs.append(rAN)
modelos.append(modelo)

if bestIris[2] <= acc:
    bestIris = (10, 'distance', acc)

printReporte("Iris", nKFolds, accs)

print("\n"*3)
accs = []
print(*"-"*50, "Dataset Emails", "-"*50)
modelo, rAN = entrenarModelo(tEx, tEy, pesos=None, kFolds=nKFolds, nNeighbors=1)
acc = metricas(modelo, teEx, teEy)
accs.append(rAN)
modelos.append(modelo)

bestEmail = (1, '-----', acc)

modelo, rAN = entrenarModelo(tEx, tEy, pesos='uniform', kFolds=nKFolds, nNeighbors=10)
acc = metricas(modelo, teEx, teEy)
accs.append(rAN)
modelos.append(modelo)

if bestEmail[2] <= acc:
    bestEmail = (10, 'uniform', acc)

modelo, rAN = entrenarModelo(tEx, tEy, pesos='distance', kFolds=nKFolds, nNeighbors=10)
acc = metricas(modelo, teEx, teEy)
accs.append(rAN)
modelos.append(modelo)

if bestEmail[2] <= acc:
    bestEmail = (10, 'distance', acc)

printReporte("Emails", nKFolds, accs)

################################################################################
#bestIris = (0, 0, 0)
#bestEmail = (0, 0, 0)
#################################################################################

print("\n"*3)
print("="*50, "Pruebas con todo el dataset", "="*50)
print("IRIS: ")
modelo = entrenarModeloFullData(tIx, tIy, pesos=None, nNeighbors=1)
acc = metricas(modelo, teIx, teIy)

if bestIris[2] <= acc:
    bestIris = (1, '----- FD', acc)

modelo = entrenarModeloFullData(tIx, tIy, pesos='uniform', nNeighbors=10)
acc = metricas(modelo, teIx, teIy)

if bestIris[2] <= acc:
    bestIris = (10, 'uniform FD', acc)

modelo = entrenarModeloFullData(tIx, tIy, pesos='distance', nNeighbors=10)
acc = metricas(modelo, teIx, teIy)

if bestIris[2] <= acc:
    bestIris = (10, 'distance FD', acc)

print("EMAILS: ")
modelo = entrenarModeloFullData(tEx, tEy, pesos=None, nNeighbors=1)
acc = metricas(modelo, teEx, teEy)

if bestEmail[2] <= acc:
    bestEmail = (1, '----- FD', acc)

modelo = entrenarModeloFullData(tEx, tEy, pesos='uniform', nNeighbors=10)
acc = metricas(modelo, teEx, teEy)

if bestEmail[2] <= acc:
    bestEmail = (10, 'uniform FD', acc)

modelo = entrenarModeloFullData(tEx, tEy, pesos='distance', nNeighbors=10)
acc = metricas(modelo, teEx, teEy)

if bestEmail[2] <= acc:
    bestEmail = (10, 'distance FD', acc)

print("="*60, "="*60, end="\n"*3)

print("Tabla de resultados finales:")
print("≣"*120)
print("Dataset \t | \t Clasificador \t | \t Vecinos  \t | \t Pesos\t \t | \t Distribucion \t | \t Accuracy")
print("Iris \t \t | \t Naive Bayes \t | \t ---- \t \t | \t ---- \t \t | \t Normal    \t | \t", f"{1:.2f}")
print("\t \t | \t KNN \t \t | \t", bestIris[0], "\t \t | \t", bestIris[1], "\t | \t ---- \t \t | \t", f"{bestIris[2]:.2f}")
print("Emails \t \t | \t Naive Bayes \t | \t ---- \t \t | \t ---- \t \t | \t Normal    \t | \t", f"{0.95:.2f}")
print("\t \t | \t KNN \t \t | \t", bestEmail[0], "\t \t | \t", bestEmail[1], " \t | \t ---- \t \t | \t", f"{bestEmail[2]:.2f}")
print("≣"*120)