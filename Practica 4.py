import sklearn as scik
import pandas as pd
import matplotlib.pyplot as plt

def entrenarModel(dataX, dataY, modelo='Normal'):
    if modelo == 'Normal':
        model = scik.naive_bayes.GaussianNB()
    elif modelo == 'Multinomial':
        model = scik.naive_bayes.MultinomialNB()
    else:
        raise ValueError("Modelo no reconocido. Use 'Normal' o 'Multinomial'.")

    model.fit(dataX, dataY)
    return model

def apKfolds(dataX, dataY, modelo='Normal', k=3):
    kf = scik.model_selection.KFold(n_splits=k)
    kf.get_n_splits(dataX, dataY)
    accs = []
    models = []

    for train_id, test_id in kf.split(dataX, dataY):
        X_train, X_test = dataX.iloc[train_id], dataX.iloc[test_id]
        y_train, y_test = dataY.iloc[train_id], dataY.iloc[test_id]

        if modelo == 'Normal':
            model = scik.naive_bayes.GaussianNB()
        elif modelo == 'Multinomial':
            model = scik.naive_bayes.MultinomialNB()
        else:
            raise ValueError("Modelo no reconocido. Use 'Normal' o 'Multinomial'.")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = scik.metrics.accuracy_score(y_test, y_pred)
        accs.append(acc)
        
        if acc == max(accs):
            rModel = model

    bAcc = max(accs)
    print(f"Acc para cada pliegue: {accs}.\nMejor accuracy: {bAcc:.4f}.")

    return rModel, accs

def metricas(model, testX, testY):
    y_pred = model.predict(testX)
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

def printReporte(nombre, pliegues, accuracies):
    print(f"Dataset \t | \t Distribución \t | \t Pliegue \t | \t Accuracy")
    for i, accs in enumerate(accuracies):
        print(f"{nombre}\t \t | \t {'Normal    ' if i == 0 else 'Multinomial'}", end="\t | \t")
        for j, acc in enumerate(accs):
            print(f"\t \t | \t \t \t | \t {j+1} \t\t | \t {acc:.4f}") if j != 0 else print(f" {j+1} \t\t | \t {acc:.4f}")
        print("="*50, f" Promedio \t | \t {sum(accs)/pliegues:.4f}")

dIris = pd.read_csv('iris.csv')
dEmail = pd.read_csv('emails.csv')

dIx = dIris.iloc[:, :-1]
dIy = dIris.iloc[:, -1]

dEx = dEmail.iloc[:, 1:-1]
dEy = dEmail.iloc[:, -1]

tIx, teIx, tIy, teIy = scik.model_selection.train_test_split(dIx, dIy, test_size=0.3, train_size=0.7, shuffle=True, random_state=0)
tEx, teEx, tEy, teEy = scik.model_selection.train_test_split(dEx, dEy, test_size=0.3, train_size=0.7, shuffle=True, random_state=0)

nFolds = 3

print("-"*50, "Dataset IRIS", "-"*50)
accs = []
modelo, rAN = apKfolds(tIx, tIy, modelo='Normal', k=nFolds)
metricas(modelo, teIx, teIy)
accs.append(rAN)

modelo, rAN = apKfolds(tIx, tIy, modelo='Multinomial', k=nFolds)
metricas(modelo, teIx, teIy)
accs.append(rAN)

printReporte("IRIS", nFolds, accs)

print("="*50, "Dataset EMAILS", "="*50)
accs = []
modelo, rAN = apKfolds(tEx, tEy, modelo='Normal', k=nFolds)
metricas(modelo, teEx, teEy)
accs.append(rAN)

modelo, rAN = apKfolds(tEx, tEy, modelo='Multinomial', k=nFolds)
metricas(modelo, teEx, teEy)
accs.append(rAN)

printReporte("EMAILS", nFolds, accs)

#PRUEBA
modelo = entrenarModel(tEx, tEy, modelo='Multinomial')
metricas(modelo, teEx, teEy)

modelo = entrenarModel(tEx, tEy, modelo='Normal')
metricas(modelo, teEx, teEy)

modelo = entrenarModel(tIx, tIy, modelo='Multinomial')
metricas(modelo, teIx, teIy)

modelo = entrenarModel(tIx, tIy, modelo='Normal')
metricas(modelo, teIx, teIy)

