import sklearn as scik
import pandas as pd

dataset = pd.read_csv("metodosDeValidacion.csv")

train, test = scik.model_selection.train_test_split(dataset, test_size=0.3, train_size=0.7, shuffle=False)
print("#"*30, "Conjunto de Entrenamiento", "#"*30)
print(train)
print("#"*30, "Conjunto de Prueba", "#"*30)
print(test)

print("#"*30, "KFold()", "#"*30)
kf = scik.model_selection.KFold(n_splits=6)
kf.get_n_splits(train)

for i, (train_id, test_id) in enumerate(kf.split(train)):
    print(f"Iteracion {i}:")
    print(f"  Indices de entrenamiento: {train_id}")
    print(f"  Indices de prueba: {test_id}")
    print(f"  Conjunto de entrenamiento: [", end="")
    for ii in train_id:
        print(f" ({train.iloc[ii].to_dict()['x']},{train.iloc[ii].to_dict()['y']}) ", end="" if ii != train_id[-1] else "]\n")
    print(f"  Conjunto de prueba: [", end="")
    for iii in test_id:
        print(f" ({train.iloc[iii].to_dict()['x']},{train.iloc[iii].to_dict()['y']}) ", end="" if iii != test_id[-1] else "]\n\n")

print("#"*30, "LeaveOneOut()", "#"*30)
loo = scik.model_selection.LeaveOneOut()
loo.get_n_splits(train)
for i, (train_id, test_id) in enumerate(loo.split(train)):
    print(f"Iteracion {i}:")
    print(f"  Indices de entrenamiento: {train_id}")
    print(f"  Indices de prueba: {test_id}")
    for ii in train_id:
        print(f" ({train.iloc[ii].to_dict()['x']},{train.iloc[ii].to_dict()['y']}) ", end="" if ii != train_id[-1] else "]\n")
    print(f"  Conjunto de prueba: [", end="")
    for iii in test_id:
        print(f" ({train.iloc[iii].to_dict()['x']},{train.iloc[iii].to_dict()['y']}) ", end="" if iii != test_id[-1] else "]\n\n")

print("#"*30, "Resample()", "#"*30)
c1 = scik.utils.resample(train, n_samples=9)
c2 = scik.utils.resample(train, n_samples=9)

p1 = train.loc[~train.index.isin(c1.index)]
p2 = train.loc[~train.index.isin(c2.index)]

print(f"Conjunto de entrenamiento 1: \n {c1}")
print(f"Conjunto de prueba 1: \n {p1}")
print(f"Conjunto de entrenamiento 2: \n {c2}")
print(f"Conjunto de prueba 2: \n {p2}")
