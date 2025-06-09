import pulp as pl
import numpy as np
import matplotlib.pyplot as plt

### macierz odleglosci
D = [
    [16.160, 24.080, 24.320, 21.120],
    [19.000, 26.470, 27.240, 17.330],
    [25.290, 32.490, 33.420, 12.250],
    [0.000, 7.930, 8.310, 36.120],
    [3.070, 6.440, 7.560, 37.360],
    [1.220, 7.510, 8.190, 36.290],
    [2.800, 10.310, 10.950, 33.500],
    [2.870, 5.070, 5.670, 38.800],
    [3.800, 8.010, 7.410, 38.160],
    [12.350, 4.520, 4.350, 48.270],
    [11.110, 3.480, 2.970, 47.140],
    [21.990, 22.020, 24.070, 39.860],
    [8.820, 3.300, 5.360, 43.310],
    [7.930, 0.000, 2.070, 43.750],
    [9.340, 2.250, 1.110, 45.430],
    [8.310, 2.070, 0.000, 44.430],
    [7.310, 2.440, 1.110, 43.430],
    [7.550, 0.750, 1.530, 43.520],
    [11.130, 18.410, 19.260, 25.400],
    [17.490, 23.440, 24.760, 23.210],
    [11.030, 18.930, 19.280, 25.430],
    [36.120, 43.750, 44.430, 0.000]
]

### pracochlonnosc
P = [0.1609, 0.1164, 0.1026, 0.1516, 0.0939, 0.1320, 0.0687, 0.0930, 0.2116, 0.2529, 0.0868, 0.0828, 0.0975, 0.8177,
     0.4115, 0.3795, 0.0710, 0.0427, 0.1043, 0.0997, 0.1698, 0.2531]

### OBECNY PRZYDZIAL
A = [
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
]

### OBECNE LOKALIZACJE SIEDZIB
L = [4, 14, 16, 22]


### OBLICZENIA

not_A = np.logical_not(A).astype(int)

def f2(x):
    return np.sum([x[i][j]*not_A[i][j]*P[i] for i in range(len(D)) for j in range(len(D[0]))])*25

def pareto_check(f1_val, f2_val):
    """Eliminate pareto non optimal solutions

    """
    pareto_f1 = []
    pareto_f2 = []
    for i, j in zip(f1_val, f2_val):
        for v, w in zip(f1_val, f2_val):
            if v == i and w == j:
                continue
            if v < i and w < j:
                break
        else:
            pareto_f1.append(i)
            pareto_f2.append(j)
    return pareto_f1, pareto_f2

### TODO

f1_values = []
f2_values = []



x = [[] for _ in range(len(D))]
for i in range(len(D)):
    for j in range(len(D[0])):
        x[i].append(pl.LpVariable(f"x{i}{j}", cat="Binary"))

for epsilon in np.linspace(5,30,100):
    model = pl.LpProblem(name="Pfitzer", sense=pl.LpMinimize)

    model += pl.lpSum(x[i][j] * D[i][j] for i in range(len(D)) for j in range(len(D[0]))), "Objective function"

    for i in range(len(D)):
        model += x[i][0] + x[i][1] + x[i][2] + x[i][3] == 1, f"Region consistency {i}"

    for i in range(len(D[0])):
        model += pl.lpSum(x[j][i]*P[j] for j in range(len(D))) <= 1.1, f"Effortfulness upper boundary {i}"

    for i in range(len(D[0])):
        model += pl.lpSum(x[j][i]*P[j] for j in range(len(D))) >= 0.9, f"Effortfulness lower boundary {i}"

    model += pl.lpSum(x[i][j]*not_A[i][j]*P[i] for i in range(len(D)) for j in range(len(D[0])))*25 <= epsilon, "Constraint"

    status = model.solve()

    print(f"Status: {pl.LpStatus[model.status]}")

    print(f"Obcjective: {model.objective.value()}")

    x_bin = [[x[i][j].value() for j in range(len(x[0]))] for i in range(len(x))]

    print(f"f2 value: {f2(x_bin)}")

    f1_values.append(model.objective.value())
    f2_values.append(f2(x_bin))

f1_values, f2_values = pareto_check(f1_values, f2_values)

save = [[],[]]
fig, ax = plt.subplots(figsize=(12,6))
ax.scatter(f1_values, f2_values, zorder=5, color="#680078")
for i, j in zip(f1_values, f2_values):
    if len(save[0]) == 0 or not (i in save[0] and j in save[1]):
        ax.plot([min(f1_values)-2,i], [j,j], "--r", alpha=0.3)
        ax.plot([i,i], [min(f2_values)-1,j], "--r", alpha=0.3)
        save[0].append(i)
        save[1].append(j)
ax.grid(True, zorder=0)
ax.set_xlabel("f1")
ax.set_ylabel("f2")
ax.set_title("Rozwiązania Pareto optymalne ze stożkami dominacji")
plt.show()