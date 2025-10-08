import numpy as np
from collections import defaultdict
import pennylane as qml
from docplex.mp.model import Model
from openqaoa.problems import FromDocplex2IsingModel
import matplotlib.pyplot as plt
import time

shots = 5000  # Number of samples used
dev = qml.device("default.qubit", shots=shots)


@qml.qnode(dev)
def qaoa_circuit(gammas, betas, h, J, num_qubits):
    wmax = max(
        np.max(np.abs(list(h.values()))), np.max(np.abs(list(h.values())))
    )  # Normalizing the Hamiltonian is a good idea
    p = len(gammas)
    # Apply the initial layer of Hadamard gates to all qubits
    for i in range(num_qubits):
        qml.Hadamard(wires=i)
    # repeat p layers the circuit shown in Fig. 1
    for layer in range(p):
        # ---------- COST HAMILTONIAN ----------
        for ki, v in h.items():  # single-qubit terms
            qml.RZ(2 * gammas[layer] * v / wmax, wires=ki[0])
        for kij, vij in J.items():  # two-qubit terms
            qml.CNOT(wires=[kij[0], kij[1]])
            qml.RZ(2 * gammas[layer] * vij / wmax, wires=kij[1])
            qml.CNOT(wires=[kij[0], kij[1]])
        # ---------- MIXER HAMILTONIAN ----------
        for i in range(num_qubits):
            qml.RX(-2 * betas[layer], wires=i)
    return qml.sample()


def samples_dict(samples, n_items):
    """Just sorting the outputs in a dictionary"""
    results = defaultdict(int)
    for sample in samples:
        print(f"Inside samples_dict {sample}")
        results["".join(str(i) for i in sample)[:n_items]] += 1
    return results

betas = np.linspace(0, 1, 20)[::-1]  # Parameters for the mixer Hamiltonian
gammas = np.linspace(0, 1, 20)  # Parameters for the cost Hamiltonian (Our Knapsack problem)

# 4. Knapsack Problem -----------------------------------------------------------------------------------
def knapsack_problem(obj_func, set_constraints):
    nu_items = set_constraints.shape[1]
    nu_constr = set_constraints.shape[0]

    mdl = Model(name="Knapsack Problem")
    x = mdl.binary_var_list(nu_items, name="x")

    objective = mdl.sum(x[i] * obj_func[i] for i in range(nu_items))
    mdl.minimize(objective)

    mdl.add_constraint(mdl.sum(x[k] * set_constraints[0, k] for k in range(nu_items)) <= 10)

    mdl.add_constraint(mdl.sum(x[k] * set_constraints[1, k] for k in range(nu_items)) <= 20)

    num_vars = mdl.number_of_variables
    print(mdl.export_as_lp_string())
    return mdl

obj_function = [-3, -4, -5, -4, -3]
set_constraints = np.array([
                            [3, 4, 3, 2, 4],
                            [6, 8, 7, 9, 6]
                           ])

mdl = knapsack_problem(obj_function,set_constraints)

lambda_1, lambda_2 = (
    20,
    20,
)  # Parameters of the unbalanced penalization function (They are in the main paper)
ising_hamiltonian = FromDocplex2IsingModel(
    mdl,
    unbalanced_const=True,
    strength_ineq=[lambda_1, lambda_2],  # https://arxiv.org/abs/2211.13914
).ising_model

h_new = {
    tuple(i): w for i, w in zip(ising_hamiltonian.terms, ising_hamiltonian.weights) if len(i) == 1
}
J_new = {
    tuple(i): w for i, w in zip(ising_hamiltonian.terms, ising_hamiltonian.weights) if len(i) == 2
}

samples_unbalanced = samples_dict(
    qaoa_circuit(gammas, betas, h_new, J_new, num_qubits=5), 5
    )

print(samples_unbalanced)


# Checar si son factibles

mat = np.array([
[3, 4, 3, 2, 4],
[6, 8, 7, 9, 6]])

b = np.array([10, 20])

c = np.array([3, 4, 5, 4, 3])

# Listas para almacenar resultados y repeticiones
resultados = []
repeticiones = []
factibles = []

# Calcular los resultados como los primeros 5 valores de la solución multiplicados por la matriz para el escalar
for clave, veces in samples_unbalanced.items():
    # Extraer las soluciones como vector
    vector = [int(bit) for bit in clave]

    # Convertir en un vector numpy
    vector= np.array(vector)

    # Verificar factibilidad con la matriz de restricciones
    factible = np.all(np.dot(mat, vector) <= b)  # Comparación con b_restricciones

    # Multiplicar por la matriz para el cálculo del escalar
    resultado = np.dot(vector, c)

    # Guardar el resultado escalar, las repeticiones y si es factible o no
    resultados.append(resultado)
    repeticiones.append(veces)
    factibles.append(factible)

# Recorrer las listas de atrás hacia adelante y eliminar los elementos con repetición <=2
#for i in range(len(repeticiones) - 1, -1, -1):
 #   if repeticiones[i] <= 2:
  #      del resultados[i]
   #     del repeticiones[i]
    #    del factibles[i]

# Diccionarios para agrupar repeticiones separadas por factibles e infactibles
factibles_agrupados = defaultdict(int)
infactibles_agrupados = defaultdict(int)

# Agrupar las repeticiones sumando factibles e infactibles por cada resultado
for resultado, rep, factible in zip(resultados, repeticiones, factibles):
    if factible:
        factibles_agrupados[resultado] += rep  # Sumar solo factibles
    else:
        infactibles_agrupados[resultado] += rep  # Sumar solo infactibles

# Obtener los valores únicos de resultados ordenados
resultados_unicos = sorted(set(resultados))  # Ordenamos para mantener el eje X limpio

# Obtener las repeticiones de cada tipo (si no existe el valor, poner 0)
repeticiones_factibles = [factibles_agrupados[r] for r in resultados_unicos]
repeticiones_infactibles = [infactibles_agrupados[r] for r in resultados_unicos]

# Graficar el histograma apilado
plt.figure(figsize=(12, 6))

plt.bar(
    [str(r) for r in resultados_unicos], repeticiones_factibles,
    color='green', label="Factibles"
)

plt.bar(
    [str(r) for r in resultados_unicos], repeticiones_infactibles,
    color='red', bottom=repeticiones_factibles, label="Infactibles"
)

# Etiquetas y formato
plt.xlabel('Energía')
plt.ylabel('Repeticiones')
plt.title('Histograma de soluciones factibles e infactibles')
plt.xticks(rotation=90)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Mostrar gráfico
plt.show()

# Generar el box plot
plt.figure(figsize=(8, 5))
plt.boxplot(resultados, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))

# Etiquetas
plt.xlabel("Valores de Energía")
plt.title("Box Plot de Soluciones")
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Mostrar gráfico
plt.show() # No toma en cuenta las repeticiones
