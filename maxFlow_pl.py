from docplex.mp.model import Model

# Dati del grafo (nodi e capacità sugli archi)
nodes = [1, 2, 3, 4, 5, 6]
edges = {
    (1, 2): 3,
    (1, 3): 3,
    (1, 4): 2,
    (2, 5): 1,
    (2, 4): 1,
    (3, 4): 1,
    (3, 6): 2,
    (4, 5): 2,
    (4, 6): 2,
    (5, 6): 1,
}

source = 1
sink = 6

# Crea il modello
mdl = Model(name='MaxFlow')

# Variabili di flusso
x = { (i, j): mdl.continuous_var(lb=0, ub=cap, name=f"x_{i}_{j}") 
      for (i, j), cap in edges.items() }

# Variabile del valore del flusso massimo
v = mdl.continuous_var(lb=0, name="v")

# Vincoli di bilancio di flusso
for node in nodes:
    # Flusso in uscita dal nodo
    out_flow = mdl.sum(x[i, j] for (i, j) in edges if i == node)
    
    # Flusso in entrata nel nodo
    in_flow = mdl.sum(x[i, j] for (i, j) in edges if j == node)

    if node == source:
        mdl.add_constraint(out_flow - in_flow == v, ctname=f"flow_source_{node}")
    elif node == sink:
        mdl.add_constraint(out_flow - in_flow == -v, ctname=f"flow_sink_{node}")
    else:
        mdl.add_constraint(out_flow - in_flow == 0, ctname=f"flow_node_{node}")

# Funzione obiettivo: massimizza il flusso totale
mdl.maximize(v)

# Risoluzione del modello
solution = mdl.solve(log_output=True)

# Output risultati
if solution:
    print(f"Max Flow: {v.solution_value}")
    for (i, j) in edges:
        flow = x[i, j].solution_value
        if flow > 0:
            print(f"Flow x_{i}_{j} = {flow}")
else:
    print("Nessuna soluzione trovata.")
