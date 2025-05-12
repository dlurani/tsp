import time

import numpy as np
from docplex.mp.model import Model
from docplex.cp.expression import INFINITY

import matplotlib.pyplot as plt


def addCuts(x,capacity_cuts, tsp_model, nodes, p):
    while True:

        sol = tsp_model.solve()

        # From the solution extract the set of selected arcs
        edges = [e.name for e in tsp_model.iter_continuous_vars() if e.solution_value > 0]
        edges = [tuple([int(c) for c in e.split('_')[1:]]) for e in edges]

        # The value of the variable corresponding to an arc e is considered as capacity in the max flow problem
        cap = {}
        for e in edges:
            cap[e] = x[e].solution_value

        plot(nodes, edges, p, sol.objective_value, cap)

        # Consider 1 as sink and solve a max flow problem from s to the other nodes
        s = 1
        stop = True

        for t in nodes:

            if s != t:

                sstar = []

                phi = ford_fulkerson(nodes, s, t, sstar, edges, cap)

                if phi < 1 and str(sstar) not in capacity_cuts:

                    print("Found a cut: flow from", s, "to", t, "is", phi, ". sstar:", sstar)
                    print("Added: ", sum([x[(i, j)] for i in sstar[0] for j in sstar[1]]), ' >= 1')
                    tsp_model.add(sum([x[(i, j)] for i in sstar[0] for j in sstar[1]]) >= 1)

                    capacity_cuts.append(str(sstar))
                    stop = False

        time.sleep(2)

        if stop:
            return tsp_model, sol, edges


def main():

    # define number of nodes
    N = 30
    nodes = list(range(1, N + 1))

    # Generating random coordinates
    p = np.random.uniform(low=0.0, high=10.0, size=(N, 2))

    # Computing euclidean cost for each pair of nodes
    c = {}
    for i in nodes:
        for j in nodes:
            c[(i, j)] = np.sqrt((p[i - 1, 0] - p[j - 1, 0]) ** 2 + (p[i - 1, 1] - p[j - 1, 1]) ** 2) + j / 10

    # plot(nodes, {}, p, 0, {})

    # Integer Linear Programming Model
    tsp_model = Model("TSP")
    x = tsp_model.continuous_var_matrix(keys1=nodes, keys2=nodes, name='x', lb=0.0, ub=1.0)

    # All nodes must have exactly 1 outgoing arc 2)
    for i in nodes:
        out_edges = 0
        for j in nodes:
            if i != j:
                out_edges += x[(i, j)]
        tsp_model.add(out_edges == 1)

    # All nodes must have exactly 1 incoming arc 3)
    for j in nodes:
        in_edges = 0
        for i in nodes:
            if i != j:
                in_edges += x[(i, j)]
        tsp_model.add(in_edges == 1)

    for i in nodes:
        flow = 0
        for j in nodes:
            if i != j:
                flow += x[(i, j)]
                flow -= x[(j, i)]
        tsp_model.add(flow == 0)

    # Minimize the total travel distance
    tsp_model.set_objective("min", sum(x[(i, j)] * c[(i, j)] for i in nodes for j in nodes))

    tsp_model.prettyprint()
    tsp_model.print_information()

    capacity_cuts = []

    tsp_model,sol,edges= addCuts(x, capacity_cuts, tsp_model, nodes, p)

    return tsp_model, sol, edges,nodes,p


def plot(nodes, edges, p, objective_value, cap):
    # Create a figure
    plt.figure(figsize=(8, 6))

    # Plot nodes
    plt.scatter(p[:, 0], p[:, 1], c='lightblue', s=200)

    # Annotate nodes
    for i in nodes:
        x_p = float(p[i - 1, 0])
        y_p = float(p[i - 1, 1])
        plt.text(x_p, y_p, str(i), fontsize=12, ha='right')

    for e in edges:
        start_x, start_y = tuple(p[e[0] - 1, :])
        end_x, end_y = tuple(p[e[1] - 1, :])

        plt.plot([start_x, end_x], [start_y, end_y], 'r-', linewidth=2)

        # Calculate the midpoint of the edge
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2

        # Add a label at the midpoint
        plt.text(mid_x, mid_y, "{:1.2f}".format(cap[e]), fontsize=10, ha='center', color='blue')

    plt.title('TSP Solution: ' + str(objective_value))
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    #plt.show(block=False)
    plt.draw()
    plt.pause(0.1)


def ford_fulkerson(nodes, s, t, sstar, edges, cap):

    # maximum flow
    phi = 0

    x = {}
    for e in edges:
        x[e] = 0

    while True:

        # Update labels
        prev = {}
        for j in nodes:
            prev[j] = 0

        eps = {}
        for j in nodes:
            eps[j] = INFINITY

        Q = [s]
        prev[s] = s

        # bfs
        while len(Q) != 0 and prev[t] == 0:

            h = Q.pop()

            # scan for direct arcs starting from node h
            for j in nodes:

                if (h, j) in edges and x[(h, j)] < cap[(h, j)] and prev[j] == 0:

                    eps[j] = min(eps[h], cap[(h, j)] - x[(h, j)])
                    prev[j] = h
                    Q.append(j)

            # scan for reversed arcs ending in node h
            for i in nodes:

                if (i, h) in edges and x[(i, h)] > 0 and prev[i] == 0:

                    eps[i] = min(eps[i], x[(i, h)])
                    prev[i] = -h
                    Q.append(i)

        # An incrementing path has been found
        if prev[t] != 0:

            delta = eps[t]
            phi += delta
            j = t

            while j != s:

                i = prev[j]

                if i > 0:

                    x[(i, j)] += delta

                else:

                    x[(j, -i)] -= delta

                j = abs(i)

            continue

        # No incrementing path has been found. Stop the algorithm
        else:

            sstar.append([])
            sstar.append([])

            for j in nodes:

                if prev[j] != 0:
                    sstar[0].append(j)
                else:
                    sstar[1].append(j)

            break

    return phi


if __name__ == '__main__':

   tsp_model, sol, edges,nodes,p = main()

   # Branch and Bound
   def branch_and_bound(original_model, initial_solution,nodes, p):
       print("\n--- Avvio algoritmo Branch and Bound ---")
       
       # Inizializziamo la migliore soluzione intera trovata
       best_integer_solution = None
       best_objective = float('inf')
       
       # Coda dei problemi da risolvere
       problem_queue = []
       
       # Verifichiamo se la soluzione iniziale è già intera
       fractional_vars = []
       for var in original_model.iter_continuous_vars():
           if 0.01 < var.solution_value < 0.99:
               fractional_vars.append((var, var.solution_value))
       
       if not fractional_vars:
           print("La soluzione iniziale è già intera!")
           return initial_solution, False,{}
       
       # Aggiungiamo il problema iniziale alla coda
       problem_queue.append((original_model.clone(), initial_solution, []))
       
       # Contatore dei nodi esplorati
       nodes_explored = 0
       
       while problem_queue:
           # Prendiamo il prossimo problema da risolvere
           current_model, current_solution, branching_decisions = problem_queue.pop(0)
           nodes_explored += 1
           
           print(f"\nAnalisi nodo {nodes_explored}")
           print(f"Decisioni di branching finora: {branching_decisions}")
           print(f"Miglior obiettivo intero finora: {best_objective}")
           
           # Se la soluzione corrente ha un valore obiettivo peggiore della migliore soluzione intera trovata,
           # possiamo fare pruning
           if current_solution and current_solution.objective_value >= best_objective:
               print("Pruning - Limite inferiore peggiore della migliore soluzione intera")
               continue
           
           # Risolviamo il modello corrente
           try:
               sol = current_model.solve()
               if not sol:
                   print("Problema non risolvibile")
                   continue
               
               # Estraiamo gli archi della soluzione
               edges = [e.name for e in current_model.iter_continuous_vars() if e.solution_value > 0]
               edges = [tuple([int(c) for c in e.split('_')[1:]]) for e in edges]
               
               # Cerchiamo variabili frazionarie
               fractional_vars = []
               for var in current_model.iter_continuous_vars():
                   if 0.01 < var.solution_value < 0.99:
                       fractional_vars.append((var, var.solution_value))
               
               # Se non ci sono variabili frazionarie, abbiamo trovato una soluzione intera
               if not fractional_vars:
                   print(f"Trovata soluzione intera con obiettivo: {sol.objective_value}")
                   if sol.objective_value < best_objective:
                       best_objective = sol.objective_value
                       best_integer_solution = (current_model, sol, edges)
                       print("Nuova migliore soluzione intera trovata!")
                   continue
               
               # Ordiniamo le variabili frazionarie in base alla distanza da 0.5
               # più vicino a 0.5 = più indeciso e probabilmente più influente
               fractional_vars.sort(key=lambda x: abs(x[1] - 0.5))
               
               # Prendiamo la variabile più "indecisa"
               branching_var, branching_val = fractional_vars[0]
               var_name = branching_var.name
               print(f"Branching sulla variabile {var_name} con valore {branching_val}")
               
               # Creiamo due nuovi problemi - uno con la variabile fissata a 0, l'altro a 1
               # Modello con variabile = 0
               model_with_var_0 = current_model.clone()
               var_indices = [int(c) for c in var_name.split('_')[1:]]
               i, j = var_indices
               x_var_0 = model_with_var_0.get_var_by_name(var_name)
               model_with_var_0.add_constraint(x_var_0 == 0)
               
               # Modello con variabile = 1
               model_with_var_1 = current_model.clone()
               x_var_1 = model_with_var_1.get_var_by_name(var_name)
               model_with_var_1.add_constraint(x_var_1 == 1)
               
               # Risolviamo entrambi i modelli e li aggiungiamo alla coda
               try:
                   # Creiamo un dizionario delle variabili x per il modello con var = 0
                   x_vars_0 = {}
                   for var in model_with_var_0.iter_continuous_vars():
                       if var.name.startswith('x_'):
                           indices = var.name.split('_')[1:]
                           if len(indices) == 2:
                               i, j = int(indices[0]), int(indices[1])
                               x_vars_0[(i, j)] = var
                   
                   # Usiamo addCuts con il dizionario delle variabili
                   tsp_model_0, sol_0, edges_0 = addCuts(x_vars_0, [], model_with_var_0, nodes, p)
                   
                   if sol_0:
                       print(f"Ramo con {var_name}=0: obiettivo {sol_0.objective_value}")
                       new_decisions = branching_decisions + [(var_name, 0)]
                       problem_queue.append((model_with_var_0, sol_0, new_decisions))
               except Exception as e:
                   print(f"Errore nella risoluzione con {var_name}=0: {e}")
               
               try:
                   # Creiamo un dizionario delle variabili x per il modello con var = 1
                   x_vars_1 = {}
                   for var in model_with_var_1.iter_continuous_vars():
                       if var.name.startswith('x_'):
                           indices = var.name.split('_')[1:]
                           if len(indices) == 2:
                               i, j = int(indices[0]), int(indices[1])
                               x_vars_1[(i, j)] = var
                   
                   # Usiamo addCuts con il dizionario delle variabili
                   tsp_model_1, sol_1, edges_1 = addCuts(x_vars_1, [], model_with_var_1, nodes, p)
                   
                   if sol_1:
                       print(f"Ramo con {var_name}=1: obiettivo {sol_1.objective_value}")
                       new_decisions = branching_decisions + [(var_name, 1)]
                       problem_queue.append((model_with_var_1, sol_1, new_decisions))
               except Exception as e:
                   print(f"Errore nella risoluzione con {var_name}=1: {e}")
               
           except Exception as e:
               print(f"Errore: {e}")
       
       print("\n--- Fine Branch and Bound ---")
       print(f"Nodi esplorati: {nodes_explored}")
       
       if best_integer_solution:
           final_model, final_sol, final_edges = best_integer_solution
           print(f"Miglior soluzione intera trovata con obiettivo: {final_sol.objective_value}")
           
           # Creiamo un dizionario con le capacità per la visualizzazione
           final_cap = {}
           for e in final_edges:
               var_name = f"x_{e[0]}_{e[1]}"
               final_cap[e] = final_model.get_var_by_name(var_name).solution_value
           
           # Visualizziamo la soluzione
           p = np.array([[var.solution_value for var in final_model.iter_continuous_vars() if "x_" in var.name]])
           nodes = list(range(1, int(np.sqrt(len(p[0]))) + 1))
           
           return final_sol, final_edges, final_cap
       else:
           print("Nessuna soluzione intera trovata")
           return None, None, None
   
   # Creiamo un dizionario dei costi
  
   # Eseguiamo il Branch and Bound
   best_sol, best_edges, best_cap = branch_and_bound(tsp_model, sol,nodes, p)
   time.sleep(2000)
   
   if best_edges:
       # Visualizziamo la soluzione finale
       plot(nodes, best_edges, p, best_sol.objective_value, best_cap)
       time.sleep(2000)


