# Algoritmi di Ottimizzazione per TSP e Flusso Massimo

Questo repository contiene implementazioni di algoritmi di ottimizzazione per due problemi classici di ricerca operativa:
1. **Problema del Commesso Viaggiatore (TSP)** con approccio basato su cutting planes e branch-and-bound
2. **Problema del Flusso Massimo** utilizzando programmazione lineare

## Requisiti

- Python 3.6+
- docplex (IBM Decision Optimization CPLEX Modeling)
- Matplotlib
- NumPy
- Python 3.11 (default) o inferiore

Per installare le dipendenze:

```bash
pip install docplex matplotlib numpy
```

## Problema del Commesso Viaggiatore (`tsp_with_maxflow.py`)

Questo script implementa una soluzione per il problema del commesso viaggiatore (TSP) utilizzando un approccio ibrido:

1. **Formulazione iniziale**: Modello di base con vincoli di grado (ogni nodo ha esattamente un arco entrante e uno uscente)
2. **Cutting Planes**: Utilizzo dell'algoritmo di Ford-Fulkerson per identificare i tagli di capacità e aggiungere vincoli che eliminano i sottocicli
3. **Branch and Bound**: Tecnica esatta per trovare soluzioni intere partendo dalle variabili frazionarie

4. **Enumerazione vincoli 4** Possibilita' di risolverlo enumerando tutti i vincoli (4) con la formulazione dei Sotto insiemi (cut-set)


### Caratteristiche principali

- Implementazione completa di un metodo basato su cutting planes per TSP
- Algoritmo di Branch and Bound per trovare soluzioni intere
- Visualizzazione degli archi con frecce direzionali per mostrare il percorso
- Sistema di logging e salvataggio di risultati intermedi
- Generazione di istanze casuali per il problema TSP

### Utilizzo

```bash
python tsp_with_maxflow.py
```

Il programma creerà una cartella `output/run_[timestamp]` che conterrà:
- File di log dettagliato dell'esecuzione
- Rappresentazione grafica delle soluzioni intermedie e finali
- File JSON con l'istanza generata e i risultati

## Problema del Flusso Massimo (`maxFlow_pl.py`)

Implementazione del problema del flusso massimo utilizzando programmazione lineare tramite il package docplex.

### Caratteristiche

- Formulazione del problema di flusso massimo come problema di programmazione lineare
- Vincoli di conservazione del flusso in ciascun nodo
- Calcolo del flusso massimo tra un nodo sorgente e un nodo destinazione

### Utilizzo

```bash
python maxFlow_pl.py
```

L'output mostrerà:
- Il valore del flusso massimo
- Il flusso su ciascun arco della soluzione

## Struttura dell'output

La directory `output` contiene sottocartelle per ogni esecuzione, identificate da timestamp. Ogni sottocartella contiene:

- `instance.json`: Descrizione dell'istanza TSP (nodi, coordinate, costi)
- `results.json`: Risultati dell'esecuzione (valore obiettivo, archi, tempo di soluzione)
- `solution_plot.png`: Visualizzazione della soluzione rilassata
- `solution_final_plot.png`: Visualizzazione della soluzione finale dopo Branch and Bound
- `TSP_*.png`: Visualizzazioni delle soluzioni intermedie durante l'aggiunta di cutting planes
- `B&B_*.png`: Visualizzazioni dei rami esplorati durante il Branch and Bound
- `tsp_execution.log`: Log dettagliato dell'esecuzione

## Licenza

[MIT](https://choosealicense.com/licenses/mit/)


