# -*- coding: utf-8 -*-
"""
OSR – Implementación completa de los métodos de la Tabla 4
===========================================================

Contiene TODOS los métodos comparados en la Tabla 4 del artículo:

 GNG

Incluye:
 - carga dataset
 - splits EXACTOS del artículo (Fig. 4)
 - tuning en validación open
 - evaluación micro-F
"""
# =====================================================================
# IMPORTS
# =====================================================================

import numpy as np
import pandas as pd
import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import re
import GNG_UTILS_OSR_2025 as GNG_Funciones
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score,accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import osr_utils as osr
from osr_utils import OSRMetrics

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')#LINUX
#matplotlib.use('TkAgg')#WINDOWS

colores_clases=['r','b','yellow','green','m','c']
colors_trasladados = ['turquoise', 'chocolate', 'blue', 'palevioletred', 'lightpink']
colors_SINGULARES = ['green', 'lightcoral', 'cyan', 'crimson','mediumslateblue','turquoise', 'chocolate', 'blue', 'palevioletred', 'lightpink']
colors_TEST = ['deepskyblue', 'orange', 'slategray', 'fuchsia', 'darkviolet']
coordenadas=['X','Y','Z','T','E','I','J','K','H','W']

TOLERANCE_ZONA=0.15
TOLERANCE_ENTROPIA_CAMBIO=0.75
tolerancia_coincidencia_punto=0.001
numero_vecinos_ENTROPIA=5
MAX_ITERATIONS = 300
INITIALIZE_CLUSTERS = 'k-means++'
CONVERGENCE_TOLERANCE = 0.0001
SILHOUTE_TOLERANCE = 0.11#0.015
LIMITE_S=0.27
tolerancia_coincidencia_cluster=0.01
TOL=0.07
ref_num=3
metricaD='1_NN'

including_percentage=0.7

tamaño_dataset=67000
NUM_THREADS = 8

import settings_osr
settings_osr.init()
path=settings_osr.path

import os
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "DATA")
RESULTS_DIR = os.path.join(BASE_DIR, "resultsGNG")

DEFAULT_LETTER_DATA     = os.path.join(DATA_DIR, "letter_dataset.csv")
DEFAULT_PENDIGITS_DATA  = os.path.join(DATA_DIR, "pendigits_dataset.csv")
DEFAULT_LETTER_LOG      = os.path.join(DATA_DIR, "letter_Partitions00.log")
DEFAULT_PENDIGITS_LOG   = os.path.join(DATA_DIR, "pendigits_Partitions00.log")


# ===============================================================
#  NORMALIZACIÓN DE IDs → etiquetas reales del dataset
# ===============================================================
def normalize_known_for_dataset(dataset_name: str, known_ids, y_all):
    """
    Convierte los IDs '1..26' o '0..9' del log a las etiquetas REALES del dataset.

    LETTER:
        - Si las etiquetas son 'A'..'Z', mapea 1→A, 2→B...
        - Si las etiquetas son números, devuelve números tal cual.
    PENDIGITS:
        - Devuelve strings numéricos.
    """
    uniq = np.unique(y_all)

    if dataset_name.lower() == "letter":
        # Si etiquetas son letras
        if all((len(u) == 1 and u.isalpha()) for u in uniq):
            mapped = []
            for k in known_ids:
                v = int(k)
                mapped.append(chr(64 + v))  # 1->A
            return mapped
        else:
            return [str(int(k)) for k in known_ids]

    else:  # pendigits
        return [str(int(k)) for k in known_ids]
# ===============================================================
#  LOAD DATA (CSV/TXT con separador desconocido)
# ===============================================================
def load_txt_any(path: str):
    """
    Carga robusta que:
    - Infiera separador (coma, espacios, tabulación, ;)
    - Infiera si la etiqueta está en la PRIMERA o ÚLTIMA columna
    Devuelve:
    X → float
    y → string
    """
    df = pd.read_csv(path, header=None, sep=None, engine='python')
    arr = df.values
    first, last = arr[:,0], arr[:,-1]

    def is_numeric(col):
        try:
            col.astype(float)
            return True
        except:
            return False

    first_num = is_numeric(first)
    last_num  = is_numeric(last)

    if not first_num and last_num:
        # etiqueta en primera columna
        y = first.astype(str)
        X = arr[:,1:].astype(float)
    elif first_num and not last_num:
        # etiqueta en última columna
        y = last.astype(str)
        X = arr[:,:-1].astype(float)
    elif not first_num and not last_num:
        # ambas no-numéricas → asumimos primera como etiqueta
        y = first.astype(str)
        X = arr[:,1:].astype(float)
    else:
        # ambas numéricas → asumimos última como etiqueta
        y = last.astype(str)
        X = arr[:,:-1].astype(float)

    return X, y
def load_csv_dataset(path):
    df = pd.read_csv(path, header=None)
    arr = df.values
    first, last = arr[:,0], arr[:,-1]

    def numeric(a):
        try:
            _ = a.astype(float)
            return True
        except:
            return False

    first_num = numeric(first)
    last_num  = numeric(last)

    # Heurística dataset UCI
    if not first_num and last_num:
        y = first.astype(str); X = arr[:,1:].astype(float)
    elif not last_num and first_num:
        y = last.astype(str); X = arr[:,:-1].astype(float)
    elif not first_num and not last_num:
        y = first.astype(str); X = arr[:,1:].astype(float)
    else:
        y = last.astype(str); X = arr[:,:-1].astype(float)

    return X, y
# ===============================================================
#  PARSEAR PARTICIONES: "known_class = [ ... ]"
# ===============================================================
KNOWN_RE = re.compile(r"known_class\s*=\s*\[(.*?)\]")
def parse_known_lists(log_path: str, expected_runs: int = 5):
    """
    Lee un fichero .log con líneas del tipo:
        known_class = [2,5,16,7,4,...];
    Devuelve una lista de listas con cada partición.
    """
    out = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = KNOWN_RE.search(line)
            if m:
                inside = m.group(1)
                parts = [p.strip().replace(";", "") for p in inside.split(",") if p.strip()]
                out.append(parts)

    if len(out) < expected_runs:
        raise ValueError(f"El log {log_path} tiene {len(out)} particiones; se esperaban {expected_runs}")

    return out[:expected_runs]
def read_partition_logold(log_path):
    partitions = {}
    current_openness = None

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            # detectar openness
            if "Openness" in line:
                val = re.findall(r"[\d.]+", line)[0]
                current_openness = float(val)

                partitions[current_openness] = []

            # detectar runs
            elif line.startswith("Run"):

                ids = re.findall(r"\d+", line)[1:]
                ids = [int(i) for i in ids]

                partitions[current_openness].append(ids)

    return partitions
def read_partition_log(log_path):
    known_lists = []
    with open(log_path, "r") as f:
        for line in f:
            if "known_class" in line:
                nums = re.findall(r"\d+", line.split("known_class")[1])
                nums = [int(x) for x in nums]
                known_lists.append(nums)
    return known_lists
# =====================================================================
# UTILIDADES BÁSICAS
# =====================================================================
def apply_pca_pipeline(X_train, X_val,X_val_C, X_test, variance_threshold=0.95):
    """
    Aplica escalado estándar y PCA manteniendo un porcentaje de la varianza.
    variance_threshold=0.95 significa que mantiene los componentes que
    explican el 95% de la varianza total.
    """
    # 1. Escalado (Obligatorio antes de PCA)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_val_C_s = scaler.transform(X_val_C)
    X_test_s = scaler.transform(X_test)

    # 2. Ajuste de PCA
    # Usamos n_components=variance_threshold para que PCA decida
    # cuántas dimensiones mantener automáticamente.
    pca = PCA(n_components=variance_threshold)
    X_train_pca = pca.fit_transform(X_train_s)
    X_val_pca = pca.transform(X_val_s)
    X_val_C_pca = pca.transform(X_val_C_s)
    X_test_pca = pca.transform(X_test_s)

    print(f"Dimensiones originales: {X_train.shape[1]}")
    print(f"Dimensiones reducidas con PCA: {X_train_pca.shape[1]}")

    return X_train_pca, X_val_pca, X_val_C_pca,X_test_pca
def standardize_fit(X):
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True)
    sigma[sigma == 0] = 1.0
    return (X - mu)/sigma, mu, sigma

def standardize_apply(X, mu, sigma):
    return (X - mu)/sigma

def l2_normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1
    return X / n

# =====================================================================
# SPLITS DEL ARTÍCULO
# =====================================================================
@dataclass
class OSRSplits:
    X_fit: np.ndarray
    y_fit: np.ndarray
    X_val_closed: np.ndarray
    y_val_closed: np.ndarray
    X_val_open: np.ndarray
    y_val_open: np.ndarray
    X_te: np.ndarray
    y_te: np.ndarray
    kkc_labels_fit: List[str]
    kkc_labels_all: List[str]
    # 60% de TODAS las KKCs: reentrenamiento final tras el tuning
    X_tr: np.ndarray
    y_tr: np.ndarray

def create_splits_explicit(X, y, known_labels):
    """
    Recibe: lista explícita de clases conocidas
    unknown_classes = todas las clases que NO están en known_labels
    Follows the same protocol as Figure 4 (Fitting/Closed/Open/Test)
    """

    known_labels = [str(k) for k in known_labels]  # Asegurar strings

    all_labels = np.unique(y).tolist()
    unknown_labels = [c for c in all_labels if c not in known_labels]

    # Dividir dataset en KKCs y UUCs
    idx_kkc = np.isin(y, known_labels)
    X_kkc, y_kkc = X[idx_kkc], y[idx_kkc]
    X_uuc, y_uuc = X[~idx_kkc], y[~idx_kkc]

    # 60% train por clase conocida
    X_tr = []
    y_tr = []
    X_te = []
    y_te = []
    rng = np.random.default_rng(123)

    for c in known_labels:
        idx = np.where(y_kkc == c)[0]
        idx = rng.permutation(idx)

        ntrain = math.ceil(0.6 * len(idx))
        X_tr.append(X_kkc[idx[:ntrain]])
        y_tr.append(y_kkc[idx[:ntrain]])

        X_te.append(X_kkc[idx[ntrain:]])
        y_te.append(y_kkc[idx[ntrain:]])

    X_tr = np.vstack(X_tr)
    y_tr = np.concatenate(y_tr)

    # Test = UUC completos + 40% KKCs
    X_te = np.vstack([np.vstack(X_te), X_uuc.reshape(-1, X.shape[1])])
    y_te = np.concatenate([np.concatenate(y_te), np.array(["UUC"] * len(X_uuc))])

    # 2/3 de las KKCs (Fitting)
    n_fit = int(2 / 3 * len(known_labels))
    kkc_fit = known_labels[:n_fit]  # sin aleatoriedad
    sim_uuc = known_labels[n_fit:]

    idx_fit = np.isin(y_tr, kkc_fit)
    Xtf, ytf = X_tr[idx_fit], y_tr[idx_fit]
    Xtu, ytu = X_tr[~idx_fit], y_tr[~idx_fit]

    # 60% fitting, 40% closed-val
    X_fit = []
    y_fit = []
    X_valc = []
    y_valc = []

    for c in kkc_fit:
        idx = np.where(ytf == c)[0]
        idx = rng.permutation(idx)
        nf = math.ceil(0.6 * len(idx))

        X_fit.append(Xtf[idx[:nf]])
        y_fit.append(ytf[idx[:nf]])
        X_valc.append(Xtf[idx[nf:]])
        y_valc.append(ytf[idx[nf:]])

    X_fit = np.vstack(X_fit)
    y_fit = np.concatenate(y_fit)

    X_valc = np.vstack(X_valc)
    y_valc = np.concatenate(y_valc)

    # Open-Val = closed-val + samples de las UUC simuladas
    X_valo = np.vstack([X_valc, Xtu])
    y_valo = np.concatenate([y_valc, np.array(["UUC"] * len(Xtu))])

    return OSRSplits(
        X_fit, y_fit,
        X_valc, y_valc,
        X_valo, y_valo,
        X_te, y_te,
        kkc_fit, known_labels,
        X_tr, y_tr
    )
# =====================================================================
# MAIN EXPERIMENTO PARA UN MÉTODO
# =====================================================================

def evaluate_full_pipeline(flag,params, numero_clusters,DIM,train, y_train, Xval, yval,kkc_fit,min_distance,
                           openess, metrica, scaler, epochs=50):
    """
    Ejecuta todo tu flujo: GNG -> KMeans -> Densidad -> Predicción -> Métricas
    Retorna el F1 Macro para que el optimizador sepa si los parámetros fueron buenos.
    """
    # Crea una instancia de la clase de métricas
    metrics = OSRMetrics()
    # --- PASO 1: GNG con los parámetros a probar ---
    model = GNG_Funciones.create_gng(
        n_i=DIM,
        step_gng=params['step'],
        max=params['max_nodes'],
        max_age=params['max_age'],
        n_start=2,
        min_distance=min_distance
    )
    tmp_node, model = GNG_Funciones.train_gng(train, model, epochs=epochs)

    labelsknn_tmp = GNG_Funciones.funcion_comparativaREAL(numero_clusters, train, y_train,
                                                          np.array(tmp_node).reshape(np.array(tmp_node).shape[0],
                                                                                     DIM), metrica='KNN5_GNG')
    nodos_temporales = np.array(tmp_node).reshape(np.array(tmp_node).shape[0], DIM)

    lista_ordenadostmp, indices_ordenados_tmp, etiquetas, prop = GNG_Funciones.filtrarGNGREALES(nodos_temporales,
                                                                                                labelsknn_tmp,
                                                                                                numero_clusters)
    array_posicionestmp, Array_etiquetastmp = GNG_Funciones.from_list_t_arrayREAL(lista_ordenadostmp, DIM,
                                                                                  etiquetas,
                                                                                  numero_clusters)

    # Al final, generas las predicciones sobre Xval (simulando el test)
    labels_DEFINITIVA, umbral_dist,unbral_conf = GNG_Funciones.funcion_validar_scoresREAL(
        Xfeatures=array_posicionestmp,
        Xlabels=Array_etiquetastmp,
        testX=Xval,  # <--- Validamos sobre Xval
        testY=yval,
        metrica=metrica,
        kkc=kkc_fit,
        openess=openess,
        scaler=scaler,
        XTRAIN=train,
        YTRAIN=y_train
    )
    if flag == 0:
        labels_DEFINITIVA = [int(x) for x in labels_DEFINITIVA]
        y_pred = GNG_Funciones.numeric_to_target(labels_DEFINITIVA)
        yvaltalpha = GNG_Funciones.numeric_to_target(yval)
    else:
        labels_DEFINITIVA = [int(x) for x in labels_DEFINITIVA]
        y_pred = ['UUC' if x == 100 else str(x) for x in labels_DEFINITIVA]
        yvaltalpha = ['UUC' if x == 100 else str(x) for x in yval]

    # Calculamos el F1 Macro sobre Xval
    F = metrics.micro_f1(yvaltalpha, y_pred, kkc_fit)
    print('scores',F)
    return F,umbral_dist,unbral_conf
def run_method(method, splits,flag,fichero,openess,metrica=metricaD,epochs=50):
    metrics = OSRMetrics()
    Xf, yf = splits.X_fit, splits.y_fit
    Xvc, yvc = splits.X_val_closed, splits.y_val_closed
    Xvo, yvo = splits.X_val_open, splits.y_val_open
    Xte, yte = splits.X_te, splits.y_te
    kkc_fit = splits.kkc_labels_fit
    kkc_all = splits.kkc_labels_all

    if method=='GNG':
        lista_numero_clusters=[]
        lista_etiquetas_test = []
        DIMENSION = Xf.shape[1]
        vol_batcht=Xf.shape[0]
        vol_batchv = Xvo.shape[0]#CAMBIO JULIO2026 Xvo
        vol_batche = Xte.shape[0]
        print(vol_batcht,vol_batchv,vol_batche)
        dim_max=1
        POSICIONESGNG = []
        labelsGNG = []
        F1 = np.full(dim_max, -1.1)
        array_resultados = np.full(dim_max, -1.1)
        array_resultados_acumulados = np.full(dim_max, -1.1)
        array_error = np.full(dim_max, -1.1)
        array_error_acumulado = np.full(dim_max, -1.1)

        mytrain = Xf
        # Un único scaler, ajustado sobre el fitting, para validación y predicción:
        # así los umbrales de distancia se aplican en el mismo espacio.
        scaler = StandardScaler().fit(Xf)
        if flag==0:
            my_y_train= GNG_Funciones.target_to_numeric(yf)
            clases_train = np.unique(my_y_train)
            my_y_train_full = GNG_Funciones.target_to_numeric(splits.y_tr)
            my_y_val = GNG_Funciones.target_to_numeric(yvo)
            clases_VAL = np.unique(my_y_val)
            my_y_test1 = GNG_Funciones.target_to_numeric(yte)
            my_y_test = [x if x in np.unique(my_y_train_full) else 100 for x in  my_y_test1]
            umbral_conf=0.5
            umbral_dist=1.6
        else:
            my_y_train = [ int(x) for x in yf]
            clases_train = np.unique(my_y_train)
            my_y_train_full = [int(x) for x in splits.y_tr]
            my_y_val =[100 if x == 'UUC' else int(x) for x in yvo]
            clases_VAL = np.unique(my_y_val)
            my_y_test1 =  [100 if x == 'UUC' else int(x) for x in yte]
            my_y_test = [x if x in np.unique(my_y_train_full) else 100 for x in  my_y_test1]
            umbral_conf = 0.8
            umbral_dist = 2.6

        numero_clases_ORIGINAL = len(np.unique(my_y_train))
        clases_train=np.unique(my_y_train)
        numero_clases_VAL= len(np.unique(yvo))
        numero_clasesTEST= len(np.unique(yte))

        print('\nclases TRAIN', numero_clases_ORIGINAL)

        Xval = Xvo
        yval = my_y_val
        numero_clases_ORIGINAL = len(np.unique(my_y_train))

        numero_clasestopredict = len(np.unique(yte))
        Xtest = Xte
        ytest = my_y_test

        print(0,numero_clasestopredict)
        limit = int((Xtest.shape[0] // 4) ** 0.5)+1
        scores = []
        scores.append(0.0)
        scores.append(0.1)
        for k in range(2, limit + 1):
            model = KMeans(n_clusters=k)
            model.fit(mytrain)
            pred = model.predict(mytrain)
            score = silhouette_score(mytrain, pred)
            scores.append(score)
        second_element = sorted(scores)[-2]
        index_second = np.where(np.array(scores) == second_element)[0][0]-1

        diferencia_clusteresNEW = scores[np.argmax(scores)] - second_element

        if diferencia_clusteresNEW<0.01:
            flag_OSR=1
            print(flag_OSR,np.argmax(scores), index_second)
        else:
            flag_OSR=0
            print(flag_OSR,np.argmax(scores), index_second)

        if np.argmax(scores)>numero_clases_ORIGINAL:
            if index_second!=1:
                numero_clusters = numero_clases_ORIGINAL
            else:
                numero_clusters=numero_clases_ORIGINAL
        else:
            numero_clusters =numero_clases_ORIGINAL

        lista_numero_clusters.append(numero_clusters)
        numero_nodos_GNG = 100*numero_clusters

        if openess>0:
            # --- BLOQUE DE OPTIMIZACIÓN ---
            param_candidates = [
                {'step': 0.1, 'max_nodes': 100000, 'max_age': 10},
                {'step': 0.2, 'max_nodes': 200000, 'max_age': 20},
                {'step': 0.3, 'max_nodes': 150000, 'max_age': 15},
                {'step': 0.2, 'max_nodes': 100*numero_nodos_GNG, 'max_age': 10},
            ]
            param_candidates = [
                {'step': 0.2, 'max_nodes': 100 * numero_nodos_GNG, 'max_age': 10},
            ]

            best_f1 = -1
            best_params = param_candidates[0]

            print("Buscando parámetros óptimos usando Xval...")
            for p in param_candidates:
                current_f1 ,u1,u2= evaluate_full_pipeline(flag,p,numero_clusters,DIMENSION, mytrain, my_y_train, Xval, yval, kkc_fit,min_distance=tolerancia_coincidencia_punto,
                                                          openess=openess, metrica=metrica, scaler=scaler, epochs=epochs)
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_params = p
                    umbral_dist, umbral_conf = u1, u2

            print(f"Mejores parámetros encontrados: {best_params} con F1: {best_f1}")
            if metrica == 'KDE_EVT':
                print(f"delta (umbral P_incl):  {umbral_dist}, h (ancho de banda):  {umbral_conf}")
            else:
                print(f"umbral distancia:  {umbral_dist}, umbral confianza:  {umbral_conf}")
        else:
            param_candidates = [
                {'step': 0.2, 'max_nodes': 100 * numero_nodos_GNG, 'max_age': 10},
            ]
            best_f1 = -1
            best_params = param_candidates[0]
            if metrica == 'KDE_EVT':
                # Apertura 0: no hay rechazo; delta sin efecto y h por defecto
                umbral_dist, umbral_conf = 0.0, 1.0

        # --- EJECUCIÓN FINAL ---
        # Reentrenamiento con el 60% de TODAS las KKCs (no solo kkc_fit)
        mytrain = splits.X_tr
        my_y_train = my_y_train_full
        clases_train = np.unique(my_y_train)
        numero_clusters = len(clases_train)
        # Ahora código original UNA VEZ usando best_params
        gng_slayer1 = GNG_Funciones.create_gng(
            n_i=DIMENSION,
            step_gng=best_params['step'],
            max=best_params['max_nodes'],
            max_age=best_params['max_age'],
            n_start=2,min_distance=tolerancia_coincidencia_punto )

        tmp_node, gng_slayer = GNG_Funciones.train_gng(mytrain, gng_slayer1, epochs=epochs)

        cent = []
        for i in clases_train:
            res = np.array(GNG_Funciones.centeroidpython(mytrain[np.array(my_y_train) == i]))
            cent.append(res)

        labelsknn_tmp = GNG_Funciones.funcion_comparativaREAL(numero_clusters,mytrain, my_y_train,
                                                          np.array(tmp_node).reshape(np.array(tmp_node).shape[0],
                                                                                     DIMENSION), metrica='KNN5_GNG')
        nodos_temporales = np.array(tmp_node).reshape(np.array(tmp_node).shape[0], DIMENSION)

        lista_ordenadostmp, indices_ordenados_tmp,etiquetas, prop = GNG_Funciones.filtrarGNGREALES(nodos_temporales, labelsknn_tmp,
                                                                                   numero_clusters)
        array_posicionestmp, Array_etiquetastmp = GNG_Funciones.from_list_t_arrayREAL(lista_ordenadostmp, DIMENSION,etiquetas,
                                                                                  numero_clusters)

        POSICIONESGNG.append(GNG_Funciones.flatten(lista_ordenadostmp))
        labelsGNG.append(Array_etiquetastmp)

        labels_DEFINITIVA,scores_DEF = GNG_Funciones.funcion_predecir_scoresREAL(Xfeatures=array_posicionestmp, Xlabels=Array_etiquetastmp,
                                                           test=Xtest,metrica=metrica,openess=openess,umbralDIST=umbral_dist,umbralCONF=umbral_conf,
                                                           scaler=scaler,XTRAIN=mytrain,YTRAIN=my_y_train)

        cent_tr, cent_tst = GNG_Funciones.obtencion_ceniodes(numero_clusters, DIMENSION, mytrain, my_y_train, Xtest,
                                                             np.array(labels_DEFINITIVA))
        distancia_train_test_INICIAL = GNG_Funciones.calculo_distancia_centroides(cent_tr, cent_tst, numero_clusters)
        distancia_train_test_OROGINARIA = distancia_train_test_INICIAL
        distancia_limite = (np.max(distancia_train_test_OROGINARIA)) * ref_num
        print( '  distancia_train_test_OROGINARIA  ', distancia_train_test_OROGINARIA)
        print( '  distancia LIMITE  ', distancia_limite)

        if flag==1:
            resultado = [100 if x == 'UUC' else int(x) for x in ytest]
            F1[0] = (f1_score(resultado, labels_DEFINITIVA, average='macro'))
            lista_etiquetas_test.append(resultado)
            comparativa = resultado
        else:
            F1[0] = (f1_score(ytest, labels_DEFINITIVA, average='macro'))
            lista_etiquetas_test.append(ytest)
            comparativa = ytest

        resVERDADERA = np.abs(np.array(comparativa )- np.array(labels_DEFINITIVA))
        errorVERDADERA = len(np.where(resVERDADERA != 0)[0]) / len(resVERDADERA)
        totalVERDADERA = 100 * (1 - errorVERDADERA)
        array_resultados[0] = (1 - errorVERDADERA)
        array_error[0] = errorVERDADERA
        acumulado = GNG_Funciones.calcularAcumuladonew(0, array_resultados, array_resultados[0])
        array_resultados_acumulados[0] = acumulado
        array_error_acumulado[0] = GNG_Funciones.calcular_ERROR_Acumuladonew(0, array_error,array_error[0])

        print('F1 ', F1[0])
        print(0, '  RESULTADOS:  ', totalVERDADERA)

        if flag==0:
            labels_DEFINITIVA = [int(x) for x in labels_DEFINITIVA]
            y_pred =GNG_Funciones.numeric_to_target(labels_DEFINITIVA)
            ytestalpha=GNG_Funciones.numeric_to_target(ytest)
        else:
            labels_DEFINITIVA = [int(x) for x in labels_DEFINITIVA]
            y_pred = ['UUC' if x == 100 else str(x) for x in labels_DEFINITIVA]
            ytestalpha = ['UUC' if x == 100 else str(x) for x in ytest]

        # métricas
        F1 = metrics.micro_f1(ytestalpha, y_pred, kkc_all)
        J1 = metrics.youden_index_osr(ytestalpha, y_pred, kkc_all)
        acc1=metrics.accuracy(ytestalpha, y_pred, kkc_all)
        auc_score1,fpr1, tpr1  = metrics.compute_auroc(scores_DEF, ytestalpha, kkc_all)
        print('ARTICULO micro F1 OSR ', 100.0 * F1)
        print("Accuracy:", acc1 * 100)
        print("Youden:", J1)
        print("AUROC:", auc_score1)

        # Sin UUC en test (apertura 0) no hay curva ROC
        if fpr1 is not None:
            title = "ROC Curve"
            plt.figure()
            plt.plot(fpr1, tpr1, label=f"AUC = {auc_score1:.3f}")
            plt.plot([0, 1], [0, 1], '--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(title)
            plt.legend()
            plt.savefig(fichero +str(0) +'_AUROC_.jpg')
            plt.close('all')

    else:
        raise ValueError("Método desconocido")


    return F1*100,acc1*100, J1,auc_score1

def parse_args():
    p = argparse.ArgumentParser(description="OSR con GNG: train / validación abierta / test")
    p.add_argument("--datasets", nargs="+", default=["PENDIGITS", "LETTER"],
                   choices=["PENDIGITS", "LETTER"])
    p.add_argument("--openness", nargs="+", default=["0"],
                   help="Aperturas a ejecutar (claves de los logs: 0, 8.71, 18.35, 15.48, 25.46) o 'all'")
    p.add_argument("--metricas", nargs="+", default=["1_NN", "KDE_EVT"],
                   choices=["1_NN", "KDE_EVT"],
                   help="Predictor OSR sobre los nodos GNG")
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--out-dir", default=RESULTS_DIR)
    p.add_argument("--epochs", type=int, default=50, help="Épocas de entrenamiento GNG")
    p.add_argument("--subsample", type=float, default=1.0,
                   help="Fracción estratificada del dataset (p.ej. 0.05 para prueba rápida)")
    p.add_argument("--max-runs", type=int, default=None, help="Máximo de particiones por apertura")
    p.add_argument("--smoke", action="store_true",
                   help="Prueba rápida: subsample 0.05, 2 épocas, 1 partición")
    return p.parse_args()


def subsample_estratificado(X, y, frac, seed=0):
    if frac >= 1.0:
        return X, y
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        ic = np.where(y == c)[0]
        n = max(10, int(round(frac * len(ic))))
        idx.append(rng.choice(ic, size=min(n, len(ic)), replace=False))
    idx = np.sort(np.concatenate(idx))
    return X[idx], y[idx]


if __name__ == "__main__":
    args = parse_args()
    if args.smoke:
        args.subsample = min(args.subsample, 0.05)
        args.epochs = min(args.epochs, 2)
        args.max_runs = 1

    # Ruta Datasets y sus logs de particiones (clave = apertura objetivo)
    DATASETS = {
        "PENDIGITS": {
            "data": os.path.join(args.data_dir, "pendigits_dataset.csv"),
            "logs": {
                0.0: os.path.join(args.data_dir, "pendigits_Partitions00.log"),
                8.71: os.path.join(args.data_dir, "pendigits_Partitions80.log"),
                18.35: os.path.join(args.data_dir, "pendigits_Partitions180.log")
            }
        },
        "LETTER": {
            "data": os.path.join(args.data_dir, "letter_dataset.csv"),
            "logs": {
                0.0: os.path.join(args.data_dir, "letter_Partitions00.log"),
                15.48: os.path.join(args.data_dir, "letter_Partitions150.log"),
                25.46: os.path.join(args.data_dir, "letter_Partitions250.log")
            }
        }
    }
    if "all" in [o.lower() for o in args.openness]:
        openness_sel = None
    else:
        openness_sel = [float(o) for o in args.openness]

    METHODS = ["GNG"]
    OUT_CSV = os.path.join(args.out_dir, "resultados_GNG_Tr_Val_Test_loop2.csv")
    PATH_AUROC = os.path.join(args.out_dir, "AUROC") + os.sep
    os.makedirs(PATH_AUROC, exist_ok=True)
    print('RESULTADOS ', OUT_CSV)
    print('ARGUMENTOS ', vars(args))
    # resultados
    rows = []
    # datasets
    for ds_name in args.datasets:
        cfg = DATASETS[ds_name]
        print(f"\n=== DATASET: {ds_name} ===")

        X, y = load_txt_any(cfg["data"])
        X, y = subsample_estratificado(X, y, args.subsample)

        # loop openness
        for openness_target, log_path in cfg["logs"].items():
            if openness_sel is not None and not any(abs(openness_target - o) < 1e-6 for o in openness_sel):
                continue
            print("\nTarget openness:", openness_target)
            known_lists = read_partition_log(log_path)
            if args.max_runs is not None:
                known_lists = known_lists[:args.max_runs]

            # particiones (runs)
            for run_idx, known_ids in enumerate(known_lists, start=1):
                # 5.1 Mapeo de IDs del log a etiquetas del dataset
                known_labels = normalize_known_for_dataset(ds_name.lower(), known_ids, y_all=y)

                # clases desconocidas
                all_labels = np.unique(y)
                unknown_labels = [c for c in all_labels if c not in known_labels]
                # |C_TR|
                c_tr = len(known_labels)
                # |C_TE|
                c_te =  len(all_labels)
                # openness
                O = 100 * osr.openness(c_tr, c_te)
                print(f"Run {run_idx}: |C_TR|={c_tr}, |C_TE|={c_te}, Openness={O:.2f}%")

                # generar splits
                splits = create_splits_explicit(X, y, known_labels)

                if ds_name=='LETTER':
                    flag_datset=0
                else:
                    flag_datset=1

                # métodos
                for method in METHODS:
                  for metrica in args.metricas:
                    print(f"    > {method} [{metrica}] ...", end="", flush=True)
                    name_file_IMG=str(ds_name)+str(run_idx)+str(((int)(openness_target)))+'_'+metrica
                    print('FICHERO  ',PATH_AUROC+name_file_IMG)
                    #run_one_method
                    microF,acc,youden,auroc = run_method(method, splits,flag_datset,PATH_AUROC+name_file_IMG,O,
                                                         metrica=metrica, epochs=args.epochs)
                    rows.append({
                        "dataset": ds_name,
                        "run": run_idx,
                        "method": method,
                        "metrica": metrica,
                        "microF": microF,
                        "accosr": acc,
                        "youden": youden,
                        "AUROC":  auroc,
                        "openness": O,
                        "known_classes": ",".join(splits.kkc_labels_all)
                    })
                    # Se guarda tras cada ejecución para no perder resultados en ejecuciones largas
                    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    # CSV consolidado y mostrar resumen
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nResultados guardados en: {OUT_CSV}")

    # Resumen media±std por dataset, método y predictor
    summary = df.groupby(["dataset", "openness", "method", "metrica"])[["microF", "youden", "AUROC"]].agg(["mean", "std"])
    print("\n=== RESUMEN ===")
    print(summary.to_string(float_format=lambda v: f"{v:.2f}"))
    print('FIN')



