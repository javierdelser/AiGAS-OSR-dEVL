# -*- coding: utf-8 -*-
"""
 - 1-vs-Set            (Scheirer et al. 2013)
 - W-SVM + W-OSVM CAP  (Scheirer et al. 2014)
 - PI-SVM              (Jain et al. 2014)
 - SROSR               (Zhang & Patel 2017)
 - OSNN                (Junior et al. 2017)
 - EVM                 (Rudd et al. 2018)
"""
# =====================================================================
# IMPORTS
# =====================================================================
import sys
import os
os.environ['THEANO_FLAGS'] = 'blas__ldflags='
os.environ['THEANO_FLAGS'] = 'blas.ldflags='

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger("theano").setLevel(logging.ERROR)
logging.getLogger("theano.tensor.blas").setLevel(logging.ERROR)

import json
import numpy as np
import pandas as pd
import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import re
import sklearn.metrics.pairwise
from sklearn.svm import SVC,OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from multiprocessing import Pool,cpu_count
from itertools import product
from libsvm.svmutil import svm_train, svm_predict
from contextlib import contextmanager
from sklearn.mixture import BayesianGaussianMixture
from scipy.stats import genpareto, weibull_min
from scipy.special import expit
import time
try:
    import libmr  # solo lo usa EVM; requiere compilador C++ en Windows
except ImportError:
    libmr = None
import itertools as it
from sklearn.metrics import f1_score,accuracy_score
from sklearn.metrics import roc_curve, auc
import settings_osr
settings_osr.init()
path=settings_osr.path

class SilenceStderr:
    def __enter__(self):
        self.ins = os.dup(sys.stderr.fileno())
        self.out = open(os.devnull, 'w')
        os.dup2(self.out.fileno(), sys.stderr.fileno())
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        os.dup2(self.ins, sys.stderr.fileno())
        os.close(self.ins)
        self.out.close()

class OSRMetrics:
    @staticmethod
    def _unify_types(y_true, y_pred, known_labels):
        """
        Normalización inteligente y selectiva:
        - Si el valor es numérico (ej: '0.', '0.0', 0, 0.0), lo convierte a string entero ('0').
        - Si es un string no numérico (ej: 'UUC', 'A', 'B'), lo deja como string limpio.
        """

        def normalize_value(x):
            if x is None: return "None"
            s = str(x).strip()
            if s == "": return ""

            # Intentamos ver si es un número
            try:
                # Si puede convertirse a float, comprobamos si es equivalente a un entero
                val_f = float(s)
                if val_f == int(val_f):
                    return str(int(val_f))  # '0.0' -> 0 -> '0'
                return str(val_f)  # Para casos como 0.5 que no son enteros
            except (ValueError, TypeError):
                # No es un número (ej: 'UUC', 'A', 'B'), devolvemos el string limpio
                return s

        # Aplicamos la normalización elemento a elemento
        y_true_s = np.array([normalize_value(x) for x in np.array(y_true).flatten()])
        y_pred_s = np.array([normalize_value(x) for x in np.array(y_pred).flatten()])
        known_labels_s = np.array([normalize_value(x) for x in np.array(known_labels).flatten()])

        return y_true_s, y_pred_s, known_labels_s

    @classmethod
    def micro_f1(cls, y_true, y_pred, known_labels):
        y_true_s, y_pred_s, known_labels_s = cls._unify_types(y_true, y_pred, known_labels)

        # Muestra es conocida
        mask_known = np.isin(y_true_s, known_labels_s)
        # Muestra es desconocida
        mask_unknown = ~mask_known

        # TP: Es conocida Y predijo la clase correcta
        TP = np.sum((y_true_s == y_pred_s) & mask_known)
        # FN: Es conocida PERO predijo otra cosa (incluido UUC o cualquier label fuera de known)
        FN = np.sum((y_true_s != y_pred_s) & mask_known)
        # FP: Es desconocida PERO predijo CUALQUIERA de las clases conocidas
        FP = np.sum(np.isin(y_pred_s, known_labels_s) & mask_unknown)

        denominator = (2 * TP + FP + FN)
        f1_debug = (2 * TP) / denominator if denominator > 0 else 0.0
        #print(f"DEBUG FINAL -> TP: {TP}, FN: {FN}, FP: {FP}, F1: {f1_debug}")

        return (2 * TP) / denominator if denominator > 0 else 0.0

    @classmethod
    def accuracy(cls, y_true, y_pred, known_labels):
        y_true_s, y_pred_s, known_labels_s = cls._unify_types(y_true, y_pred, known_labels)

        mask_known = np.isin(y_true_s, known_labels_s)

        # Aciertos en conocidas: Coincidencia exacta
        correct_known = np.sum((y_true_s == y_pred_s) & mask_known)
        # Aciertos en desconocidas: Realmente es desconocida Y la predicción NO es ninguna conocida
        correct_unknown = np.sum((~np.isin(y_pred_s, known_labels_s)) & (~mask_known))

        return (correct_known + correct_unknown) / len(y_true) if len(y_true) > 0 else 0.0

    @classmethod
    def youden_index_osr(cls, y_true, y_pred, known_labels):
        y_true_s, y_pred_s, known_labels_s = cls._unify_types(y_true, y_pred, known_labels)
        known_set = set(known_labels_s)

        TP = TN = FP = FN = 0
        for yt, yp in zip(y_true_s, y_pred_s):
            is_true_open = yt not in known_set
            is_pred_open = yp not in known_set

            if is_true_open and is_pred_open:
                TP += 1  # Correctamente identificado como desconocido
            elif not is_true_open and not is_pred_open:
                TN += 1  # Correctamente identificado como conocido
            elif not is_true_open and is_pred_open:
                FP += 1  # Falsa alarma: era conocido pero dijo que es nuevo
            elif is_true_open and not is_pred_open:
                FN += 1  # Fallo de detección: era nuevo pero dijo que es conocido
            '''if yt in known_set:
                if yp == yt:
                    TP += 1
                else:
                    FN += 1
            else:
                if yp not in known_set:
                    TN += 1
                else:
                    FP += 1'''
        # Sensibilidad (TPR): Proporción de clases nuevas detectadas
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        # Especificidad (TNR): Proporción de clases conocidas mantenidas como conocidas
        TNR = TN / (TN + FP) if (TN + FP) > 0 else 0
        return TPR + TNR - 1

    def youden_index_evl(cls, y_true, y_pred, known_labels):
        '''
        El problema técnico:
        El Índice de Youden (y la matriz de confusión) no están diseñados para conjuntos de datos donde una de las clases está ausente.
        Matemáticamente, la Sensibilidad es indeterminada ( 0/0 ) si no hay positivos reales.

        En tu investigación sobre Concept Drift (CD) y Extreme Verification Latency, esto es un problema porque el CD hace que las clases "se muevan", y el sistema debe ser capaz de decir "todo sigue siendo conocido" sin lanzar falsas alarmas, y eso es un éxito.
        Opción B: Penalización Neutra (La más pragmática)
        Si quieres un número para tus gráficas, puedes considerar que si no había nada que detectar y no hubo falsas alarmas, el modelo ha cumplido su función.

        Lógica: Si TP + FN == 0 y FP == 0, entonces considera la TPR=1.
        Resultado: 1(TPR)+1(TNR)−1=1.
        '''
        y_true_s, y_pred_s, known_labels_s = cls._unify_types(y_true, y_pred, known_labels)
        known_set = set(known_labels_s)

        TP = TN = FP = FN = 0
        for yt, yp in zip(y_true_s, y_pred_s):
            is_true_open = yt not in known_set
            is_pred_open = yp not in known_set

            if is_true_open and is_pred_open:
                TP += 1
            elif not is_true_open and not is_pred_open:
                TN += 1
            elif not is_true_open and is_pred_open:
                FP += 1
            elif is_true_open and not is_pred_open:
                FN += 1

        # --- LÓGICA ESTRICTA PARA OSR + CONCEPT DRIFT ---

        # 1. Sensibilidad (TPR)
        if (TP + FN) > 0:
            # Si hay clases nuevas, se mide la capacidad real de detectarlas
            TPR = TP / (TP + FN)
        else:
            # Si NO hay clases nuevas:
            # Si el modelo lanzó falsas alarmas (FP > 0), penalizamos la TPR
            # Si el modelo fue silencioso y correcto, TPR = 1 (estabilidad)
            TPR = 1.0 if FP == 0 else 0.0

        # 2. Especificidad (TNR)
        if (TN + FP) > 0:
            TNR = TN / (TN + FP)
        else:
            # Si no hay clases conocidas en el lote
            TNR = 1.0 if FN == 0 else 0.0

        return TPR + TNR - 1

    @classmethod
    def compute_auroc(cls, scores, y_true, known_labels):
        y_true_s, _, known_labels_s = cls._unify_types(y_true, [], known_labels)

        # Binarizar: 1 si es conocida, 0 si es desconocida
        y_bin = np.array([1 if y in known_labels_s else 0 for y in y_true_s])

        if len(np.unique(y_bin)) < 2:
            return np.nan, None, None

        fpr, tpr, thresholds = roc_curve(y_bin, scores)
        return auc(fpr, tpr), fpr, tpr

@contextmanager
def timer(message):
    """
    Simple timing method. Logging should be used instead for large scale experiments.
    """
    print( message)
    start = time.time()
    yield
    stop = time.time()
    print ("...elapsed time: {}".format(stop-start))

# =====================================================================
# UTILIDADES BÁSICAS
# =====================================================================

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
#  LOAD DATA
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

# =====================================================================
# MICRO-F METRICAS
# =====================================================================
#PROVIENEN DE OSR
'''
def f_measure_osrold(y_true, y_pred, known_labels):
    TP = 0
    FP = 0
    FN = 0

    for i in range(len(y_true)):
        if y_true[i] in known_labels:
            if y_pred[i] == y_true[i]:
                TP += 1
            else:
                FN += 1
        else:
            if y_pred[i] in known_labels:
                FP += 1

    if (2*TP + FP + FN) == 0:
        return 0.0

    return 2*TP / (2*TP + FP + FN)
def f_measure_osr11(y_true, y_pred, known_labels):
    # Convertimos a numpy arrays para asegurar operaciones vectorizadas
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    known_labels = np.array(known_labels)

    # Verificación de seguridad: deben tener el mismo tamaño
    if len(y_true) != len(y_pred):
        raise ValueError(f"Tamaños incompatibles: y_true({len(y_true)}) vs y_pred({len(y_pred)})")

    # TP: La etiqueta es conocida Y la predicción es correcta
    # np.isin comprueba si el elemento está en la lista de etiquetas conocidas
    mask_known = np.isin(y_true, known_labels)
    TP = np.sum((y_true == y_pred) & mask_known)

    # FN: La etiqueta es conocida PERO la predicción es incorrecta (incluye predecir UUC)
    FN = np.sum((y_true != y_pred) & mask_known)

    # FP: La etiqueta es desconocida (Open) PERO el modelo predijo una clase conocida
    mask_unknown = ~mask_known
    FP = np.sum(np.isin(y_pred, known_labels) & mask_unknown)

    denominator = (2 * TP + FP + FN)
    if denominator == 0:
        return 0.0

    return (2 * TP) / denominator
def f_measure_osr1(y_true, y_pred, known_labels):
    y_true_s = np.array(y_true).astype(str)
    y_pred_s = np.array(y_pred).astype(str)
    known_labels_s = np.array(known_labels).astype(str)

    mask_known = np.isin(y_true_s, known_labels_s)

    # TP: Es conocida Y la predicción coincide exactamente
    TP = np.sum((y_true_s == y_pred_s) & mask_known)

    # FN: Es conocida PERO la predicción es distinta (sea UUC o cualquier otra clase)
    FN = np.sum((y_true_s != y_pred_s) & mask_known)

    # FP: La muestra es desconocida PERO la predicción es CUALQUIERA de las conocidas
    mask_unknown = ~mask_known
    FP = np.sum(np.isin(y_pred_s, known_labels_s) & mask_unknown)

    denominator = (2 * TP + FP + FN)
    if denominator == 0: return 0.0
    return (2 * TP) / denominator
#PROVIENEN DE EVL
def compute_auroc2(scores, y_true, known_labels):
    # 1. UNIFICACIÓN: Convertimos todo a string para asegurar que el 'in' funcione
    # independientemente de si son floats, ints o strings.
    y_true_s = np.array(y_true).astype(str)
    known_labels_s = np.array(known_labels).astype(str)

    # 2. Crear el vector binario: 1 si es conocido, 0 si es desconocido (UUC)
    # Ahora la comparación 'y in known_labels_s' es segura porque ambos son strings.
    y_bin = np.array([1 if y in known_labels_s else 0 for y in y_true_s])

    # 3. Cálculo de AUROC
    # scores deben ser los scores de "confianza" (ej. el score de la clase ganadora)
    fpr, tpr, thresholds = roc_curve(y_bin, scores)
    roc_auc = auc(fpr, tpr)

    return roc_auc, fpr, tpr
def accuracy_osr2(y_true, y_pred, known_labels):
    # UNIFICACIÓN: Todo a string para evitar 0.0 != '0.0'
    y_true = np.array(y_true).astype(str)
    y_pred = np.array(y_pred).astype(str)
    known_labels = np.array(known_labels).astype(str)

    N = len(y_true)
    if N == 0: return 0.0

    # Máscara de muestras que son conocidas
    mask_known = np.isin(y_true, known_labels)

    # Aciertos en conocidas: es conocida Y la predicción coincide exactamente
    correct_known = np.sum((y_true == y_pred) & mask_known)

    # Aciertos en desconocidas: NO es conocida Y la predicción es "UUC"
    correct_unknown = np.sum((y_pred == "UUC") & (~mask_known))

    return (correct_known + correct_unknown) / N
def youden_index_osr2(y_true, y_pred, known_labels):
    y_true = np.array(y_true).astype(str)
    y_pred = np.array(y_pred).astype(str)
    known_labels = np.array(known_labels).astype(str)
    known_set = set(known_labels)

    TP = TN = FP = FN = 0
    for yt, yp in zip(y_true, y_pred):
        if yt in known_set:
            if yp == yt:
                TP += 1
            else:
                FN += 1
        else:
            # CAMBIO CLAVE: Si la predicción NO está en el set de conocidas,
            # se considera un acierto de detección de desconocido (TN)
            if yp not in known_set:
                TN += 1
            else:
                FP += 1

    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0
    return TPR + TNR - 1
def f_measure_osr2(y_true, y_pred, known_labels):
    TP = FP = FN = 0
    for c in known_labels:
        tp = np.sum((np.array(y_true) == c) & (np.array(y_pred) == c))
        fn = np.sum((np.array(y_true) == c) & (np.array(y_pred) != c))
        fp = np.sum((np.array(y_true) != c) & (np.array(y_pred) == c))

        TP += tp
        FN += fn
        FP += fp

    if TP == 0:
        return 0.0
    P = TP / (TP + FP)
    R = TP / (TP + FN)

    return 2 * P * R / (P + R)
'''
# ===============================================================
#  OPENNESS
# ===============================================================
def openness(c_tr, c_te):
    """
    O = 1 - sqrt( 2*|C_TR| / (|C_TR| + |C_TE|) )
    """
    return 1.0 - math.sqrt((2*c_tr) / (c_tr + c_te))
# =====================================================================
# EVT – AJUSTE WEIBULL M
# =====================================================================

@dataclass
class WeibullParams:
    k: float
    lam: float

def fit_weibull_mle(x, max_iter=50, tol=1e-6):
    x = np.asarray(x, float)
    x = x[x>0]
    if len(x)==0:
        return WeibullParams(1.0,1.0)

    logx = np.log(x)
    k = 1.2
    for _ in range(max_iter):
        xk = x**k
        xklogx = xk * logx
        sxk = np.sum(xk)
        sxklogx = np.sum(xklogx)
        f = (1/k) + np.mean(logx) - (sxklogx/(sxk+1e-12))
        sxklogx2 = np.sum(xk*(logx**2))
        denom = sxk*sxk + 1e-12
        fprime = (-1/(k*k)) - ((sxklogx2*sxk - sxklogx*sxklogx)/denom)
        step = f/(fprime+1e-12)
        k_new = k - step
        if not np.isfinite(k_new) or k_new<=0:
            k_new = max(1e-3, k*0.5)
        if abs(k_new-k)<tol:
            k = k_new
            break
        k = k_new
    lam = (np.mean(x**k))**(1/k)
    lam = max(lam,1e-6)
    return WeibullParams(k=float(k), lam=float(lam))

# =====================================================================
#  OVR_LIBSVM
# =====================================================================
class OVR_LIBSVM:
    def __init__(self, C=1.0, gamma=0.1):#gamma='scale'
        self.C = C
        self.gamma =  gamma
        self.models = {}
        self.labels = []

    def fit(self, X, y):
        self.labels = sorted(np.unique(y))

        for c in self.labels:
            y_bin = np.where(y == c, 1, -1)

            model = SVC(
                C=self.C,
                gamma=self.gamma,
                kernel='rbf',
                #kernel='poly',degree=3,
                probability=True,
                class_weight='balanced'
            )
            model.fit(X, y_bin)
            self.models[c] = model

        return self

    def scores(self, X):
        out = {}

        for c in self.labels:
            s = self.models[c].decision_function(X)

            # ESCALADO MIN-MAX: Forzamos los scores a estar entre 0 y 1
            # Esto mantiene la distribución pero la pone en el rango que Weibull ama
            s_min = np.min(s)
            s_max = np.max(s)
            out[c] = (s - s_min) / (s_max - s_min + 1e-8)

            #out[c] = s
            # NORMALIZACIÓN DE SCORES:
            # Esto evita que una clase con distancias muy grandes domine el ratio
            '''mean_s = np.mean(s)
            std_s = np.std(s) + 1e-8
            out[c] = (s - mean_s) / std_s'''
            # Obtenemos las probabilidades para la clase 1 (la clase actual)
            # predict_proba devuelve [prob_clase_0, prob_clase_1]
            #probs = self.models[c].predict_proba(X)[:, 1]
            #out[c] = probs

        return out
class OVR_LIBSVMEVL:
    def __init__(self, C=1.0, gamma=0.1):#gamma='scale'
        self.C = C
        self.gamma =  gamma
        self.models = {}
        self.labels = []

    def fit(self, X, y):
        self.labels = sorted(np.unique(y))
        num_classes = len(self.labels)

        for c in self.labels:
            # CASO CRÍTICO: Solo hay una clase en todo el train
            if num_classes < 2:
                # Guardamos None o un objeto dummy para indicar que no hay modelo
                self.models[c] = None
                continue

            y_bin = np.where(y == c, 1, -1)
            current_gamma = self.gamma * (1.0 + 0.001 if c == self.labels[0] else 1.0)
            model = SVC(
                C=self.C,
                gamma=current_gamma,
                kernel='poly',
                degree=3,
                probability=True,
            )
            model.fit(X, y_bin)
            self.models[c] = model

        return self

    def scores(self, X):
        out = {}
        for c in self.labels:
            if self.models[c] is None:
                # Si no hubo modelo porque solo había una clase,
                # devolvemos un score alto constante (ej. 1.0)
                # para que el OSR decida basándose en el Tau.
                out[c] = np.ones(X.shape[0]) * 1.0
            else:
                out[c] = self.models[c].decision_function(X)
        return out

# =====================================================================
# 1-VS-SET
# =====================================================================

class OneVsSet_LIBSVM:
    def __init__(self, C=1.0, gamma=0.1):
        self.svm = OVR_LIBSVM(C=C, gamma=gamma)
        self.tau = {}
        self.labels = []

    def fit(self, Xf, yf):
        self.svm.fit(Xf, yf)
        self.labels = self.svm.labels
        scores = self.svm.scores(Xf)

        for c in self.labels:
            pos_scores = scores[c][yf == c]
            self.tau[c] = np.percentile(pos_scores, 5)#95
        return self

    def tune(self, Xv, yv, known_labels):
        # Crea una instancia de la clase de métricas
        metrics = OSRMetrics()
        scores = self.svm.scores(Xv)
        bestF = -1
        best_tau = None
        for scale in [0.6, 0.8, 1.0, 1.2,1.4,1.6,1.8]:
            tau_scaled = {c: scale*self.tau[c] for c in self.labels}
            preds = []
            for i in range(len(Xv)):
                best_c = None
                best_s = -1e18
                for c in self.labels:
                    s = scores[c][i]
                    if  s >= tau_scaled[c] and s > best_s:#s >= 0 and s <= tau_scaled[c] and s > best_s
                        best_s = s
                        best_c = c
                preds.append(best_c if best_c else "UUC")

            F = metrics.micro_f1(yv, np.array(preds), known_labels)
            if F > bestF:
                bestF = F
                best_tau = tau_scaled

        self.tau = best_tau
        #print('tau',best_tau,F)
        return self

    def predict(self, X):
        scores = self.svm.scores(X)
        preds = []
        for i in range(len(X)):
            best_c = None
            best_s = -1e18
            for c in self.labels:
                s = scores[c][i]
                if s >= self.tau[c] and s > best_s:# s >= 0 and s <= self.tau[c] and s > best_s
                    best_s = s
                    best_c = c
            preds.append(best_c if best_c else "UUC")
        return np.array(preds)

class OneVsSet_LIBSVMEVL:
    def __init__(self, C=1.0, gamma=0.1):
        self.svm = OVR_LIBSVMEVL(C=C, gamma=gamma)
        self.tau = {}
        self.labels = []

    def fit(self, Xf, yf):
        self.svm.fit(Xf, yf)
        self.labels = self.svm.labels
        scores = self.svm.scores(Xf)
        self.stats = {}  # Para guardar media y std

        for c in self.labels:
            s_c = scores[c]
            self.stats[c] = {'mean': np.mean(s_c), 'std': np.std(s_c) + 1e-6}

            pos_scores = s_c[yf == c]
            self.tau[c] = np.percentile(pos_scores, 1)
        return self

    def tune(self, Xv, yv, known_labels):
        # Crea una instancia de la clase de métricas
        metrics = OSRMetrics()
        scores = self.svm.scores(Xv)
        bestF = -1
        best_tau = None

        # Ampliamos el rango de offsets para forzar al modelo a moverse
        for offset in [-0.5, -0.2, 0, 0.2, 0.5]:
            tau_scaled = {c: self.tau[c] + offset for c in self.labels}
            preds = []
            for i in range(len(Xv)):
                best_c = None
                best_s_norm = -1e18
                for c in self.labels:
                    s_raw = scores[c][i]
                    s_norm = (s_raw - self.stats[c]['mean']) / (self.stats[c]['std'] + 1e-6)

                    if s_raw >= tau_scaled[c]:
                        if s_norm > best_s_norm:
                            best_s_norm = s_norm
                            best_c = c
                preds.append(best_c if best_c else "UUC")

            # --- MEJORA DE MÉTRICA: Penalizar la aniquilación de clases ---
            F = metrics.micro_f1(yv, np.array(preds), known_labels)  # USAR OSR1

            # Penalización si el modelo solo predice una de las clases conocidas
            unique_preds = np.unique(preds)
            diversity_penalty = 0
            for c in known_labels:
                if str(c) not in unique_preds and c not in unique_preds:
                    diversity_penalty += 0.1  # Restamos 10% al F1 por cada clase ignorada

            score = F - diversity_penalty

            if score > bestF:
                bestF = score
                best_tau = tau_scaled

        self.tau = best_tau
        return self

    def predict(self, X):
        scores = self.svm.scores(X)
        preds = []
        for i in range(len(X)):
            candidates = []

            # PASO 1: Identificar TODAS las clases que superan su propio umbral
            for c in self.labels:
                s_raw = scores[c][i]
                if s_raw >= self.tau[c]:
                    # Calculamos la confianza normalizada solo para los que pasan el filtro
                    s_norm = (s_raw - self.stats[c]['mean']) / (self.stats[c]['std'] + 1e-6)
                    candidates.append((c, s_norm))

            # PASO 2: Decidir entre los candidatos
            if not candidates:
                # Nadie superó su tau -> Desconocido
                preds.append("UUC")
            elif len(candidates) == 1:
                # Solo una clase es apta -> Esa es la predicción
                preds.append(candidates[0][0])
            else:
                # Varias clases son aptas -> Elegimos la que tenga la mayor confianza normalizada
                # Ordenamos por el score normalizado (segundo elemento de la tupla)
                best_c = max(candidates, key=lambda x: x[1])[0]
                preds.append(best_c)

        return np.array(preds)

# =====================================================================
# W-OSVM CAP + W-SVM COMPLETO
# =====================================================================

class WOSVM_CAP:
    def __init__(self):
        self.centers={}; self.weib={}

    def fit(self, X, y):
        labels = sorted(np.unique(y))
        for c in labels:
            Xc = X[y==c]
            mu = Xc.mean(axis=0)
            d = np.linalg.norm(Xc-mu,axis=1)
            self.centers[c] = mu
            self.weib[c]    = fit_weibull_mle(d)
        self.labels = labels
        return self

    def prob(self, X):
        out={}
        for c in self.labels:
            mu = self.centers[c]; wp=self.weib[c]
            d = np.linalg.norm(X-mu,axis=1)
            lam = max(wp.lam,1e-8)
            out[c] = np.exp(-(d/lam)**wp.k)
        return out
class WSVM:
    def __init__(self, C=1.0, gamma=0.1,open=0.25):
        self.svm = OVR_LIBSVM(C=C,gamma=gamma)
        self.cap = WOSVM_CAP()
        self.pos_w = {}
        self.neg_w = {}
        self.thr_p = {}
        self.thr_n = {}
        self.eps = 0.001
        self.R = 0.5
        self.labels = None
        self.open=open

    def fit(self, Xf, yf, Xv_open, yv_open, kkc_labels):
        # Crea una instancia de la clase de métricas
        metrics = OSRMetrics()
        self.labels = sorted(np.unique(yf))
        self.svm.fit(Xf, yf)
        self.cap.fit(Xf, yf)
        sc = self.svm.scores(Xf)
        for c in self.labels:
            s = sc[c]
            s_pos = s[yf == c]
            s_neg = s[yf != c]
            # --- POSITIVE TAIL ---
            if len(s_pos) > 0:
                tailsize = max(50, int(len(s_pos) * 0.10))#POS
                s_sorted = np.sort(s_pos)
                thr_p = s_sorted[-tailsize]
                tail_p = s_sorted[-tailsize:] - thr_p + 1e-6
                self.thr_p[c] = thr_p
                self.pos_w[c] = fit_weibull_mle(tail_p)
            else:
                self.thr_p[c] = 0
                self.pos_w[c] = WeibullParams(1, 1)

            # --- NEGATIVE TAIL ---
            if len(s_neg) > 0:
                tailsize = max(50, int(len(s_neg) * 0.10))#NEG
                s_sorted = np.sort(s_neg)
                thr_n = s_sorted[tailsize - 1]
                tail_n = thr_n - s_sorted[:tailsize] + 1e-6
                self.thr_n[c] = thr_n
                self.neg_w[c] = fit_weibull_mle(tail_n)
            else:
                self.thr_n[c] = 0
                self.neg_w[c] = WeibullParams(1, 1)

        best_val = -1
        best_params = (self.eps, self.R)
        target_openness = self.open
        for eps in [0.1, 0.3, 0.5]:#   0.1,0.3,
            for R in[0.1, 0.5, 0.7]:# [0.5,0.7] [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
                yp = self.predict_inner(Xv_open, eps, R)
                F = metrics.micro_f1(yv_open, yp, kkc_labels)

                # Convertimos a arrays de numpy: JULIO 2026
                yp_array = np.array(yp)
                known_labels_array = np.array(kkc_labels)

                # np.isin(yp_array, known_labels_array) devuelve True si el elemento ESTÁ en conocidos
                # El símbolo ~ INVIERTE el resultado (True se vuelve False y viceversa)
                # Por lo tanto, ~np.isin devuelve True para todo lo que es "Open Set"
                is_open = ~np.isin(yp_array, known_labels_array)

                # Sumamos los Trues y dividimos por el total
                actual_openness = np.sum(is_open) / len(yp)
                openness_penalty = abs(actual_openness - target_openness)
                # Métrica combinada: F-score menos penalización por rechazo incorrecto
                #score = F - (openness_penalty * 0.2)
                score = F
                if score > best_val:
                    best_val = score
                    best_params = (eps, R)

        self.eps, self.R = best_params
        #print('BEST eps, R ',best_val, self.eps, self.R)
        return self
    def predict_inner(self, X, eps, R):
        sc = self.svm.scores(X)
        PO = self.cap.prob(X)
        n = len(X)
        out = []
        for i in range(n):
            best_c = None
            scores_for_sample = [sc[c][i] for c in self.labels]
            idx_best = np.argmax(scores_for_sample)
            best_c_svm = self.labels[idx_best]
            s = sc[best_c_svm][i]

            # 1. Verificación de Openness (CAP)
            if PO[best_c_svm][i] < eps:#+0.1
                out.append("UUC")
                continue # Salta al siguiente i

            # 2. Cálculo de Probabilidades Weibull
            # Positiva
            if s >= self.thr_p[best_c_svm]:
                z = s - self.thr_p[best_c_svm]
                wp = self.pos_w[best_c_svm]
                Ppos = np.exp(-(z / max(wp.lam, 1e-8)) ** wp.k)#9
            else:
                Ppos = 1.0

            # Negativa
            if s <= self.thr_n[best_c_svm]:
                z2 = self.thr_n[best_c_svm] - s
                wn = self.neg_w[best_c_svm]
                Pneg = np.exp(-(z2 / max(wn.lam, 1e-8))** wn.k)#9
            else:
                Pneg = 1.0

            # 3. DECISIÓN FINAL
            #ratio = Ppos / (Pneg + 1e-12)
            ratio = Pneg / (Ppos + 1e-12)
            # === INSERTA ESTO PARA EL DIAGNÓSTICO ===
            '''if i < 10:  # Imprimimos solo las primeras 10 muestras para no saturar la consola
                print(
                    f"Muestra {i} | Clase SVM: {best_c_svm} | Ppos: {Ppos:.4f} | Pneg: {Pneg:.4f} | Ratio: {ratio:.4f}")'''
            # ========================================
            if ratio >= R:
                out.append(best_c_svm)
            else:
                out.append("UUC")
        return np.array(out)

    def predict(self, X):
        return self.predict_inner(X, self.eps, self.R)
class WSVMEVL:
    def __init__(self, C=1.0, gamma=0.1,open=0.25):
        self.svm = OVR_LIBSVM(C=C,gamma=gamma)
        self.cap = WOSVM_CAP()
        self.pos_w = {}
        self.neg_w = {}
        self.thr_p = {}
        self.thr_n = {}
        self.eps = 0.001
        self.R = 0.5
        self.labels = None
        self.open=open

    def fit(self, Xf, yf, Xv_open, yv_open, kkc_labels):
        # Crea una instancia de la clase de métricas
        metrics = OSRMetrics()
        self.labels = sorted(np.unique(yf))
        self.svm.fit(Xf, yf)
        self.cap.fit(Xf, yf)
        sc = self.svm.scores(Xf)
        for c in self.labels:
            s = sc[c]
            s_pos = s[yf == c]
            s_neg = s[yf != c]
            # --- POSITIVE TAIL ---
            if len(s_pos) > 0:
                # Aseguramos que tailsize sea al menos 1, pero NUNCA más que la longitud del array
                if len(s_pos) <= 20:
                    tailsize = len(s_pos)  # Si hay menos de 20, usamos todos los disponibles
                else:
                    tailsize = int(len(s_pos) * 0.10)

                s_sorted = np.sort(s_pos)
                # Usamos el índice -tailsize pero asegurando que sea >= 0
                idx = max(0, len(s_sorted) - tailsize)
                thr_p = s_sorted[idx]

                tail_p = s_sorted[idx:] - thr_p + 1e-6
                self.thr_p[c] = thr_p
                s# Envolvemos el fit en un try-except por si MLE no converge con pocos datos
                try:
                    self.pos_w[c] = fit_weibull_mle(tail_p)
                except:
                    self.pos_w[c] = WeibullParams(1, 1)
            else:
                self.thr_p[c] = 0
                self.pos_w[c] = WeibullParams(1, 1)

            # --- NEGATIVE TAIL ---
            if len(s_neg) > 0:
                if len(s_neg) <= 20:
                    tailsize = len(s_neg)
                else:
                    tailsize = int(len(s_neg) * 0.10)
                s_sorted = np.sort(s_neg)
                # Para la cola negativa, tomamos los más pequeños
                idx = min(len(s_sorted) - 1, tailsize - 1)
                thr_n = s_sorted[idx]
                tail_n = thr_n - s_sorted[:idx + 1] + 1e-6
                self.thr_n[c] = thr_n
                try:
                    self.neg_w[c] = fit_weibull_mle(tail_n)
                except:
                    self.neg_w[c] = WeibullParams(1, 1)
            else:
                self.thr_n[c] = 0
                self.neg_w[c] = WeibullParams(1, 1)

        best_val = -1
        best_params = (self.eps, self.R)
        target_openness = self.open
        for eps in [0.3, 0.5]:#   0.1,0.3,
            for R in[0.5, 1.0, 2.0]:# [0.5,0.7] [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
                yp = self.predict_inner(Xv_open, eps, R)
                F = metrics.micro_f1(yv_open, yp, kkc_labels)

                # Convertimos a arrays de numpy
                yp_array = np.array(yp)
                known_labels_array = np.array(kkc_labels)

                # np.isin(yp_array, known_labels_array) devuelve True si el elemento ESTÁ en conocidos
                # El símbolo ~ INVIERTE el resultado (True se vuelve False y viceversa)
                # Por lo tanto, ~np.isin devuelve True para todo lo que es "Open Set"
                is_open = ~np.isin(yp_array, known_labels_array)

                # Sumamos los Trues y dividimos por el total
                actual_openness = np.sum(is_open) / len(yp)
                #actual_openness = np.sum(np.array(yp) == "UUC") / len(yp)

                openness_penalty = abs(actual_openness - target_openness)
                # Métrica combinada: F-score menos penalización por rechazo incorrecto
                #score = F - (openness_penalty * 0.2)
                score = F
                if score > best_val:
                    best_val = score
                    best_params = (eps, R)

        self.eps, self.R = best_params
        #print('BEST eps, R ',best_val, self.eps, self.R)
        return self
    def predict_inner(self, X, eps, R):
        sc = self.svm.scores(X)
        PO = self.cap.prob(X)
        n = len(X)
        out = []
        for i in range(n):
            best_c = None
            scores_for_sample = [sc[c][i] for c in self.labels]
            idx_best = np.argmax(scores_for_sample)
            best_c_svm = self.labels[idx_best]
            s = sc[best_c_svm][i]

            # 1. Verificación de Openness (CAP)
            if PO[best_c_svm][i] < eps:#+0.1
                out.append("UUC")
                continue # Salta al siguiente i

            # 2. Cálculo de Probabilidades Weibull
            # Positiva
            if s >= self.thr_p[best_c_svm]:
                z = s - self.thr_p[best_c_svm]
                wp = self.pos_w[best_c_svm]
                Ppos = np.exp(-(z / max(wp.lam, 1e-8)) ** wp.k)#9
            else:
                Ppos = 1.0

            # Negativa
            if s <= self.thr_n[best_c_svm]:
                z2 = self.thr_n[best_c_svm] - s
                wn = self.neg_w[best_c_svm]
                Pneg = np.exp(-(z2 / max(wn.lam, 1e-8))** wn.k)#9
            else:
                Pneg = 1.0

            # 3. DECISIÓN FINAL
            #ratio = Ppos / (Pneg + 1e-12)
            ratio = Pneg / (Ppos + 1e-12)
            # === INSERTA ESTO PARA EL DIAGNÓSTICO ===
            '''if i < 10:  # Imprimimos solo las primeras 10 muestras para no saturar la consola
                print(
                    f"Muestra {i} | Clase SVM: {best_c_svm} | Ppos: {Ppos:.4f} | Pneg: {Pneg:.4f} | Ratio: {ratio:.4f}")'''
            # ========================================
            if ratio >= R:
                out.append(best_c_svm)
            else:
                out.append("UUC")
            # ... final de predict ...
            '''preds_array = np.array(out)
            print(f"DEBUG: Labels originales: {self.labels}")
            print(f"DEBUG: Predicciones únicas: {np.unique(preds_array)}")'''
        return np.array(out)

    def predict(self, X):
        return self.predict_inner(X, self.eps, self.R)

# =====================================================================
# WOSVMModel
# =====================================================================
class WOSVMModel:
    def __init__(self, C=1.0, openess=0.25, gamma="scale", nu=0.05, dt=1e-3, tail_size=50):
        self.C = C
        self.openess = openess
        self.gamma = gamma
        self.nu = nu
        self.dt = dt
        self.tail_size = tail_size
        # Cambiamos Dict[int, ...] por Dict[any, ...] para soportar letras o números
        self.models: Dict = {}
        self.weibull_params: Dict = {}
        self.score_shift: Dict = {}
        self.classes_: np.ndarray | None = None

    def fit(self, X, y, *args, **kwargs):
        self.classes_ = np.unique(y)
        self.models = {}
        self.weibull_params = {}
        self.score_shift = {}

        for c in self.classes_:
            pos_mask = (y == c)
            oc = OneClassSVM(kernel="rbf", nu=self.nu, gamma=self.gamma)
            oc.fit(X[pos_mask])

            s = oc.decision_function(X[pos_mask]).astype(float).ravel()
            tail_n = min(max(5, self.tail_size), len(s))
            tail = np.sort(s)[:tail_n]
            smin = float(tail.min()) - 1e-6
            z = np.clip(tail - smin, 1e-9, None)

            shape, loc, scale = weibull_min.fit(z, floc=0)

            # GUARDAMOS USANDO 'c' (la letra), NO el índice
            self.models[c] = oc
            self.weibull_params[c] = (shape, loc, scale)
            self.score_shift[c] = smin
            #print(f"Fitted class: {c}")

        return self

    def _get_scores(self, X):
        # Usamos directamente self.classes_ sin intentar convertir a int
        out = np.empty((X.shape[0], len(self.classes_)), dtype=float)

        for j, c in enumerate(self.classes_):
            # Accedemos al modelo usando la etiqueta original (ej. 'A')
            oc = self.models[c]
            shape, loc, scale = self.weibull_params[c]
            smin = self.score_shift[c]

            raw = oc.decision_function(X).astype(float).ravel()
            z = np.clip(raw - smin + 1e-6, 1e-9, 1e6)
            p = weibull_min.cdf(z, c=shape, loc=loc, scale=scale)

            p[p <= self.dt] = 0.0
            out[:, j] = p
        return out

    def predict(self, X, thr=0.35):
        scores = self._get_scores(X)
        best_idx = np.argmax(scores, axis=1)
        best_scores = scores[np.arange(scores.shape[0]), best_idx]

        preds = []
        for i in range(len(X)):
            if best_scores[i] < thr:
                preds.append("UUC")
            else:
                # best_idx[i] nos da la posición, que corresponde a self.classes_
                preds.append(self.classes_[best_idx[i]])
        return np.array(preds)
# =====================================================================
# WSVMModel
# =====================================================================
class WSVMModel:
    def __init__(self, C=1.0, openess=0.25, gamma="scale", nu=0.05, dt=1e-3, tail_size=20):
        self.C = C
        self.openess = openess
        self.gamma = gamma
        self.nu = nu
        self.dt = dt
        self.tail_size = tail_size
        self.bin_models = {}
        self.oc_models = {}
        self.classes_ = None
        self.pos_weibull, self.pos_shift = {}, {}
        self.neg_weibull, self.neg_vmax = {}, {}
        self.oc_weibull, self.oc_shift = {}, {}

    def fit(self, X, y, *args, **kwargs):
        self.classes_ = np.unique(y)

        for idx, c in enumerate(self.classes_):
            pos_mask = (y == c)
            neg_mask = ~pos_mask

            clf = SVC(C=self.C, kernel="rbf", gamma=self.gamma)
            clf.fit(X, np.where(pos_mask, 1, -1))
            self.bin_models[idx] = clf

            s = clf.decision_function(X)
            spos, sneg = s[pos_mask], s[neg_mask]

            tp = np.sort(spos)[:self.tail_size]
            ps = float(tp.min()) - 1e-6
            self.pos_weibull[idx] = weibull_min.fit(np.clip(tp - ps, 1e-9, None), floc=0)
            self.pos_shift[idx] = ps

            tn = np.sort(sneg)[-self.tail_size:]
            vm = float(tn.max()) + 1e-6
            self.neg_weibull[idx] = weibull_min.fit(np.clip(vm - tn, 1e-9, None), floc=0)
            self.neg_vmax[idx] = vm

            oc = OneClassSVM(kernel="rbf", nu=self.nu, gamma=self.gamma)
            oc.fit(X[pos_mask])
            self.oc_models[idx] = oc
            ocs = oc.decision_function(X[pos_mask])
            oct = np.sort(ocs)[:self.tail_size]
            os = float(oct.min()) - 1e-6
            self.oc_weibull[idx] = weibull_min.fit(np.clip(oct - os, 1e-9, None), floc=0)
            self.oc_shift[idx] = os
        return self

    def _get_scores(self, X):
        out = np.empty((X.shape[0], len(self.classes_)))
        for idx in range(len(self.classes_)):
            raw = self.bin_models[idx].decision_function(X)
            ph = weibull_min.cdf(np.clip(raw - self.pos_shift[idx], 1e-9, 1e6), *self.pos_weibull[idx])
            pc = weibull_min.cdf(np.clip(self.neg_vmax[idx] - raw, 1e-9, 1e6), *self.neg_weibull[idx])

            oc_raw = self.oc_models[idx].decision_function(X)
            po = weibull_min.cdf(np.clip(oc_raw - self.oc_shift[idx], 1e-9, 1e6), *self.oc_weibull[idx])
            gate = (po > self.dt).astype(float)
            out[:, idx] = ph * pc * gate
        return out

    def predict(self, X, thr=0.5):
        scores = self._get_scores(X)
        best_idx = np.argmax(scores, axis=1)
        best_scores = scores[np.arange(scores.shape[0]), best_idx]

        preds = []
        for i in range(len(X)):
            if best_scores[i] < thr:
                preds.append("UUC")
            else:
                preds.append(self.classes_[best_idx[i]])
        return np.array(preds)

    def predict_inner(self, X, eps=0.001, R=0.5):
        return self.predict(X, thr=R)
# =====================================================================
# CDOSRModel
# =====================================================================
class CDOSRModel:
    def __init__(self, max_components=12, weight_concentration_prior=0.1, reg_covar=1e-4):
        self.max_components = max_components
        self.weight_concentration_prior = weight_concentration_prior
        self.reg_covar = reg_covar
        # ELIMINADO: self.scaler = StandardScaler()
        self.classes_ = None
        self.mixers = {}
        self.known_logpost_floor = {}

    def fit(self, X, y, _args, *_kwargs):
        self.classes_ = np.unique(y)
        for idx, c in enumerate(self.classes_):
            Xc = X[y == c]
            if Xc.shape[0] < 2:
                print(f"Warning: Clase {c} tiene solo {Xc.shape[0]} muestra(s). Saltando GMM.")
                self.mixers[idx] = None
                self.known_logpost_floor[idx] = -1e10
                continue

            try:
                # Intento 1: Configuración normal
                mix = BayesianGaussianMixture(
                    n_components=min(self.max_components, Xc.shape[0]),
                    covariance_type="full",
                    weight_concentration_prior_type="dirichlet_process",
                    weight_concentration_prior=self.weight_concentration_prior,
                    reg_covar=self.reg_covar,
                    random_state=0
                ).fit(Xc)
            except (ValueError, np.linalg.LinAlgError):
                print(f"⚠️ Fallo numérico en Clase {c} con cov='full'. Intentando fallback...")
                try:
                    # Intento 2: Aumentar drásticamente reg_covar
                    mix = BayesianGaussianMixture(
                        n_components=min(self.max_components, Xc.shape[0]),
                        covariance_type="full",
                        weight_concentration_prior_type="dirichlet_process",
                        weight_concentration_prior=self.weight_concentration_prior,
                        reg_covar=1e-2,  # Regularización fuerte
                        random_state=0
                    ).fit(Xc)
                except (ValueError, np.linalg.LinAlgError):
                    print(f"❌ Fallo persistente en Clase {c}. Cambiando a cov='diag'.")
                    # Intento 3: Covarianza diagonal (siempre es estable)
                    mix = BayesianGaussianMixture(
                        n_components=min(self.max_components, Xc.shape[0]),
                        covariance_type="diag",
                        weight_concentration_prior_type="dirichlet_process",
                        weight_concentration_prior=self.weight_concentration_prior,
                        reg_covar=self.reg_covar,
                        random_state=0
                    ).fit(Xc)

            self.mixers[idx] = mix
            self.known_logpost_floor[idx] = float(np.quantile(mix.score_samples(Xc), 0.05))
        return self
    def fitold(self, X, y, *args, **kwargs):
        # USAMOS X DIRECTAMENTE
        self.classes_ = np.unique(y)

        for idx, c in enumerate(self.classes_):
            Xc = X[y == c]
            # --- VALIDACIÓN DE MUESTRAS MÍNIMAS ---
            if Xc.shape[0] < 2:
                print(f"Warning: Clase {c} tiene solo {Xc.shape[0]} muestra(s). Saltando GMM.")
                # Guardamos None o un objeto indicador para saber que no hay modelo GMM
                self.mixers[idx] = None
                # Definimos un floor muy bajo para que sea difícil que esta clase gane
                self.known_logpost_floor[idx] = -1e10
                continue
            # ---------------------------------------

            mix = BayesianGaussianMixture(
                n_components=min(self.max_components, Xc.shape[0]),
                covariance_type="full",
                weight_concentration_prior_type="dirichlet_process",
                weight_concentration_prior=self.weight_concentration_prior,
                reg_covar=self.reg_covar,
                random_state=0
            ).fit(Xc)

            self.mixers[idx] = mix
            self.known_logpost_floor[idx] = float(np.quantile(mix.score_samples(Xc), 0.05))
        return self

    def _get_scores(self, X):
        out = np.empty((X.shape[0], len(self.classes_)))

        for idx in range(len(self.classes_)):
            # --- CORRECCIÓN AQUÍ: Verificar si existe el modelo ---
            if self.mixers[idx] is None:
                # Si no hay modelo, asignamos la probabilidad más baja posible (0.0)
                # para que no sea seleccionada como la mejor clase.
                out[:, idx] = 0.0
            else:
                logp = self.mixers[idx].score_samples(X)
                # Usamos expit (sigmoid) para normalizar el score entre 0 y 1
                out[:, idx] = expit(logp - self.known_logpost_floor[idx])

        return out


    def predict(self, X, thr=0.1):
        scores = self._get_scores(X)
        best_idx = np.argmax(scores, axis=1)
        best_scores = scores[np.arange(scores.shape[0]), best_idx]

        preds = []
        for i in range(len(X)):
            if best_scores[i] < thr:
                preds.append("UUC")
            else:
                # CORRECCIÓN: Se recupera la etiqueta original usando el índice
                preds.append(self.classes_[best_idx[i]])

        return np.array(preds)
# =====================================================================
# PI-SVM
# =====================================================================

class PISVM_LIBSVM:
    def __init__(self, C=1.0, gamma=0.1):
        self.svm = OVR_LIBSVM(C=C, gamma=gamma)
        self.labels = []
        self.tau = 0.5

    def fit(self, Xf, yf, Xv, yv, known_labels):
        # Crea una instancia de la clase de métricas
        metrics = OSRMetrics()
        self.svm.fit(Xf, yf)
        self.labels = self.svm.labels

        bestF = -1
        best_tau = 0.5

        for tau in np.arange(0.1, 0.99, 0.05):
            preds = self._predict_inner(Xv, tau)
            F = metrics.micro_f1(yv, preds, known_labels)
            if F > bestF:
                bestF = F
                best_tau = tau

        self.tau = best_tau
        return self

    def _predict_inner(self, X, tau):
        scores = self.svm.scores(X)
        preds = []

        for i in range(len(X)):
            best_c = None
            best_s = -1e18

            for c in self.labels:
                s = scores[c][i]
                if s > best_s:
                    best_s = s
                    best_c = c

            preds.append(best_c if best_s >= tau else "UUC")

        return np.array(preds)

    def predict(self, X):
        return self._predict_inner(X, self.tau)
# =====================================================================
# SROSR (OMP + EVT en errores)
# =====================================================================

class SROSR:
    def __init__(self, variance_keep=0.95):#0.95
        self.variance_keep = variance_keep  # Solo mantenemos el 95% de la varianza
        self.dict = {}
        self.thr = {}
        self.tau = 0.5
        self.labels = []

    def fit(self, Xf, yf, *args, **kwargs):
        # fit ahora es rápido: solo construye subespacios
        labels = sorted(np.unique(yf))
        self.labels = labels
        self.dict = {}
        thresholds = {}

        for c in labels:
            X_c = Xf[yf == c]
            X_mean = np.mean(X_c, axis=0)
            X_centered = X_c - X_mean
            U, S, Vh = np.linalg.svd(X_centered, full_matrices=False)

            explained_variance = (S ** 2) / np.sum(S ** 2)
            cumulative_variance = np.cumsum(explained_variance)
            k = np.argmax(cumulative_variance >= self.variance_keep) + 1

            self.dict[c] = {'basis': Vh[:k].T, 'mean': X_mean}

            basis = self.dict[c]['basis']
            X_proj = (X_centered @ basis) @ basis.T
            errors = np.linalg.norm(X_centered - X_proj, axis=1)
            thresholds[c] = np.percentile(errors, 90)

        self.thr = thresholds
        return self

    def fitold(self, Xf, yf, Xv, yv, kkc_labels):
        # Crea una instancia de la clase de métricas
        metrics = OSRMetrics()
        labels = sorted(np.unique(yf))
        self.labels = labels

        self.dict = {}
        thresholds = {}
        for c in labels:
            X_c = Xf[yf == c]
            # Centrado de datos para que el SVD sea realmente PCA
            X_mean = np.mean(X_c, axis=0)
            X_centered = X_c - X_mean

            U, S, Vh = np.linalg.svd(X_centered, full_matrices=False)

            # --- TRUNCAMIENTO DE VARIANZA (Clave para OSR) ---
            # Calculamos cuántos componentes necesitamos para explicar el % de varianza
            explained_variance = (S ** 2) / np.sum(S ** 2)
            cumulative_variance = np.cumsum(explained_variance)
            k = np.argmax(cumulative_variance >= self.variance_keep) + 1

            # Guardamos solo los primeros k componentes y la media
            self.dict[c] = {
                'basis': Vh[:k].T,
                'mean': X_mean
            }

            # Error de proyección con el subespacio truncado
            basis = self.dict[c]['basis']
            X_proj = (X_centered @ basis) @ basis.T
            errors = np.linalg.norm(X_centered - X_proj, axis=1)
            thresholds[c] = np.percentile(errors, 90)#90

        self.thr = thresholds

        # Tuning de Tau
        best_f1 = -1
        best_tau = 0.5
        raw_scores = self.predict(Xv, raw=True)

        # Buscamos Tau en un rango más amplio y fino
        for tau in np.linspace(0.001, 2.0, 500):
            yp = np.array([p[0] if p[1] >= tau else "UUC" for p in raw_scores])
            f1 = metrics.micro_f1(yv, yp, kkc_labels)
            if f1 > best_f1:
                best_f1 = f1
                best_tau = tau

        self.tau = best_tau
        print(f"SROSR-PCA -> Best Tau: {self.tau:.4f} | Val F1: {best_f1:.4f}")
        return self

    def _get_proj_err(self, c, x):
        # Proyección con datos centrados y base truncada
        basis = self.dict[c]['basis']
        mean = self.dict[c]['mean']
        x_centered = x - mean

        # Proyección: x_proj = (x_centered . basis) . basis^T
        proj_coords = x_centered @ basis
        x_proj = proj_coords @ basis.T
        return np.linalg.norm(x_centered - x_proj)

    def _score(self, x):
        errs = {c: self._get_proj_err(c, x) for c in self.labels}
        c_hat = min(errs, key=errs.get)

        # Ratio de confianza: Umbral / Error
        # Si Error > Umbral, el score es < 1 -> Candidato a UUC
        score = self.thr[c_hat] / (errs[c_hat] + 1e-6)
        return c_hat, score

    def predictold(self, X, raw=False):
        out = []
        for i in range(len(X)):
            out.append(self._score(X[i]))
        if raw: return out
        final_preds = [p[0] if p[1] >= self.tau else "UUC" for p in out]
        return np.array(final_preds)

    def predict(self, X, tau=0.5):  # tau ahora es un argumento
        out = []
        for i in range(len(X)):
            out.append(self._score(X[i]))

        final_preds = [p[0] if p[1] >= tau else "UUC" for p in out]
        return np.array(final_preds)
# =====================================================================
# OSNN
# =====================================================================

class OSNN:

    def __init__(self, k=10):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def ratio(self, X):
        d2 = self._dist2(X, self.X)
        idx = np.argsort(d2, axis=1)

        ratios = []
        nn_labels = []
        for i in range(len(X)):
            nn1 = idx[i, 0]
            c1 = self.y[nn1]
            d1 = np.sqrt(d2[i, nn1])

            d2_ = None
            for j in idx[i, 1:]:
                if self.y[j] != c1:
                    d2_ = np.sqrt(d2[i, j])
                    break

            if d2_ is None:
                ratios.append(1.0)
            else:
                ratios.append(d1 / (d2_ + 1e-12))

            nn_labels.append(c1)
        return np.array(ratios), np.array(nn_labels)

    def _dist2(self, A, B):
        AA = np.sum(A*A, axis=1)[:, None]
        BB = np.sum(B*B, axis=1)[None, :]
        d2 = AA + BB - 2*A@B.T
        return np.maximum(d2, 0)

    def predict(self, X, thr):
        ''''''
        d2 = self._dist2(X, self.X)
        idx = np.argsort(d2, axis=1)
        preds = []

        for i in range(len(X)):
            nn1 = idx[i, 0]
            c1 = self.y[nn1]
            d1 = np.sqrt(d2[i, nn1])
            d2_ = None
            for j in idx[i, 1:]:
                if self.y[j] != c1:
                    d2_ = np.sqrt(d2[i, j])
                    break

            if d2_ is None:
                ratio = 1.0
            else:
                ratio = d1 / (d2_ + 1e-12)
            if ratio <= thr:
                preds.append(c1)
            else:
                preds.append("UUC")

        return np.array(preds)


class OSNNEVL:
    def __init__(self, k=10):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def _dist2(self, A, B):
        AA = np.sum(A * A, axis=1)[:, None]
        BB = np.sum(B * B, axis=1)[None, :]
        d2 = AA + BB - 2 * A @ B.T
        return np.maximum(d2, 0)

    def ratio(self, X):
        d2 = self._dist2(X, self.X)
        idx = np.argsort(d2, axis=1)
        ratios = []
        nn_labels = []
        for i in range(len(X)):
            nn1 = idx[i, 0]
            c1 = self.y[nn1]
            d1 = np.sqrt(d2[i, nn1])
            d2_ = None
            for j in idx[i, 1:]:
                if self.y[j] != c1:
                    d2_ = np.sqrt(d2[i, j])
                    break

            # CORRECCIÓN 1: Si no hay vecino de otra clase,
            # significa que es MUY similar a la clase c1.
            # Debe tener un ratio muy BAJO (muy seguro), no 1.0.
            if d2_ is None:
                ratios.append(0.0)
            else:
                ratios.append(d1 / (d2_ + 1e-12))
            nn_labels.append(c1)
        return np.array(ratios), np.array(nn_labels)

    def predict(self, X, thr):
        d2 = self._dist2(X, self.X)
        idx = np.argsort(d2, axis=1)
        preds = []
        for i in range(len(X)):
            nn1 = idx[i, 0]
            c1 = self.y[nn1]
            d1 = np.sqrt(d2[i, nn1])
            d2_ = None
            for j in idx[i, 1:]:
                if self.y[j] != c1:
                    d2_ = np.sqrt(d2[i, j])
                    break

            # CORRECCIÓN 2: Consistencia con el método ratio()
            if d2_ is None:
                ratio = 0.0  # Se considera muy seguro
            else:
                ratio = d1 / (d2_ + 1e-12)

            if ratio <= thr:
                preds.append(c1)
            else:
                preds.append("UUC")
        return np.array(preds)

# =====================================================================
# EVM
# =====================================================================
def euclidean_cdist( X, Y):
    return sklearn.metrics.pairwise.pairwise_distances(X, Y, metric="euclidean", n_jobs=1)

def euclidean_pdist( X):
    return sklearn.metrics.pairwise.pairwise_distances(X, metric="euclidean", n_jobs=1)
def cosine_cdist( X, Y):
    return sklearn.metrics.pairwise.pairwise_distances(X, Y, metric="cosine", n_jobs=1)
def cosine_pdist( X):
    return sklearn.metrics.pairwise.pairwise_distances(X, metric="cosine", n_jobs=1)

def set_cover_greedy(universe, subsets, cost=lambda x: 1.0):
    universe = set(universe)
    # subsets = map(set,subsets)
    subsets = list(map(set, subsets))  # Convierte 'subsets' a una lista antes de usarla
    covered = set()
    cover_indices = []
    while covered != universe:
        subset_lengths = [len(subset - covered) for subset in subsets]
        # Verifica que subset_lengths no tenga valores constantes
        #print(f"Subset lengths: {subset_lengths}")
        # Calculamos el índice del conjunto con la longitud máxima (el que más elementos no cubiertos tiene)
        max_index = np.argmax(subset_lengths)
        #print(f"max_index: {max_index}")
        if max(subset_lengths) == 0:
            break

        # Asegúrate de que max_index esté dentro del rango de la lista
        if 0 <= max_index < len(subsets):
            # Actualiza el conjunto cubierto con el subconjunto seleccionado
            covered |= subsets[max_index]
            cover_indices.append(max_index)  # Guarda el índice del subconjunto elegido
            #print(f"Subconjunto seleccionado: {subsets[max_index]}")
        else:
            print(f"Índice fuera de rango: {max_index}")
            break  # Termina si el índice es inválido

    # print(f"Conjunto cubierto final: {covered}")
    return cover_indices

def set_cover(points, weibulls, solver=set_cover_greedy):
    universe = range(len(points))
    d_mat = pdist_func(points)
    p = Pool(cpu_count())
    probs = np.array(p.map(weibull_eval_parallel, zip(d_mat, weibulls)))
    p.close()
    p.join()

    # --- CORRECCIÓN CRÍTICA AQUÍ ---
    # 1. Inicializamos el diccionario con conjuntos vacíos para TODOS los puntos del universo
    # Esto evita el KeyError: 0 porque garantizamos que cada índice existe.
    subsets_dict = {i: set() for i in universe}

    # 2. Encontramos dónde se cumple el umbral
    # np.where devuelve (filas, columnas).
    # filas = qué punto es cubierto, columnas = qué modelo Weibull lo cubre.
    rows, cols = np.where(probs >= cover_threshold)

    # 3. Llenamos el diccionario: para cada punto (row), añadimos el modelo (col) que lo cubre
    for r, c in zip(rows, cols):
        if r in subsets_dict:
            subsets_dict[r].add(c)

    # 4. Ahora creamos la lista manteniendo el orden del universo
    # Si un punto no fue cubierto por nada, tendrá un set() vacío, lo cual es correcto.
    subsets_list = [subsets_dict[i] for i in universe]

    # Llamamos al solver
    keep_indices = solver(universe, subsets_list)
    return keep_indices

def weibull_fit_parallel(args):
    """Parallelized for efficiency"""
    global tailsize
    dists, row, labels = args

    # 1. Filtrar distancias de muestras de OTRAS clases
    others_mask = (labels != labels[row])
    others_dists = dists[others_mask]

    # 2. VALIDACIÓN CRÍTICA: ¿Tenemos datos suficientes para el tailsize?
    # Si no hay muestras de otras clases, o hay menos que el tailsize
    if len(others_dists) == 0:
        # No hay datos de otras clases: devolvemos un modelo vacío/genérico
        # Creamos un objeto MR y lo devolvemos como string para mantener consistencia
        dummy_mr = libmr.MR()
        return str(dummy_mr)

    # Ajustamos el tailsize al tamaño real de los datos disponibles
    # para evitar el ValueError: fit_size larger than data size
    actual_tailsize = min(tailsize, len(others_dists))

    try:
        # Usamos np.partition para obtener los 'actual_tailsize' más cercanos
        nearest = np.partition(others_dists, actual_tailsize - 1)[:actual_tailsize]

        mr = libmr.MR()
        mr.fit_low(nearest, actual_tailsize)
        return str(mr)
    except Exception as e:
        # En caso de cualquier error numérico en libmr, devolvemos un modelo dummy
        print(f"Error fitting Weibull for row {row}: {e}")
        return str(libmr.MR())

def weibull_eval_parallel(args):
    """Parallelized for efficiency"""
    dists, weibull_params = args
    mr = libmr.load_from_string(weibull_params)
    probs = mr.w_score_vector(dists)
    return probs
def fuse_prob_for_label(prob_mat, num_to_fuse):
    # 1. Verificamos cuántos elementos hay realmente disponibles
    # prob_mat.shape[0] es el número de instancias de la clase actual
    actual_k = min(num_to_fuse, prob_mat.shape[0])

    # 2. Caso extremo: no hay ninguna instancia
    if actual_k <= 0:
        # Devolvemos ceros con la forma (n_test,)
        return np.zeros(prob_mat.shape[1]) if prob_mat.ndim > 1 else np.zeros(1)

    # 3. Usamos actual_k para el particionamiento
    # Esto evita el ValueError: kth out of bounds
    partitioned = np.partition(prob_mat, -actual_k, axis=0)

    # Tomamos los actual_k mejores y calculamos el promedio
    return np.average(partitioned[-actual_k:, :], axis=0)
# =====================================================================
# Utilidades old
# =====================================================================
def set_cover_greedyold(universe, subsets, cost=lambda x: 1.0):
    universe = set(universe)
    subsets = list(map(set, subsets))
    covered = set()
    cover_indices = []
    while covered != universe:
        # cuántos elementos nuevos cubre cada subconjunto
        gains = [len(s - covered) for s in subsets]
        if not gains or max(gains) == 0:
            break
        max_index = int(np.argmax(gains))
        covered |= subsets[max_index]
        cover_indices.append(max_index)
    return cover_indices
def set_coverold2(points, weibulls, solver=set_cover_greedy):
    universe = range(len(points))
    d_mat = pdist_func(points)  # points array_Points
    p = Pool(cpu_count())
    probs = np.array(p.map(weibull_eval_parallel, zip(d_mat, weibulls)))
    p.close()
    p.join()
    thresholded = zip(*np.where(probs >= cover_threshold))
    subsets = {k: tuple(set(x[1] for x in v)) for k, v in it.groupby(thresholded, key=lambda x: x[0])}
    subsets = [subsets[i] for i in universe]
    keep_indices = solver(universe, subsets)
    return keep_indices
def set_coverold(points, weibulls, solver=set_cover_greedyold):
    # points: (n_pts, n_feat) ; weibulls: lista de len n_pts
    n = len(points)
    if n == 0:
        return []
    if n == 1:
        return [0]

    d_mat = pdist_func(points)  # (n, n)
    # Evaluar Weibull fila a fila
    probs = np.array(list(map(weibull_eval_parallel, zip(d_mat, weibulls))))  # (n, n)

    # construir subconjuntos por fila (qué columnas superan el umbral)
    rows, cols = np.where(probs >= cover_threshold)
    subsets = [set() for _ in range(n)]
    for r, c in zip(rows, cols):
        subsets[r].add(c)

    keep_indices = solver(range(n), subsets)

    # Si no se eligió nada, conservar todo (fallback seguro)
    if not keep_indices:
        keep_indices = list(range(n))

    return keep_indices

def weibull_fit_parallelold(args):
    """Parallelized for efficiency"""
    global tailsize
    dists, row, labels = args
    nearest = np.partition(dists[np.where(labels != labels[row])], tailsize)
    mr = libmr.MR()
    mr.fit_low(nearest, tailsize)
    return str(mr)

def fuse_prob_for_labelold(prob_mat, num_to_fuse):
    return np.average(np.partition(prob_mat, -num_to_fuse, axis=0)[-num_to_fuse:, :], axis=0)

# =====================================================================
# Utilidades
# =====================================================================
def load_data(fname):
    with open(fname) as f:
        data = f.read().splitlines()
    data = [x.split(",") for x in data]
    labels = [x[0] for x in data]

    data = [x[1:] for x in data]
    data = [x for x in data if not callable(x)]
    data = [[float(x) for x in sublist] for sublist in data]

    # data = [map(lambda y: float(y),x[1:]) for x in data]
    return np.array(data), np.array(labels)

def get_accuracy(predictions, labels):
    return sum(predictions == labels) / float(len(predictions))

def update_params(n_tailsize,
                  n_cover_threshold,
                  n_cdist_func,
                  n_pdist_func,
                  n_num_to_fuse,
                  n_margin_scale):
    global tailsize, cover_threshold, cdist_func, pdist_func, num_to_fuse, margin_scale
    tailsize = n_tailsize
    cover_threshold = n_cover_threshold
    cdist_func = n_cdist_func
    pdist_func = n_pdist_func
    num_to_fuse = n_num_to_fuse
    margin_scale = n_margin_scale

dist_func_lookup = {
            "cosine": {"cdist": cosine_cdist,
                       "pdist": cosine_pdist},

            "euclidean": {"cdist": euclidean_cdist,
                          "pdist": euclidean_pdist}
}

parser = argparse.ArgumentParser()
parser.add_argument("--tailsize",
                    type=int,
                    help="number of points that constitute \'extrema\'",
                    default=50)
parser.add_argument("--cover_threshold",
                    type=float,
                    help="probabilistic threshold to designate redundancy between points",
                    default=0.5)
parser.add_argument("--distance",
                    type=str,
                    default="euclidean",
                    choices=dist_func_lookup.keys())
parser.add_argument("--nfuse",
                    type=int,
                    help="number of extreme vectors to fuse over",
                    default=4)
parser.add_argument(
    "--margin_scale",
    type=float,
    help="multiplier by which to scale the margin distribution",
    default=0.5)

# set parameters; default if no command line arguments
args = parser.parse_args()
tailsize = args.tailsize
cover_threshold = args.cover_threshold
cdist_func = dist_func_lookup[args.distance]["cdist"]
pdist_func = dist_func_lookup[args.distance]["pdist"]
num_to_fuse = args.nfuse
margin_scale = args.margin_scale

class EVM_OFFICIAL:
    def __init__(self,
                 tailsize=50,
                 cover_threshold=0.5,
                 nfuse=4,
                 margin_scale=0.5,
                 ):
        self.n_tailsize = tailsize
        self.n_cover_threshold = cover_threshold
        self.n_num_to_fuse = nfuse
        self.n_margin_scale = margin_scale
        self.n_cdist_func=cdist_func
        self.n_pdist_func=pdist_func

        self.points = None  # se fija en fit
        self.weibulls = None  # se fija en fit y se reduce en reduce_model
        self.labels = None  # se fija en fit y se reduce en reduce_model

    def fit(self, X, y):
        # fija estado interno para que reduce_model no dependa de arrays externos
        self.points = np.asarray(X)
        self.labels = np.asarray(y)

        # Ajusta Weibulls por muestra (como ya hacías)
        d_mat = self.n_margin_scale * pdist_func(self.points)  # (n, n)

        p = Pool(cpu_count())
        row_range = range(len(d_mat))
        args = zip(d_mat, row_range, [self.labels for _ in row_range])
        weibulls = p.map(weibull_fit_parallel, args)
        p.close()
        p.join()

        self.weibulls = weibulls

    def reduce_model(self, labels_to_reduce=None):
        if self.points is None or self.labels is None or self.weibulls is None:
            raise RuntimeError("Debes llamar a fit() antes de reduce_model().")

        if cover_threshold >= 1.0:
            # caso trivial: no reducir
            return

        points = self.points
        labels = self.labels
        weibulls = self.weibulls

        ulabels = np.unique(labels)
        if labels_to_reduce is None:
            labels_to_reduce = set(ulabels)
        else:
            labels_to_reduce = set(labels_to_reduce)

        keep = []

        for ulabel in ulabels:
            ind = np.where(labels == ulabel)[0]
            if ulabel not in labels_to_reduce:
                keep.extend(ind.tolist())
                continue

            # Si la clase tiene un solo ejemplo, consérvalo
            if len(ind) == 1:
                keep.extend(ind.tolist())
                continue

            # Ejecuta set_cover dentro del bloque de la clase
            sub_keep = set_cover(points[ind], [weibulls[i] for i in ind])
            if not sub_keep:
                # fallback conservador
                keep.extend(ind.tolist())
            else:
                keep.extend(ind[sub_keep].tolist())

        keep = np.array(sorted(set(keep)), dtype=int)

        # aplica reducción in-place y deja el modelo consistente
        self.points = points[keep]
        self.labels = labels[keep]
        self.weibulls = [weibulls[i] for i in keep]

    def predictold(self, X, threshold=0.5):
        if self.points is None or self.weibulls is None or self.labels is None:
            raise RuntimeError("Debes llamar a fit() (y opcionalmente reduce_model()) antes de predict().")

        d_mat = cdist_func(self.points, X).astype(np.float64)  # (n_kept, n_test)

        p = Pool(cpu_count())
        probs = np.array(p.map(weibull_eval_parallel, zip(d_mat, self.weibulls)))  # (n_kept, n_test)
        p.close()
        p.join()

        ulabels = np.unique(self.labels)

        # fusionar probabilidades por clase
        fused = []
        for ulabel in ulabels:
            cls_mask = (self.labels == ulabel)
            fused.append(fuse_prob_for_label(probs[cls_mask], num_to_fuse))
        fused = np.vstack(fused)  # (n_classes, n_test)

        max_ind = np.argmax(fused, axis=0)  # (n_test,)
        max_prob = fused[max_ind, np.arange(fused.shape[1])]
        pred_labels = ulabels[max_ind]

        # aplicar threshold para desconocidos
        final = np.array(pred_labels, dtype=object)
        final[max_prob < threshold] = "UUC"

        return final, fused

    def predict(self, X, threshold=0.5):
        if self.points is None or self.weibulls is None or self.labels is None:
            raise RuntimeError("Debes llamar a fit() (y opcionalmente reduce_model()) antes de predict().")

        d_mat = cdist_func(self.points, X).astype(np.float64)
        p = Pool(cpu_count())
        probs = np.array(p.map(weibull_eval_parallel, zip(d_mat, self.weibulls)))
        p.close()
        p.join()

        ulabels = np.unique(self.labels)

        # --- CORRECCIÓN AQUÍ ---
        # Si la clase EVM_OFFICIAL tiene un atributo nfuse, úsalo.
        # Si no, definimos uno por defecto (ej. 4)
        num_to_fuse = getattr(self, 'nfuse', 4)

        fused = []
        for ulabel in ulabels:
            cls_mask = (self.labels == ulabel)
            # Ahora pasamos el num_to_fuse correctamente definido
            fused.append(fuse_prob_for_label(probs[cls_mask], num_to_fuse))

        fused = np.vstack(fused)
        max_ind = np.argmax(fused, axis=0)
        max_prob = fused[max_ind, np.arange(fused.shape[1])]
        pred_labels = ulabels[max_ind]

        final = np.array(pred_labels, dtype=object)
        final[max_prob < threshold] = "UUC"
        return final, fused

# =====================================================================
# SPLITS EXACTOS DEL ARTÍCULO
# =====================================================================

@dataclass
class OSRSplits:
    X_fit:np.ndarray; y_fit:np.ndarray
    X_val_closed:np.ndarray; y_val_closed:np.ndarray
    X_val_open:np.ndarray;   y_val_open:np.ndarray
    X_te:np.ndarray;         y_te:np.ndarray
    kkc_labels_fit:List[str]
    kkc_labels_all:List[str]

def create_splits_explicit(X, y, known_labels):
    known_labels = [str(k) for k in known_labels]

    all_labels = np.unique(y).tolist()

    # KKCs y UUCs
    idx_kkc = np.isin(y, known_labels)
    X_kkc, y_kkc = X[idx_kkc], y[idx_kkc]
    X_uuc, y_uuc = X[~idx_kkc], y[~idx_kkc]

    rng = np.random.default_rng(123)

    # 60% train / 40% test
    X_tr = []
    y_tr = []
    X_te_known = []
    y_te_known = []

    for c in known_labels:
        idx = np.where(y_kkc == c)[0]
        idx = rng.permutation(idx)

        ntrain = math.ceil(0.6 * len(idx))

        X_tr.append(X_kkc[idx[:ntrain]])
        y_tr.append(y_kkc[idx[:ntrain]])

        X_te_known.append(X_kkc[idx[ntrain:]])
        y_te_known.append(y_kkc[idx[ntrain:]])

    X_tr = np.vstack(X_tr)
    y_tr = np.concatenate(y_tr)

    # TEST = 40% KKCs + UUC reales
    X_te = np.vstack([np.vstack(X_te_known), X_uuc])
    y_te = np.concatenate([
        np.concatenate(y_te_known),
        y_uuc   # ← etiquetas reales
    ])

    # 2/3 KKCs para fitting
    n_fit = int(2/3 * len(known_labels))
    kkc_fit = known_labels[:n_fit]
    kkc_simulated = known_labels[n_fit:]

    idx_fit = np.isin(y_tr, kkc_fit)
    Xtf, ytf = X_tr[idx_fit], y_tr[idx_fit]
    Xtu, ytu = X_tr[~idx_fit], y_tr[~idx_fit]

    # 60% fitting / 40% closed-val
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

    # OPEN VAL = closed + simulated unknown (con etiquetas reales)
    X_valo = np.vstack([X_valc, Xtu])
    y_valo = np.concatenate([
        y_valc,
        ytu   # ← etiquetas reales
    ])

    return OSRSplits(
        X_fit, y_fit,
        X_valc, y_valc,
        X_valo, y_valo,
        X_te, y_te,
        kkc_fit,
        known_labels
    )


# =====================================================================
# Run MÉTODO
# =====================================================================
def save_hyperparams(path, method, params):
    """Guarda los parámetros en un archivo .json"""
    filename = f"{path}_best_params.json"
    with open(filename, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"Hiperparámetros guardados en: {filename}")

def load_hyperparams(path, method):
    """Carga los parámetros desde un archivo .json"""
    filename = f"{path}_{method}_best_params.json"
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None
def run_method(flagD,method, splits,openess):
    # Crea una instancia de la clase de métricas
    metrics = OSRMetrics()

    Xf,yf  = splits.X_fit, splits.y_fit
    Xvc,yvc = splits.X_val_closed, splits.y_val_closed
    Xvo,yvo = splits.X_val_open,   splits.y_val_open
    Xte,yte = splits.X_te, splits.y_te
    kkc_fit = splits.kkc_labels_fit
    kkc_all = splits.kkc_labels_all

    if flagD == 0:
        # ESCENARIO A: Solo Estandarización (Recomendado para probar primero)
        scaler = StandardScaler()
        Xf_s = scaler.fit_transform(Xf)
        Xvo_s = scaler.transform(Xvo)
        Xvc_s = scaler.transform(Xvc)
        Xte_s = scaler.transform(Xte)
    elif flagD == 1:
        # ESCENARIO C: Estandarización -> L2 (Híbrido)
        scaler = StandardScaler()
        Xf_s = l2_normalize(scaler.fit_transform(Xf))
        Xvc_s = l2_normalize(scaler.transform(Xvc))
        Xvo_s = l2_normalize(scaler.transform(Xvo))
        Xte_s = l2_normalize(scaler.transform(Xte))


    # Grid tuning C y gamma
    C_grid = [0.01,1, 10,20]#1vsset  0.01,1, 10,
    C_grid2 = [0.5, 1.0, 2.0]#[ 0.1, 1,10,100]#pivm
    C_grid3 = [0.1, 1,5, 10]#100, 500, 1000, wsvm
    nu=[0.02,0.01, 0.05]
    dt=[1e-4, 1e-3,1e-6]
    gamma_grid = [0.01, 0.1, 1,10]# 0.01, 0.1, 1,20,50    0.5, 1.0, 2.0
    gamma_grid2 = ["scale", 0.05, 0.1]#[0.1, 1, 10, 100]
    gamma_grid3 = [ "scale", 0.1,1,10]#[ 0.01, 0.1,10,100]#[1, 10, 50, 100]
    max_comp=[8, 12]
    weight=[0.05, 0.1]
    reg_covar=[1e-4, 1e-3]
    threshold_grid = np.linspace(0.01, 0.9, 50)
    bestF = -1
    best_f_val = -1
    best_model = None
    best_params = {}

    if method == "1vsset":
        for C in C_grid:
            for gamma in gamma_grid:
                model = OneVsSet_LIBSVM(C, gamma)
                model.fit(Xf_s, yf)
                model.tune(Xvo_s, yvo, kkc_fit)
                preds = model.predict(Xvo_s)
                F = metrics.micro_f1(yvo, preds, kkc_fit)

                if F > bestF:
                    bestF = F
                    best_model = model
                    print(C, gamma)
        # Test final
        preds_test = best_model.predict(Xte_s)
    elif method == "wsvm":
        #for n in nu:
        #for t in dt:
        for C in C_grid3:
            for gamma in gamma_grid3:
                model = WSVM(C, gamma,openess )
                model.fit(Xf_s, yf, Xvo_s, yvo, kkc_fit)
                #model = WSVMModel(C,openess,gamma,n,t)
                #model.fit(Xf_s, yf)
                preds_xvo = model.predict(Xvo_s)
                F = metrics.micro_f1(yvo, preds_xvo, kkc_fit)

                if F > bestF:
                    bestF = F
                    best_model = model
                    #print('best  ',C,gamma)
        # Test final
        preds_test = best_model.predict(Xte_s)
        total = len(preds_test)
        uuc_count = np.sum(preds_test == "UUC")
        print(f"Porcentaje de rechazo (UUC): {uuc_count / total * 100:.2f}%")

        # PRUEBA MANUAL:
        # Prueba con un R bajísimo (debería aceptar casi todo -> Rechazo bajo)
        p1 = best_model.predict_inner(Xte_s, eps=1e-5, R=0.0001)
        #print(f"Rechazo con R=0.0001: {np.sum(p1 == 'UUC') / len(p1)}")

        # Prueba con un R altísimo (debería rechazar casi todo -> Rechazo alto)
        p2 = best_model.predict_inner(Xte_s, eps=1e-5, R=1000.0)
        #print(f"Rechazo con R=1000: {np.sum(p2 == 'UUC') / len(p2)}")
    elif method == "wosvm":
        for n in nu:
            for t in dt:
                for C in C_grid3:
                    for gamma in gamma_grid3:
                        model = WOSVMModel(C,openess,gamma,n,t)
                        model.fit(Xf_s, yf)
                        preds_xvo = model.predict(Xvo_s)
                        F = metrics.micro_f1(yvo, preds_xvo, kkc_fit)

                        if F > bestF:
                            bestF = F
                            best_model = model
                            #print('best C, gamma ,nu, dt: ',C,gamma,n,t)
        # Test final
        preds_test = best_model.predict(Xte_s)
        total = len(preds_test)
        uuc_count = np.sum(preds_test == "UUC")
        print(f"Porcentaje de rechazo (UUC): {uuc_count / total * 100:.2f}%")

    elif method == "pisvm":
        for C in C_grid2:
            for gamma in gamma_grid2:
                model = PISVM_LIBSVM(C, gamma)
                model.fit(Xf_s, yf, Xvo_s, yvo, kkc_fit)
                preds = model.predict(Xvo_s)
                F = metrics.micro_f1(yvo, preds, kkc_fit)

                if F > bestF:
                    bestF = F
                    best_model = model
                    #print('best  ',C, gamma)
        # Test final
        preds_test = best_model.predict(Xte_s)
    elif method=="srosr":
        model=SROSR()#(sparsity=3)
        model.fit(Xf_s,yf,Xvo_s,yvo,kkc_fit)
        preds_test = model.predict(Xte_s)
    elif method=="osnn":
        m=OSNN(k=5)
        m.fit(Xf_s,yf)
        ratios, nn_labels = m.ratio(Xvo_s)
        #print("\nk: ", m.k)
        #print("ratio min:", ratios.min())
        #print("ratio max:", ratios.max())
        #print("ratio mean:", ratios.mean())
        known_mask = np.isin(yvo, kkc_fit)
        #print("known ratio mean:", ratios[known_mask].mean())
        #print("unknown ratio mean:", ratios[~known_mask].mean())

        best_sep = -1
        best_t = 0
        for t in np.linspace(ratios.min(), ratios.max(), 300):
            pred = np.where(ratios <= t, nn_labels, "UUC")
            unknown_rate = np.mean(pred == "UUC")
            if unknown_rate < 0.01:
                continue
            sep = np.mean(ratios[known_mask] <= t) + np.mean(ratios[~known_mask] > t)
            if sep > best_sep:
                best_sep = sep
                best_t = t
        r = m.ratio(Xvo_s)
        print(r[0].min(), r[0].max(), r[0].mean())
        preds_test = m.predict(Xte_s, best_t)
        known_mask2 = np.isin(yte, kkc_fit)
        acc_known = np.mean(preds_test[known_mask2] == yte[known_mask2])
        acc_unknown = np.mean(preds_test[~known_mask2] == "UUC")
        print("known acc:", acc_known)
        print("unknown acc:", acc_unknown)
        print("unknown predicted:", np.sum(preds_test == "UUC"))
        print("total test:", len(preds_test))
        print('best_t', best_t)
    elif method == "evm":
        bestF = -1
        best_model = None
        best_tau = 0.5
        #PROBAR CAMBIANDO ESTO
        for tau in [0.1]:#np.linspace(0.1, 0.9, 9):#[0.1, 0.5, 0.9]
            model = EVM_OFFICIAL(tailsize=50, cover_threshold=0.5, nfuse=4, margin_scale=0.5)
            model.fit(Xf, yf)
            model.reduce_model()
            preds, probs = model.predict(Xvo, threshold=tau)
            F = metrics.micro_f1(yvo, preds, kkc_fit)
            if F > bestF:
                bestF = F
                best_model = model
                best_tau = tau
            print('EVM Best Params',bestF,best_tau)
        preds_test, probs = best_model.predict(Xte, threshold=best_tau)
    elif method== 'cdosr':
        for m in max_comp :
            for w in weight :
                for r in reg_covar:
                    model = CDOSRModel(m,w,r)
                    model.fit(Xf_s, yf, Xvo_s, yvo, kkc_fit)

                    best_thr_for_this_model = 0.2
                    best_f_for_this_thr = -1

                    for thr in threshold_grid:
                        preds_val = model.predict(Xvo_s, thr=thr)
                        f_val = metrics.micro_f1(yvo, preds_val, kkc_fit)

                        if f_val > best_f_for_this_thr:
                            best_f_for_this_thr = f_val
                            best_thr_for_this_model = thr

                    # 3. Comparar este modelo (con su mejor umbral) contra el mejor global
                    if best_f_for_this_thr > best_f_val:
                        best_f_val = best_f_for_this_thr
                        best_model = model
                        best_params = {'m': m, 'w': w, 'r': r, 'thr': best_thr_for_this_model}

        print(f"Mejores Parámetros: {best_params} | Val Micro-F: {best_f_val:.4f}")

        # 4. EVALUACIÓN FINAL en TEST (Solo una vez, con el mejor modelo y mejor umbral)
        preds_test = best_model.predict(Xte_s, thr=best_params['thr'])
        final_f_test = metrics.micro_f1(yte, preds_test, kkc_fit)
        print(f"Resultado Final en Test: {final_f_test:.4f}")
    else:
        raise ValueError("Método desconocido")

    F = 100.0 * metrics.micro_f1(yte, preds_test, kkc_fit)
    return F

def run_EVL_methodold(XTtrain, Ytrain,Xval,yval,kkc_fit,Xtest,Ytest,method,openess):
    # Crea una instancia de la clase de métricas
    metrics = OSRMetrics()

    # Grid tuning C y gamma
    C_grid = [0.001, 0.01, 0.1]
    C_grid2 = [0.5, 1.0, 2.0]
    C_grid3 = [0.1,0.5, 1.0, 2.0]
    nu=[0.02,0.01, 0.05]
    dt=[1e-4, 1e-3,1e-6]
    gamma_grid =  [0.0001, 0.001, 0.01]
    gamma_grid2 = ["scale", 0.05, 0.1]
    gamma_grid3 = [ "scale", 0.01,0.05, 0.1]
    max_comp=[8, 12]
    weight=[0.05, 0.1]
    reg_covar=[1e-4, 1e-3]
    threshold_grid = np.linspace(0.01, 0.9, 50)
    bestF = -1
    best_f_val = -1
    best_model = None
    best_params = {}

    if method == "1vsset":
        for C in C_grid:
            for gamma in gamma_grid:
                model = OneVsSet_LIBSVMEVL(C, gamma)
                model.fit(XTtrain, Ytrain)
                model.tune(Xval, yval, kkc_fit)
                preds = model.predict(Xval)
                F = metrics.micro_f1(yval, preds, kkc_fit)

                if F > bestF:
                    bestF = F
                    best_model = model
                    #print('BEST ',bestF,C,gamma)
        # Test final
        preds_test = best_model.predict(Xtest)
        print("Distribución de predicciones:")
        unique, counts = np.unique(preds_test, return_counts=True)
        print(dict(zip(unique, counts)))
    elif method == "wsvm":
        #for n in nu:
        #for t in dt:
        for C in C_grid3:
            for gamma in gamma_grid3:
                model = WSVMEVL(C, gamma,openess )
                model.fit(XTtrain, Ytrain, Xval, yval, kkc_fit)
                #model = WSVMModel(C,openess,gamma,n,t)
                #model.fit(Xf_s, yf)
                preds_xvo = model.predict(Xval)
                F = metrics.micro_f1(yval, preds_xvo, kkc_fit)
                unique_preds = np.unique(yval)

                # Si el modelo solo predice una clase conocida o solo UUC, penaliza fuerte
                if len(unique_preds) < len(kkc_fit):
                    F -= 0.2  # Penalización del 20% del F1

                if F > bestF:
                    bestF = F
                    best_model = model
                    #print('best  ',C,gamma)#n,t
        # Test final
        preds_test = best_model.predict(Xtest)
        total = len(preds_test)
        uuc_count = np.sum(preds_test == "UUC")
        print(f"Porcentaje de rechazo (UUC): {uuc_count / total * 100:.2f}%")

        # PRUEBA MANUAL:
        # Prueba con un R bajísimo (debería aceptar casi todo -> Rechazo bajo)
        p1 = best_model.predict_inner(Xtest, eps=1e-5, R=0.0001)
        #print(f"Rechazo con R=0.0001: {np.sum(p1 == 'UUC') / len(p1)}")

        # Prueba con un R altísimo (deberíst, eps=1e-5, R=1000.0)
        #print(f"Rechazo con R=1000: {np.sum(p2 == 'UUC') / len(p2)}")
    elif method == "wosvm":
        for n in nu:
            for t in dt:
                for C in C_grid3:
                    for gamma in gamma_grid3:
                        model = WOSVMModel(C,openess,gamma,n,t)
                        model.fit(XTtrain, Ytrain)
                        preds_xvo = model.predict(Xval)
                        F = metrics.micro_f1(yval, preds_xvo, kkc_fit)

                        if F > bestF:
                            bestF = F
                            best_model = model
                            #print('best C, gamma ,nu, dt: ',C,gamma,n,t)
        # Test final
        preds_test = best_model.predict(Xtest)
        total = len(preds_test)
        uuc_count = np.sum(preds_test == "UUC")
        print(f"Porcentaje de rechazo (UUC): {uuc_count / total * 100:.2f}%")

    elif method == "pisvm":
        for C in C_grid2:
            for gamma in gamma_grid2:
                model = PISVM_LIBSVM(C, gamma)
                model.fit(XTtrain, Ytrain, Xval, yval, kkc_fit)
                preds = model.predict(Xval)
                F = metrics.micro_f1(yval, preds, kkc_fit)

                if F > bestF:
                    bestF = F
                    best_model = model
                    #print('best  ',C, gamma)
        # Test final
        preds_test = best_model.predict(Xtest)
    elif method=="srosr":
        model=SROSR()#(sparsity=3)
        model.fit(XTtrain,Ytrain,Xval,yval,kkc_fit)
        preds_test = model.predict(Xtest)
    elif method=="osnn":
        m=OSNNEVL(k=5)
        m.fit(XTtrain,Ytrain)
        ratios, nn_labels = m.ratio(Xval)
        #print("\nk: ", m.k)
        #print("ratio min:", ratios.min())
        #print("ratio max:", ratios.max())
        #print("ratio mean:", ratios.mean())
        known_mask = np.isin(yval, kkc_fit)
        #print("known ratio mean:", ratios[known_mask].mean())
        #print("unknown ratio mean:", ratios[~known_mask].mean())

        best_sep = -1
        best_t = 0
        for t in np.linspace(ratios.min(), ratios.max(), 300):
            pred = np.where(ratios <= t, nn_labels, "UUC")
            unknown_rate = np.mean(pred == "UUC")
            if unknown_rate < 0.01:
                continue
            sep = np.mean(ratios[known_mask] <= t) + np.mean(ratios[~known_mask] > t)
            if sep > best_sep:
                best_sep = sep
                best_t = t
        r = m.ratio(Xval)
        print(r[0].min(), r[0].max(), r[0].mean())
        preds_test = m.predict(Xtest, best_t)
        known_mask2 = np.isin(Ytest, kkc_fit)
        acc_known = np.mean(preds_test[known_mask2] == Ytest[known_mask2])
        acc_unknown = np.mean(preds_test[~known_mask2] == "UUC")
        print("known acc:", acc_known)
        print("unknown acc:", acc_unknown)
        print("unknown predicted:", np.sum(preds_test == "UUC"))
        print("total test:", len(preds_test))
        print('best_t', best_t)
    elif method == "evm":
        bestF = -1
        best_model = None
        best_tau = 0.5
        #PROBAR CAMBIANDO ESTO
        for tau in [0.1]:#np.linspace(0.1, 0.9, 9):#[0.1, 0.5, 0.9]
            model = EVM_OFFICIAL(tailsize=50, cover_threshold=0.5, nfuse=4, margin_scale=0.5)
            model.fit(XTtrain, Ytrain)
            model.reduce_model()
            preds, probs = model.predict(Xval, threshold=tau)
            F = metrics.micro_f1(yval, preds, kkc_fit)
            if F > bestF:
                bestF = F
                best_model = model
                best_tau = tau
            print('EVM Best params for tau ',bestF,best_tau)
        preds_test, probs = best_model.predict(Xtest, threshold=best_tau)
    elif method== 'cdosr':
        for m in max_comp :
            for w in weight :
                for r in reg_covar:
                    model = CDOSRModel(m,w,r)
                    model.fit(XTtrain, Ytrain, Xval, yval, kkc_fit)

                    best_thr_for_this_model = 0.2
                    best_f_for_this_thr = -1

                    for thr in threshold_grid:
                        preds_val = model.predict(Xval, thr=thr)
                        f_val = metrics.micro_f1(yval, preds_val, kkc_fit)

                        if f_val > best_f_for_this_thr:
                            best_f_for_this_thr = f_val
                            best_thr_for_this_model = thr

                    # 3. Comparar este modelo (con su mejor umbral) contra el mejor global
                    if best_f_for_this_thr > best_f_val:
                        best_f_val = best_f_for_this_thr
                        best_model = model
                        best_params = {'m': m, 'w': w, 'r': r, 'thr': best_thr_for_this_model}

        print(f"Mejores Parámetros: {best_params} | Val Micro-F: {best_f_val:.4f}")

        # 4. EVALUACIÓN FINAL en TEST (Solo una vez, con el mejor modelo y mejor umbral)
        preds_test = best_model.predict(Xtest, thr=best_params['thr'])
        final_f_test = metrics.micro_f1(Ytest, preds_test, kkc_fit)
        print(f"Resultado Final en Test: {final_f_test:.4f}")
    else:
        raise ValueError("Método desconocido")

    F = 100.0 * metrics.micro_f1(Ytest, preds_test, kkc_fit)
    auc =  metrics.accuracy(Ytest, preds_test, kkc_fit)
    youd = metrics.youden_index_evl(Ytest, preds_test, kkc_fit)
    return F,auc,youd, preds_test


def run_EVL_method(XTtrain, Ytrain, Xval, yval, kkc_fit, Xtest, Ytest, method, openess, params=None):
    # Crea una instancia de la clase de métricas
    metrics = OSRMetrics()

    # Grids de búsqueda (Solo se usarán si params es None)
    C_grid = [0.001, 0.01, 0.1]
    C_grid2 = [0.5, 1.0, 2.0]
    C_grid3 = [0.1, 0.5, 1.0, 2.0]
    nu = [0.02, 0.01, 0.05]
    dt = [1e-4, 1e-3, 1e-6]
    gamma_grid = [0.0001, 0.001, 0.01]
    gamma_grid2 = ["scale", 0.05, 0.1]
    gamma_grid3 = ["scale", 0.01, 0.05, 0.1]
    max_comp = [8, 12]
    weight = [0.05, 0.1]
    reg_covar = [1e-4, 1e-3]
    threshold_grid = np.linspace(0.01, 0.9, 50)

    bestF = -1
    best_model = None
    best_params = params  # Inicializamos con los params recibidos

    if method == "1vsset":
        if params is not None:
            # MODO RÁPIDO: Usar parámetros fijos
            best_model = OneVsSet_LIBSVMEVL(params['C'], params['gamma'])
            best_model.fit(XTtrain, Ytrain)
            best_model.tune(Xval, yval, kkc_fit)
        else:
            # MODO TUNING: Grid Search
            for C in C_grid:
                for gamma in gamma_grid:
                    model = OneVsSet_LIBSVMEVL(C, gamma)
                    model.fit(XTtrain, Ytrain)
                    model.tune(Xval, yval, kkc_fit)
                    preds = model.predict(Xval)
                    F = metrics.micro_f1(yval, preds, kkc_fit)
                    if F > bestF:
                        bestF = F
                        best_model = model
        preds_test = best_model.predict(Xtest)

    elif method == "wsvm":
        if params is not None:
            best_model = WSVMEVL(params['C'], params['gamma'], openess)
            best_model.fit(XTtrain, Ytrain, Xval, yval, kkc_fit)
        else:
            for C in C_grid3:
                for gamma in gamma_grid3:
                    model = WSVMEVL(C, gamma, openess)
                    model.fit(XTtrain, Ytrain, Xval, yval, kkc_fit)
                    preds_xvo = model.predict(Xval)
                    F = metrics.micro_f1(yval, preds_xvo, kkc_fit)
                    unique_preds = np.unique(yval)
                    if len(unique_preds) < len(kkc_fit): F -= 0.2
                    if F > bestF:
                        bestF = F
                        best_model = model
        preds_test = best_model.predict(Xtest)

    elif method == "wosvm":
        if params is not None:
            best_model = WOSVMModel(params['C'], openess, params['gamma'], params['nu'], params['t'])
            best_model.fit(XTtrain, Ytrain)
        else:
            for n in nu:
                for t in dt:
                    for C in C_grid3:
                        for gamma in gamma_grid3:
                            model = WOSVMModel(C, openess, gamma, n, t)
                            model.fit(XTtrain, Ytrain)
                            preds_xvo = model.predict(Xval)
                            F = metrics.micro_f1(yval, preds_xvo, kkc_fit)
                            if F > bestF:
                                bestF = F
                                best_model = model
        preds_test = best_model.predict(Xtest)

    elif method == "pisvm":
        if params is not None:
            best_model = PISVM_LIBSVM(params['C'], params['gamma'])
            best_model.fit(XTtrain, Ytrain, Xval, yval, kkc_fit)
        else:
            for C in C_grid2:
                for gamma in gamma_grid2:
                    model = PISVM_LIBSVM(C, gamma)
                    model.fit(XTtrain, Ytrain, Xval, yval, kkc_fit)
                    preds = model.predict(Xval)
                    F = metrics.micro_f1(yval, preds, kkc_fit)
                    if F > bestF:
                        bestF = F
                        best_model = model
        preds_test = best_model.predict(Xtest)


    elif method == "srosr":
        if params is not None:
            # Instanciamos con los parámetros de arquitectura (si hubiera)
            best_model = SROSR(variance_keep=params.get('variance_keep', 0.95))
            best_model.fit(XTtrain, Ytrain)
        else:
            # Tuning manual si no hay params
            model = SROSR()
            model.fit(XTtrain, Ytrain, Xval, yval, kkc_fit)  # Aquí el fit hace el tuning interno
            best_model = model
            params = {'tau': model.tau}  # Guardamos el tau encontrado
        # PREDICCIÓN: Extraemos el tau de params para que no use el 0.5 por defecto
        tau_val = params.get('tau', 0.5) if params else 0.5
        preds_test = best_model.predict(Xtest, tau=tau_val)
    elif method == "osnn":
        m = OSNNEVL(k=params['k'] if params else 5)
        m.fit(XTtrain, Ytrain)
        if params is not None:
            best_t = params['best_t']
        else:
            ratios, nn_labels = m.ratio(Xval)
            known_mask = np.isin(yval, kkc_fit)
            best_sep = -1
            best_t = 0.2
            for t in np.linspace(ratios.min(), ratios.max(), 300):
                pred = np.where(ratios <= t, nn_labels, "UUC")
                if np.mean(pred == "UUC") < 0.01: continue
                sep = np.mean(ratios[known_mask] <= t) + np.mean(ratios[~known_mask] > t)
                if sep > best_sep:
                    best_sep = sep
                    best_t = t
        preds_test = m.predict(Xtest, best_t)

    elif method == "evm":
        if params is not None:
            # MODO RÁPIDO: Usar parámetros optimizados
            # Importante: Usamos params['tailsize'] y params['tau']
            best_tau = params['tau']
            t_size = params['tailsize']
            best_model = EVM_OFFICIAL(tailsize=t_size, cover_threshold=0.5, nfuse=1, margin_scale=0.5)
            best_model.fit(XTtrain, Ytrain)
            best_model.reduce_model()
        else:
            # MODO TUNING
            best_tau = 0.5
            best_t = 50
            for t in [20, 30, 50]:
                for tau in [0.1]:
                    model = EVM_OFFICIAL(tailsize=t, cover_threshold=0.5, nfuse=1, margin_scale=0.5)
                    model.fit(XTtrain, Ytrain)
                    model.reduce_model()
                    preds, probs = model.predict(Xval, threshold=tau)
                    F = metrics.micro_f1(yval, preds, kkc_fit)

                    if F > bestF:
                        bestF = F
                        best_model = model
                        best_tau = tau
                        best_t = t  # GUARDAR EL T QUE FUNCIONA

            best_params = {'tau': best_tau, 'tailsize': best_t}
        try:
            res = best_model.predict(Xtest, threshold=best_tau)
            preds_test = res[0] if isinstance(res, tuple) else res
        except ValueError as e:
            print(f"Error crítico en EVM predict: {e}. Devolviendo predicciones vacías para este batch.")
            # En caso de error, rellenamos con 'UUC' para que el loop continúe
            preds_test = np.full(len(Xtest), "UUC")
            # Ejecución del test
        # IMPORTANTE: EVM devuelve (preds, probs), necesitamos solo preds
        res = best_model.predict(Xtest, threshold=best_tau)
        preds_test = res[0] if isinstance(res, tuple) else res

    elif method == 'cdosr':
        if params is not None:
            best_model = CDOSRModel(params['m'], params['w'], params['r'])
            best_model.fit(XTtrain, Ytrain, Xval, yval, kkc_fit)
            preds_test = best_model.predict(Xtest, thr=params['thr'])
        else:
            best_f_val = -1
            for m in max_comp:
                for w in weight:
                    for r in reg_covar:
                        model = CDOSRModel(m, w, r)
                        model.fit(XTtrain, Ytrain, Xval, yval, kkc_fit)
                        best_thr_for_this_model = 0.2
                        best_f_for_this_thr = -1
                        for thr in threshold_grid:
                            preds_val = model.predict(Xval, thr=thr)
                            f_val = metrics.micro_f1(yval, preds_val, kkc_fit)
                            if f_val > best_f_for_this_thr:
                                best_f_for_this_thr = f_val
                                best_thr_for_this_model = thr
                        if best_f_for_this_thr > best_f_val:
                            best_f_val = best_f_for_this_thr
                            best_model = model
                            best_params = {'m': m, 'w': w, 'r': r, 'thr': best_thr_for_this_model}
            preds_test = best_model.predict(Xtest, thr=best_params['thr'])
    else:
        raise ValueError("Método desconocido")

    F = 100.0 * metrics.micro_f1(Ytest, preds_test, kkc_fit)
    auc = metrics.accuracy(Ytest, preds_test, kkc_fit)
    youd = metrics.youden_index_evl(Ytest, preds_test, kkc_fit)

    return F, auc, youd, preds_test

def tune_hyperparameters(XTtrain, Ytrain, Xval, yval,  Xtest, Ytest, kkc_fit,method, openess):
    metrics = OSRMetrics()

    # Grids de búsqueda
    C_grid = [0.001, 0.01, 0.1]
    C_grid2 = [0.5, 1.0, 2.0]
    C_grid3 = [0.1, 0.5, 1.0, 2.0]
    nu = [0.02, 0.01, 0.05]
    dt = [1e-4, 1e-3, 1e-6]
    gamma_grid = [0.0001, 0.001, 0.01]
    gamma_grid2 = ["scale", 0.05, 0.1]
    gamma_grid3 = ["scale", 0.01, 0.05, 0.1]
    max_comp = [8, 12]
    weight = [0.05, 0.1]
    reg_covar = [1e-4, 1e-3]
    threshold_grid = np.linspace(0.01, 0.9, 50)

    bestF = -1
    best_model = None
    best_params = {}  # <--- Este es el diccionario clave

    if method == "1vsset":
        for C in C_grid:
            for gamma in gamma_grid:
                model = OneVsSet_LIBSVMEVL(C, gamma)
                model.fit(XTtrain, Ytrain)
                model.tune(Xval, yval, kkc_fit)
                preds = model.predict(Xval)
                F = metrics.micro_f1(yval, preds, kkc_fit)
                if F > bestF:
                    bestF = F
                    best_model = model
                    best_params = {'C': C, 'gamma': gamma}  # GUARDAR

    elif method == "wsvm":
        for C in C_grid3:
            for gamma in gamma_grid3:
                model = WSVMEVL(C, gamma, openess)
                model.fit(XTtrain, Ytrain, Xval, yval, kkc_fit)
                preds_xvo = model.predict(Xval)
                F = metrics.micro_f1(yval, preds_xvo, kkc_fit)
                unique_preds = np.unique(yval)
                if len(unique_preds) < len(kkc_fit):
                    F -= 0.2
                if F > bestF:
                    bestF = F
                    best_model = model
                    best_params = {'C': C, 'gamma': gamma}  # GUARDAR

    elif method == "wosvm":
        for n in nu:
            for t in dt:
                for C in C_grid3:
                    for gamma in gamma_grid3:
                        model = WOSVMModel(C, openess, gamma, n, t)
                        model.fit(XTtrain, Ytrain)
                        preds_xvo = model.predict(Xval)
                        F = metrics.micro_f1(yval, preds_xvo, kkc_fit)
                        if F > bestF:
                            bestF = F
                            best_model = model
                            best_params = {'C': C, 'gamma': gamma, 'nu': n, 't': t}  # GUARDAR

    elif method == "pisvm":
        for C in C_grid2:
            for gamma in gamma_grid2:
                model = PISVM_LIBSVM(C, gamma)
                model.fit(XTtrain, Ytrain, Xval, yval, kkc_fit)
                preds = model.predict(Xval)
                F = metrics.micro_f1(yval, preds, kkc_fit)
                if F > bestF:
                    bestF = F
                    best_model = model
                    best_params = {'C': C, 'gamma': gamma}  # GUARDAR

    elif method == "srosr":
        model = SROSR()
        model.fit(XTtrain, Ytrain)  # fit rápido

        # Obtener scores crudos para tunear tau
        # Implementamos un pequeño método auxiliar o usamos un loop
        raw_scores = []
        for i in range(len(Xval)):
            raw_scores.append(model._score(Xval[i]))

        scores_only = np.array([p[1] for p in raw_scores])

        # Normalización para detectar Cold Start
        yv_s, _, kkc_s = metrics._unify_types(yval, [], kkc_fit)
        has_unknowns = np.any(~np.isin(yv_s, kkc_s))

        best_f1 = -1
        best_tau = 0.5

        if has_unknowns:
            # Tuning estándar basado en Micro-F1
            for tau in np.linspace(0.001, 2.0, 500):
                preds = np.array([p[0] if p[1] >= tau else "UUC" for p in raw_scores])
                f1 = metrics.micro_f1(yval, preds, kkc_fit)
                if f1 > best_f1:
                    best_f1 = f1
                    best_tau = tau
        else:
            # Cold Start: Usamos la estadística de los scores (percentil 10)
            best_tau = np.percentile(scores_only, 10)

        best_model = model
        best_params = {'tau': best_tau}

    elif method == "osnn":
        m = OSNNEVL(k=5)
        m.fit(XTtrain, Ytrain)
        ratios, nn_labels = m.ratio(Xval)
        known_mask = np.isin(yval, kkc_fit)

        # Verificamos si existen realmente muestras desconocidas en el set de validación
        has_unknowns = np.any(~known_mask)

        best_sep = -1
        best_t = 0.2
        # Generamos el rango de umbrales
        t_range = np.linspace(ratios.min(), ratios.max(), 300)

        for t in t_range:
            pred = np.where(ratios <= t, nn_labels, "UUC")

            if has_unknowns:
                # --- CASO NORMAL: Hay desconocidos, usamos la lógica de separación ---
                unknown_rate = np.mean(pred == "UUC")
                if unknown_rate < 0.01:
                    continue
                    # Separación = (Aciertos Conocidos + Aciertos Desconocidos)
                sep = np.mean(ratios[known_mask] <= t) + np.mean(ratios[~known_mask] > t)
            else:
                # --- CASO COLD START: Solo hay conocidos ---
                # Maximizamos la tasa de aceptación de los conocidos
                sep = np.mean(ratios[known_mask] <= t)
                # Penalizamos si aceptamos DEMASIADO (para evitar thr=max(ratios))
                if np.mean(pred == "UUC") < 0.001:
                    sep -= 0.1

            if sep > best_sep:
                best_sep = sep
                best_t = t
        best_model = m
        best_params = {'k': 5, 'best_t': best_t}  # GUARDAR

    elif method == "evm":
        best_tail = 10
        best_tau = 0.5
        for t in [20,30,50]:
            for tau in [0.1]:
                model = EVM_OFFICIAL(tailsize=t, cover_threshold=0.5, nfuse=1, margin_scale=0.5)
                model.fit(XTtrain, Ytrain)
                model.reduce_model()
                preds, probs = model.predict(Xval, threshold=tau)
                F = metrics.micro_f1(yval, preds, kkc_fit)
                if F > bestF:
                    bestF = F
                    best_model = model
                    best_tau = tau
                    best_tail = t
        best_params = {'tau': best_tau, 'tailsize': best_tail}  # GUARDAR

    elif method == 'cdosr':
        best_f_val = -1
        for m in max_comp:
            for w in weight:
                for r in reg_covar:
                    model = CDOSRModel(m, w, r)
                    model.fit(XTtrain, Ytrain, Xval, yval, kkc_fit)
                    best_thr_for_this_model = 0.2
                    best_f_for_this_thr = -1
                    for thr in threshold_grid:
                        preds_val = model.predict(Xval, thr=thr)
                        f_val = metrics.micro_f1(yval, preds_val, kkc_fit)
                        if f_val > best_f_for_this_thr:
                            best_f_for_this_thr = f_val
                            best_thr_for_this_model = thr
                    if best_f_for_this_thr > best_f_val:
                        best_f_val = best_f_for_this_thr
                        best_model = model
                        best_params = {'m': m, 'w': w, 'r': r, 'thr': best_thr_for_this_model}  # GUARDAR
    else:

        raise ValueError("Método desconocido")

    # =====================================================================
    # PREDICCIÓN FINAL CORREGIDA (DESPACHO POR MÉTODO)
    # =====================================================================
    if method == 'osnn':
        # OSNN requiere el umbral best_t como segundo argumento posicional
        preds_test = best_model.predict(Xtest, best_params['best_t'])

    elif method == 'evm':
        # EVM devuelve una tupla (preds, probs) y usa el keyword 'threshold'
        preds_test, _ = best_model.predict(Xtest, threshold=best_params['tau'])

    elif method == 'cdosr':
        # CDOSR usa el keyword 'thr'
        preds_test = best_model.predict(Xtest, thr=best_params['thr'])

    elif method == 'srosr':
        preds_test = best_model.predict(Xtest, tau=best_params['tau'])
    else:
        # Para 1vsset, wsvm, wosvm, pisvm, srosr que usan .predict(X) simple
        preds_test = best_model.predict(Xtest)

    F_test = 100.0 * metrics.micro_f1(Ytest, preds_test, kkc_fit)
    auc_test = metrics.accuracy(Ytest, preds_test, kkc_fit)
    youd_test = metrics.youden_index_evl(Ytest, preds_test, kkc_fit)
    return best_params

def read_partition_log(log_path):
    known_lists = []
    with open(log_path, "r") as f:
        for line in f:
            if "known_class" in line:
                nums = re.findall(r"\d+", line.split("known_class")[1])
                nums = [int(x) for x in nums]
                known_lists.append(nums)
    return known_lists



