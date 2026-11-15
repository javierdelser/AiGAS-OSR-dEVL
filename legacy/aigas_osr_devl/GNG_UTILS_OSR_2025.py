# The part of the code corresponding to DensityPeakCluster is takeb from: https://github.com/lanbing510/DensityPeakCluster/tree/master?tab=MIT-1-ov-file
# Licencia MIT
# Copyright (c) 2014 Jason

import os
# neupy solo usa Theano al importarse (la GNG es numpy puro): sin BLAS ni
# compilación C, que en Linux evita fallos de g++. Se respeta THEANO_FLAGS si ya existe.
os.environ.setdefault('THEANO_FLAGS', 'blas.ldflags=,cxx=')

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger("theano").setLevel(logging.ERROR)
logging.getLogger("theano.tensor.blas").setLevel(logging.ERROR)

import copy
from scipy.spatial import distance,distance_matrix
from scipy.optimize import linear_sum_assignment
import math
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
import itertools
from itertools import combinations,permutations
from collections import defaultdict
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from osr_utils import OSRMetrics
try:
    import pydevd
except ImportError:
    pass
else:
    dbg = pydevd.get_global_debugger()
    if dbg is not None:
        dbg.init_matplotlib_support()

from neupy import algorithms
from sklearn.cluster import KMeans
from itertools import groupby
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import shapiro,levene,mannwhitneyu,ttest_ind
from alpha_shapes import Alpha_Shaper, plot_alpha_shape
from shapely.geometry import Polygon
from statistics import mean, median, mode
from collections import Counter

import settings_osr
settings_osr.init()
path=settings_osr.path

SILHOUTE_TOLERANCE=0.12
angulo_extra=0.002
tolerancia_angulo=0.03
tolerancia_ROTACION=0.1
tolerancia_traslacion_comparacion=0.1
tolerancia_traslacion = 0.5
TOLERANCE=6.0e-2
TOLERANCE_GRAPH=1.0e-5
TOLERANCE_ZONA=0.15
num_vecinos_entropia=5
num_vecinos=13
TOLERANCE_ENTROPIA_CAMBIO=0.75
TOL=0.07
TOL_num_cl=0.9

MAX_ITERATIONS = 300
INITIALIZE_CLUSTERS = 'k-means++'
CONVERGENCE_TOLERANCE = 0.0001
NUM_THREADS = 8

colors=['r','b','g','darkorange','lime','c','m','y','fuchsia','darkviolet']
colors_trasladados=['turquoise','chocolate','blue','palevioletred','lightpink','red']
colors_SINGULARES = ['green', 'lightcoral','cyan','crimson','mediumslateblue','m']
colors_TEST = ['deepskyblue', 'orange','slategray','fuchsia','darkviolet','y']

coordenadas=['X','Y','Z','T','E','I','J','K','H','W']
coordenadas=['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','C15','C16','C17','C18','C19','C20']

def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)
def flatten(l):
    return [item for sublist in l for item in sublist]
def centeroidpython(data):
    dim_c = data.shape[1]
    res = []
    devolver = []
    for d in range(dim_c):
        res.append([])
    res[:dim_c] = zip(*data)

    l = len(res[0])
    for d in range(dim_c):
        devolver.append(sum(res[d]) / l)

    return devolver

def obtencion_k(NUM):
    tam=NUM
    raiz=np.sqrt(tam)
    if (raiz%2==0):
        raiz=raiz-1
    k_elegida=raiz
    k_elegida=int(k_elegida)
    return k_elegida

def obtencion_centroide(n_c,lista_points,etiquetas_puntos):
    contornos = [[] for p in range(n_c)]

    name_centr = str('')
    centroids=[]

    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    for c in range(n_c):
        points = lista_points[np.array(etiquetas_puntos) == c]
        shaper = Alpha_Shaper(points)
        lpha_opt, alpha_shape = shaper.optimize()

        if alpha_shape.type == 'MultiPolygon':
            lpha_opt1=lpha_opt
            alpha_shape=shaper.get_shape(alpha=lpha_opt1)
            poly = shaper.get_shape(alpha=lpha_opt1)
            while alpha_shape.type=='MultiPolygon':
                lpha_opt1=lpha_opt1-(lpha_opt1/4)
                alpha_shape = shaper.get_shape(alpha=lpha_opt1)
                poly = shaper.get_shape(alpha=lpha_opt1)
        else:
            poly = shaper.get_shape(alpha=lpha_opt)

        # Extract the point values that define the perimeter of the polygon
        coordinates_array = np.asarray(poly.exterior.coords)
        contornos[c]=coordinates_array

        polygon = Polygon(poly)
        #print(polygon.centroid)
        centroids.append(polygon.centroid)

        axes.scatter(*zip(*points), alpha=0.35)
        plot_alpha_shape(axes, poly, colors[c + n_c])
        round(lpha_opt, 2)

    fig.suptitle(name_centr, fontsize=10)
    plt.tight_layout()

    path = settings_osr.path
    path = path + 'centr' + '/'

    #fig.savefig(path + 'img' + '_' +  '_.jpg')
    plt.show(block=False)
    plt.close('all')

    array_posiciones = np.array([[p.x, p.y] for p in centroids])

    return array_posiciones,contornos

def METRICAS_CONFIANZA():
    return 1

def tratamiento_previo_dataset(file_n,xfeatures,ylabels,dim,max,volumen_inicial,volumen_batch,bool_scaler,metodo):
    xdef=[]
    shape_original=xfeatures.shape
    if bool_scaler==True:
        scaler = StandardScaler()
        xfeatures = scaler.fit_transform(xfeatures)

    for i in range(max):
        if i==0:
            Xant = xfeatures[0:volumen_inicial]
            yant = ylabels[0:volumen_inicial]
            xtest_data = xfeatures[volumen_inicial + (i * volumen_batch):volumen_inicial + ((i + 1) * volumen_batch)]
            ytest_data = ylabels[volumen_inicial + (i * volumen_batch):volumen_inicial + ((i + 1) * volumen_batch)]
        elif i<(max-1):
            Xant = xfeatures[volumen_inicial + ((i - 1) * volumen_batch):volumen_inicial + (i * volumen_batch)]
            yant = ylabels[volumen_inicial + ((i - 1) * volumen_batch):volumen_inicial + (i * volumen_batch)]
            xtest_data = xfeatures[volumen_inicial + (i * volumen_batch):volumen_inicial + ((i + 1) * volumen_batch)]
            ytest_data = ylabels[volumen_inicial + (i * volumen_batch):volumen_inicial + ((i + 1) * volumen_batch)]
        else:
            Xant = xfeatures[volumen_inicial + ((i - 1) * volumen_batch):volumen_inicial + (i * volumen_batch)]
            yant = ylabels[volumen_inicial + ((i - 1) * volumen_batch):volumen_inicial + (i * volumen_batch)]
            xtest_data = xfeatures[volumen_inicial+((i)*volumen_batch):]
            ytest_data = ylabels[volumen_inicial+((i)*volumen_batch):]
        if metodo=='PCA':
            if i == 0:
                mapper = PCA(n_components=dim)
            mapper.fit(Xant)
        elif metodo=='KERNEL':
            if i == 0:
                mapper = KernelPCA(n_components=dim, kernel='rbf', gamma=13, random_state=42)
            mapper.fit(Xant)
        elif metodo == 'ISOMAP':
            if i == 0:
                mapper = Isomap(n_neighbors=40, n_components=dim, eigen_solver='auto')
            mapper.fit(Xant)
        elif metodo == 'INCREPCA':
            print(i)
            if i==0:
                vol=Xant.shape[0]
                mapper = IncrementalPCA(n_components=dim,batch_size=vol)
            elif i==1:
                vol = Xant.shape[0]
                mapper = IncrementalPCA(n_components=dim, batch_size=vol)
            mapper.partial_fit(Xant,y=yant)
        elif metodo == 'TRUNCATED':
            if i == 0:
                mapper = TruncatedSVD(n_components=dim, n_iter=7, random_state=42)
            mapper.fit(Xant)
        if i == 0:
            xant_trans=mapper.transform(Xant)
            yant_trans=yant
            xdef.append(xant_trans)
        else:
            xant_trans=np.array(xdef[i])
            yant_trans=yant

        test_transformados = mapper.transform(xtest_data)
        xdef.append(test_transformados)

    flat_list = [item for sublist in xdef for item in sublist]
    return np.array(flat_list), ylabels
def definir_sets_training_test(x, y, vol_batch, contador2,NUM_poriteracion):
    Xsupervised = x[contador2 * vol_batch:contador2 * vol_batch+NUM_poriteracion]
    ysupervised = y[contador2 * vol_batch:contador2 * vol_batch+NUM_poriteracion]

    principio_test =contador2 * vol_batch+NUM_poriteracion
    final_test = principio_test + vol_batch

    Xtest = x[principio_test:final_test]
    ytest = np.array([-1] * vol_batch)
    ytest = y[principio_test:final_test]

    return Xsupervised, ysupervised, Xtest, ytest

def definir_sets_training_test_parciales(x, ylabels,ypredicted,flag, contador2, vol_batch, num_labeled,num_unlabeled,NUM_poriteracion,num_inicio_no_etq):

    if( (flag==0) &(num_unlabeled<NUM_poriteracion)):
        final_lab = num_inicio_no_etq
        inicio_lab = final_lab-num_labeled
        inicio_parcial = final_lab
        final_parcial = inicio_parcial + num_unlabeled
        principio_test = final_parcial
        final_test = principio_test + vol_batch
    elif ((flag != 0) & (num_unlabeled <= NUM_poriteracion)):
        inicio_lab = num_inicio_no_etq
        final_lab = num_inicio_no_etq
        principio_test = contador2 * vol_batch + NUM_poriteracion
        final_test = principio_test + vol_batch
        final_parcial = principio_test
        inicio_parcial=final_parcial-num_unlabeled
    else:
        inicio_lab=num_inicio_no_etq
        final_lab=num_inicio_no_etq
        inicio_parcial = contador2 * vol_batch
        final_parcial = inicio_parcial + NUM_poriteracion
        principio_test = final_parcial
        final_test = principio_test + vol_batch

    if (flag == 0):
        final_train = final_parcial
        inicio_train = final_parcial - NUM_poriteracion
    else:
        final_train = final_parcial
        inicio_train = final_parcial - num_unlabeled

    X_trainP = x[inicio_train:final_train, :]

    y_trainP = np.array([-1] * (final_train - inicio_train))
    if (inicio_lab < num_inicio_no_etq):
        y_trainP = np.concatenate((ylabels[inicio_lab:final_lab],ypredicted[-num_unlabeled:]))
    else:
        y_trainP = ypredicted[:]

    Xtest = x[principio_test:final_test]
    ytest = np.array([-1] * vol_batch)
    ytest = ylabels[principio_test:final_test]

    return X_trainP, y_trainP, Xtest, ytest


def definir_sets_training_testEVL(x, y, vol_batch, contador2,NUM_poriteracion):
    Xsupervised = x[contador2 * vol_batch:contador2 * vol_batch+NUM_poriteracion-vol_batch]
    ysupervised = y[contador2 * vol_batch:contador2 * vol_batch+NUM_poriteracion-vol_batch]

    principio_VAL = contador2 * vol_batch + (NUM_poriteracion-vol_batch)
    final_VAL = principio_VAL + vol_batch

    principio_test =final_VAL
    final_test = principio_test + vol_batch

    Xval = x[principio_VAL:final_VAL]
    Yval = np.array([-1] * vol_batch)
    Yval= y[principio_VAL:final_VAL]

    Xtest = x[principio_test:final_test]
    ytest = np.array([-1] * vol_batch)
    ytest = y[principio_test:final_test]

    return Xsupervised, ysupervised,Xval,Yval, Xtest, ytest

def definir_sets_training_test_parcialesEVL(x, ylabels,ypredicted,contador2, vol_batch, num_labeled,num_unlabeled,NUM_poriteracion,num_inicio_no_etq):

    if(num_unlabeled<NUM_poriteracion):
        final_lab = num_inicio_no_etq-vol_batch
        inicio_lab = final_lab-num_labeled
        inicio_val = final_lab
        final_val = inicio_val + num_unlabeled
        inicio_parcial = final_val
        final_parcial = inicio_parcial + num_unlabeled
        principio_test = final_parcial
        final_test = principio_test + vol_batch
    else:
        inicio_lab=num_inicio_no_etq
        final_lab=num_inicio_no_etq
        inicio_parcial = contador2 * vol_batch
        final_parcial = inicio_parcial + NUM_poriteracion-vol_batch
        inicio_val = final_parcial
        final_val = final_parcial + vol_batch
        principio_test = final_val
        final_test = principio_test + vol_batch

    final_train = final_parcial
    inicio_train = inicio_parcial

    X_trainP = x[inicio_train:final_train, :]
    X_valP = x[inicio_val:final_val, :]

    y_trainP = np.array([-1] * (final_train - inicio_train))
    Y_valP = np.array([-1] * vol_batch)
    if (inicio_lab < num_inicio_no_etq-vol_batch):
        y_trainP = np.concatenate((ylabels[inicio_lab:final_lab],ypredicted[-num_unlabeled-vol_batch:]))
        Y_valP=ypredicted[-vol_batch:]
    else:
        y_trainP = ypredicted[:-vol_batch]
        Y_valP = ypredicted[-vol_batch:]

    Xtest = x[principio_test:final_test]
    ytest = np.array([-1] * vol_batch)
    ytest = ylabels[principio_test:final_test]

    return X_trainP, y_trainP,X_valP,Y_valP,  Xtest, ytest

def feature_importance(X_train,y_train):
    ss = StandardScaler()
    X_train_scaled = ss.fit_transform(X_train)
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    df = pd.DataFrame()
    for c in range(X_train.shape[1]):
        df[coordenadas[c]] = X_train[:, c]

    importances = pd.DataFrame(data={
        'Attribute': df.columns,
        'Importance': model.coef_[0]
    })
    importances = importances.sort_values(by='Importance', ascending=False)
    resta=np.abs(model.coef_[0]).max()-np.abs(model.coef_[0]).min()
    identificador=0
    if resta<0.5:
        identificador=9

    return importances,np.argmax(np.abs(model.coef_[0])),np.argmin(np.abs(model.coef_[0])),identificador

def obtencion_k(NUM):
    if NUM <= 1000:
        tam=(int)(NUM/4)
    else:
        tam=NUM
    raiz=np.sqrt(tam)
    if (int(raiz)%2==0):
        raiz=raiz-1
    k_elegida=raiz
    k_elegida=int(k_elegida)
    return k_elegida

def salvar_resultados_lista(lbl, array_resultados):
    valor_MEDIA = np.mean(array_resultados)
    valor_STD = np.std(array_resultados)
    valor = valor_MEDIA - valor_STD
    print('\nVALORES ',array_resultados,'\n')
    print(lbl, 'MEDIA:', valor_MEDIA, 'STD:', valor_STD, 'RESTA:', valor)

def calcularAcumuladonew(contador2, array_acum, inicial):
    denominador = contador2 + 1
    acumulado = array_acum[:denominador].sum()
    retorno = 0.0
    if (contador2 > 0):
        retorno = acumulado / denominador
    else:
        retorno = inicial

    return retorno

def calcularAcumulado(contador2, array_acum, inicial, vol):
    acumulado = 0
    acumulado_COPIA = 0
    for cont_acum in range(contador2 + 1):
        tmp = int(vol) * (array_acum[cont_acum])
        acumulado = acumulado + tmp
        tmp_COPIA = int(vol) * (array_acum[cont_acum])
        acumulado_COPIA = acumulado_COPIA + tmp_COPIA

    denominador = contador2 + 1
    retorno = 0
    retorno_copia = 0
    if (contador2 > 0):
        retorno = acumulado / (denominador * vol)
        retorno_copia = acumulado_COPIA / (denominador * vol)
    else:
        retorno = inicial
        retorno_copia = inicial
    return retorno_copia

def calcular_ERROR_Acumuladonew(contador2, error_acum, error_inicial):
    denominador = contador2 + 1
    acumulado = error_acum[:denominador].sum()
    retorno = 0.0
    if (contador2 > 0):
        retorno = acumulado / denominador
    else:
        retorno = error_inicial
    return retorno

def calcular_ERROR_Acumulado(contador2, error_acum, error_inicial, vol):
    acumulado = 0
    for cont_acum in range(contador2 + 1):
        tmp = int(vol) * (error_acum[cont_acum])
        acumulado = acumulado + tmp
    denominador = contador2 + 1
    retorno = 0.0
    if (contador2 > 0):
        retorno = acumulado / (denominador * vol)
    else:
        retorno = error_inicial
    return retorno
def salvar_resultados(lbl, flag, array_resultados, array_resultados_ACUMULADOS,f1=0):
    valor_MEDIA = np.mean(array_resultados)
    valor_STD = np.std(array_resultados)
    valor = valor_MEDIA - valor_STD
    valor_MEDIA_ACUMULADOS = np.mean(array_resultados_ACUMULADOS)
    valor_STD_ACUMULADOS = np.std(array_resultados_ACUMULADOS)
    valor_ACUMULADOS = valor_MEDIA_ACUMULADOS - valor_STD_ACUMULADOS

    if flag == 1:
        print(lbl, 'MEDIA:', valor_MEDIA, 'STD:', valor_STD, 'RESTA:', valor)
        print(lbl, 'MEDIA_ACUMULADOS:', valor_MEDIA_ACUMULADOS, 'STD:', valor_STD_ACUMULADOS, 'RESTA:',
              valor_ACUMULADOS)
        if array_resultados_ACUMULADOS:
            print(lbl, 'PREQUENTIAL:', array_resultados_ACUMULADOS[-1])
        print(lbl, 'F1:', np.mean(f1))
    else:
        print(lbl + str('_ERROR'), 'MEDIA:', valor_MEDIA, 'STD:', valor_STD, 'RESTA:', valor)
        print(lbl + str('_ERROR'), 'MEDIA_ACUMULADOS:', valor_MEDIA_ACUMULADOS, 'STD:', valor_STD_ACUMULADOS, 'RESTA:',
              valor_ACUMULADOS)
        if array_resultados_ACUMULADOS:
            print(lbl, 'PREQUENTIAL:', array_resultados_ACUMULADOS[-1])
def salvar_resultadosEVL(lbl, flag, array_resultados, array_resultados_ACUMULADOS,f1=0,micro=0,you=0,auc=0,auc_com=0):
    valor_MEDIA1 = np.mean(array_resultados)
    valor_STD1 = np.std(array_resultados)
    valor1 = valor_MEDIA1 - valor_STD1
    valor_MEDIA_ACUMULADOS1 = np.mean(array_resultados_ACUMULADOS)
    valor_STD_ACUMULADOS1 = np.std(array_resultados_ACUMULADOS)
    valor_ACUMULADOS1 = valor_MEDIA_ACUMULADOS1 - valor_STD_ACUMULADOS1

    valor_MEDIA2 = np.mean(auc)
    valor_STD2 = np.std(auc)
    valor2 = valor_MEDIA2 - valor_STD2
    valor_MEDIA_ACUMULADOS2 = np.mean(auc_com)
    valor_STD_ACUMULADOS2 = np.std(auc_com)
    valor_ACUMULADOS2 = valor_MEDIA_ACUMULADOS2- valor_STD_ACUMULADOS2

    if flag == 1:
        print(lbl, 'MEDIA:', valor_MEDIA1, 'STD:', valor_STD1, 'RESTA:', valor1)
        print(lbl, 'MEDIA_ACUMULADOS:', valor_MEDIA_ACUMULADOS1, 'STD:', valor_STD_ACUMULADOS1, 'RESTA:',
              valor_ACUMULADOS1)
        print(lbl, 'MEDIA ARTICULO:', valor_MEDIA2, 'STD:', valor_STD2, 'RESTA:', valor2)
        print(lbl, 'MEDIA_ACUMULADOS ARTICULO:', valor_MEDIA_ACUMULADOS2, 'STD:', valor_STD_ACUMULADOS2, 'RESTA:',
              valor_ACUMULADOS2)
        if array_resultados_ACUMULADOS:
            print(lbl, 'PREQUENTIAL:', array_resultados_ACUMULADOS[-1])
        if auc_com:
            print(lbl, 'PREQUENTIAL ARTICULO:', auc_com[-1])
        print(lbl, 'F1:', np.mean(f1))
        print(lbl, 'Micro F1 OSR (mean):', np.mean(micro))
        print(lbl, 'Youden OSR (mean)::', np.mean(you))
        print(lbl, 'AUC OSR (mean):', np.mean(auc))
    else:
        print(lbl + str('_ERROR'), 'MEDIA:', valor_MEDIA1, 'STD:', valor_STD1, 'RESTA:', valor1)
        print(lbl + str('_ERROR'), 'MEDIA_ACUMULADOS:', valor_MEDIA_ACUMULADOS1, 'STD:', valor_STD_ACUMULADOS1, 'RESTA:',
              valor_ACUMULADOS1)
        if array_resultados_ACUMULADOS:
            print(lbl, 'PREQUENTIAL:', array_resultados_ACUMULADOS[-1])
        print(lbl + str('_ERROR ARTICULO'), 'MEDIA:', valor_MEDIA2, 'STD:', valor_STD2, 'RESTA:', valor2)
        print(lbl + str('_ERROR ARTICULO'), 'MEDIA_ACUMULADOS:', valor_MEDIA_ACUMULADOS2, 'STD:', valor_STD_ACUMULADOS2, 'RESTA:',
              valor_ACUMULADOS2)
        if auc_com:
            print(lbl, 'PREQUENTIAL ARTICULO', array_resultados_ACUMULADOS[-1])

def salvar_parametros(preq, including_percentage, dim_TOTAL, metrica_param,numero_reps,dist_save_mod,contour_percentage,tolerance_iou=0,box=0,iou=0):
    print( 'INCLUDING PERCENTAGE : ',including_percentage, 'DIM TOTAL: ',dim_TOTAL,  'METRICA :', metrica_param, 'REPETICIONES :', numero_reps, '\n')
    print( 'DISTANCIA SALVAR MODELOS :',dist_save_mod, 'Parametro Contorno IOU :',contour_percentage,'Tolerance IOU :',tolerance_iou,'RESULTADOS BOXPLOT: ',box,'\n')
    print('List IOU:', iou)
    print('PREQUENTIAL:', preq)

def leer(file):
    import os
    from pathlib import Path

    data = Path(os.environ.get("AIGAS_DATA_DIR", "datasets_OSR"))
    path = data / file
    df = pd.read_csv(path, sep=",")
    return df
def leer_REALES(file,particion):
    df = pd.read_csv('datasets_OSR/TestData/' + file, sep=',')#datasets

    known_total = []
    unknown_total = []

    with open('datasets_OSR/TestData/' + particion, "r", encoding="utf-8") as f:
        for linea in f:
            known_match = re.search(r"known_class\s*=\s*\[([^\]]+)\]", linea)
            unknown_match = re.search(r"unknown_class\s*=\s*\[([^\]]+)\]", linea)

            if known_match:
                tmp1=[]
                tmp1.extend(
                    map(int, known_match.group(1).split(","))
                )
                known_total.append(tmp1)

            if unknown_match:
                tmp2=[]
                tmp2.extend(
                    map(int, unknown_match.group(1).split(","))
                )
                unknown_total.append(tmp1)

    return df,known_total, unknown_total

def letras_a_enteros_df(df, columna="Target"):
    df[columna] = df[columna].str.lower().apply(lambda x: ord(x) - ord('a') + 1)
    return df
def target_to_numeric(targets):
    result = []

    for t in targets:
        if t == 'UUC':
            result.append(100)
        elif isinstance(t, str) and len(t) == 1 and t.isalpha():
            result.append(ord(t.lower()) - ord('a') + 1)
        else:
            raise ValueError(f"Target no reconocido: {t}")

    return result
def numeric_to_target(values):
    result = []

    for v in values:
        if v == 100:
            result.append('UUC')
        elif  1 <= v <= 26:#isinstance(v, int) and
            result.append(chr(v - 1 + ord('A')))
        else:
            raise ValueError(f"Valor numérico no reconocido: {v}")

    return result

def split_by_known_unknown(df,known, unknown):
    known_set = set(known)
    unknown_set = set(unknown)

    train_df = df[df["Target"].isin(known_set)].copy()
    test_df  = df[df["Target"].isin(unknown_set)].copy()
    val_df   = df[~df["Target"].isin(known_set | unknown_set)].copy()

    return train_df, val_df,test_df


def calculate_quantization_error(model, data):
    """
    Calcula el Error de Cuantización (QE):
    La distancia promedio de cada punto al nodo más cercano.
    """
    # Obtenemos los pesos de todas las neuronas actuales
    nodes = np.array([n.weight for n in model.graph.nodes])

    total_dist = 0
    for x in data:
        # Distancia euclídea al nodo más cercano
        dist = np.min(np.linalg.norm(nodes - x, axis=1))
        total_dist += dist

    return total_dist / len(data)


def optimize_gng_topology(X_train, X_val, n_inputs):
    """
    Ajusta los parámetros de crecimiento basándose en el rendimiento sobre X_val.
    """
    # Espacio de búsqueda de hiperparámetros
    # step: velocidad de aprendizaje
    # max_nodes: límite de crecimiento
    # max_age: cuánto tiempo vive una arista antes de morir
    param_grid = {
        'step': [0.1, 0.2, 0.3],
        'max_nodes': [500, 1000, 2000],
        'max_age': [30, 50, 70]
    }

    best_qe = float('inf')
    best_params = {}

    print("Optimizando topología GNG sobre X_val...")

    for s in param_grid['step']:
        for m in param_grid['max_nodes']:
            for a in param_grid['max_age']:
                # 1. Crear modelo con parámetros actuales
                model = create_gng(
                    n_i=n_inputs,
                    step_gng=s,
                    max=m,
                    max_age=a
                )

                # 2. Entrenar con el Fitting Set (X_train)
                # Usamos una versión simplificada de tu train_gng
                model.train(X_train, epochs=10)  # Pocas épocas para optimizar rápido

                # 3. Validar sobre X_val
                current_qe = calculate_quantization_error(model, X_val)

                if current_qe < best_qe:
                    best_qe = current_qe
                    best_params = {'step': s, 'max_nodes': m, 'max_age': a}

    print(f"Optimización completada. Best QE: {best_qe:.4f}")
    return best_params



def create_gng(n_i=2, max=2000, step_gng=0.2, n_start=200, max_age=50,min_distance=0):
    return algorithms.GrowingNeuralGas(
        n_inputs=n_i,
        n_start_nodes=n_start,
        show_epoch=1,

        shuffle_data=True,
        verbose=False,

        step=step_gng,
        neighbour_step=0.005,

        max_edge_age=max_age,
        max_nodes=max,

        n_iter_before_neuron_added=10,
        after_split_error_decay_rate=0.5,
        error_decay_rate=0.995,
        min_distance_for_update=min_distance,
    )

def retrain_gng(flag,train,modelo,gng_anterior,dim,step_gng,n_start,max,max_age,min_distance):
    if flag==0:
        for epoch in range(50):
            modelo.train(train, epochs=1)
        n_temporales = []
        edges = list(modelo.graph.edges.keys())
        numero_edges = len(edges)
        for e in range(numero_edges):
            neuron_1, neuron_2 = edges[e]
            nodo1 = neuron_1.weight.T
            nodo2 = neuron_2.weight.T

            t_nodo1 = tuple(nodo1)
            t_nodo2 = tuple(nodo2)

            if t_nodo1 in np.array(n_temporales):
                V = 1
            else:
                n_temporales.append(t_nodo1)

            if (t_nodo2 in np.array(n_temporales)):
                V = 1
            else:
                n_temporales.append(t_nodo2)
    else:
        modelo = create_gng(n_i=dim, step_gng=step_gng, n_start=n_start, max=max, max_age=max_age,min_distance=min_distance)
        modelo.initialize_nodes(gng_anterior)
        for epoch in range(50):
            modelo.train(train, epochs=1)
        n_temporales = []
        edges = list(modelo.graph.edges.keys())
        numero_edges = len(edges)
        for e in range(numero_edges):
            neuron_1, neuron_2 = edges[e]
            nodo1 = neuron_1.weight.T
            nodo2 = neuron_2.weight.T

            if nodo1 in np.array(n_temporales):
                V = 1
            else:
                n_temporales.append(nodo1)
            if (nodo2 in np.array(n_temporales)):
                V = 1
            else:
                n_temporales.append(nodo2)
    return n_temporales, modelo
def train_gng(train,modelo,epochs=50):
    for epoch in range(epochs):
        modelo.train(train, epochs=1)
    # Nodos con al menos una arista, sin repetir. Se deduplica por neurona:
    # 'nodo in np.array(lista)' compara coordenada a coordenada y descartaba
    # nodos validos en cuanto coincidia una sola coordenada con otro nodo.
    n_temporales = []
    vistos = set()
    for neuron_1, neuron_2 in modelo.graph.edges.keys():
        for neurona in (neuron_1, neuron_2):
            if id(neurona) not in vistos:
                vistos.add(id(neurona))
                n_temporales.append(neurona.weight.T)

    return n_temporales,modelo
def reset_gng(flag,train,dim,step_gng,n_start,max,max_age,min_distance,gng_anterior=[]):
    modelo=create_gng(n_i=dim,step_gng=step_gng,n_start=n_start,max=max,max_age=max_age,min_distance=min_distance)
    if flag==0:
        modelo.initialize_nodes(gng_anterior)

    for epoch in range(50):
        modelo.train(train, epochs=1)
    n_temporales = []
    edges = list(modelo.graph.edges.keys())
    numero_edges = len(edges)
    for e in range(numero_edges):
        neuron_1, neuron_2 = edges[e]
        nodo1 = neuron_1.weight.T
        nodo2 = neuron_2.weight.T

        if nodo1 in np.array(n_temporales):
            V = 1
        else:
            n_temporales.append(nodo1)

        if (nodo2 in np.array(n_temporales)):
            V = 1
        else:
            n_temporales.append(nodo2)

    if len(n_temporales)<max:
        modelo = create_gng(n_i=dim, step_gng=step_gng, n_start=n_start, max=max+1, max_age=max_age,
                            min_distance=min_distance)
        if flag == 0:
            modelo.initialize_nodes(gng_anterior)

        for epoch in range(200):
            modelo.train(train, epochs=1)
        n_temporales = []
        edges = list(modelo.graph.edges.keys())
        numero_edges = len(edges)
        for e in range(numero_edges):
            neuron_1, neuron_2 = edges[e]
            nodo1 = neuron_1.weight.T
            nodo2 = neuron_2.weight.T

            if nodo1 in np.array(n_temporales):
                V = 1
            else:
                n_temporales.append(nodo1)

            if (nodo2 in np.array(n_temporales)):
                V = 1
            else:
                n_temporales.append(nodo2)

        if (len(n_temporales)>max):
            tmp=np.array(n_temporales)
            tmp=tmp[:max]
            n_temporales=tmp.tolist()
    return n_temporales,modelo

def closest_node_dim(dim,node, listnodes):
    if dim==2:
        nodes = np.asarray(listnodes)
        deltas = nodes - node
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    elif dim==1:
        nodes = np.asarray(listnodes)
        dist_2 = np.abs(nodes - node)
    else:
        total=len(listnodes)
        tmp=[]
        for p in range(total):
            tmp.append(distance.euclidean(node,listnodes[p]))
        dist_2=np.array(tmp)
    return np.argmin(dist_2)

def distances_nodes(dim,node, listnodes):
    if dim==2:
        nodes = np.asarray(listnodes)
        deltas = nodes - node
        dist_2 = np.sqrt(np.einsum('ij,ij->i', deltas, deltas))
    elif dim==1:
        nodes = np.asarray(listnodes)
        dist_2 = np.abs(nodes - node)
    else:
        total = len(listnodes)
        tmp = []
        for p in range(total):
            tmp.append(distance.euclidean(node, listnodes[p]))
        dist_2 = np.array(tmp)
    return (dist_2)
def closest_node(node, listnodes):
    nodes = np.asarray(listnodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)

    return np.argmin(dist_2)
def far_node(dim,node, listnodes):

    if dim == 2:
        nodes = np.asarray(listnodes)
        deltas = nodes - node
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    else:
        total = len(listnodes)
        tmp = []
        for p in range(total):
            tmp.append(distance.euclidean(node, listnodes[p]))
        dist_2 = np.array(tmp)

    return np.argmax(dist_2)

def comparar_conjuntos(X1,  X2,  distancia_limite=1.0):
    """
    Compara dos conjuntos de puntos etiquetados y determina correspondencia,
    reordenamiento y posibles nuevas clases.

    Parámetros:
        X1, X2 : np.ndarray de forma (n, d)

        distancia_limite : float
            Umbral máximo para considerar que un punto pertenece a una clase existente.
    Retorna:
        - correspondencia: lista de tuplas (índice_X1, índice_X2, distancia)
        - X2_reordenado
    """

    # 1. Calcular matriz de distancias
    dist_matrix = cdist(X1, X2, metric='euclidean')

    # 2. Resolver asignación óptima (minimizar suma de distancias)
    fila_ind, col_ind = linear_sum_assignment(dist_matrix)

    correspondencia = [
        (i, j, dist_matrix[i, j]) for i, j in zip(fila_ind, col_ind)
    ]

    # 3. Reordenar X2, y2 según correspondencia
    X2_reordenado = X2[col_ind]

    # 4. Detectar posibles nuevas clases por distancia
    nuevas_clases_idx = []
    for j in range(X2.shape[0]):
        if j not in col_ind:
            nuevas_clases_idx.append(j)

    return correspondencia, X2_reordenado,nuevas_clases_idx
def comparar_ratios(array1, array2):
    """
    Compara proporciones entre dos arrays de distintas longitudes (n y n+1).
    Asume que las posiciones representan clases correspondientes.
    Si array2 tiene más clases, las adicionales se consideran 'nuevas'.
    """
    # Convertir a numpy arrays
    a1 = np.array(array1, dtype=float)
    a2 = np.array(array2, dtype=float)

    # Determinar longitud máxima
    n_max = max(len(a1), len(a2))

    # Extender ambos arrays al mismo tamaño, rellenando con ceros si falta alguna clase
    a1 = np.pad(a1, (0, n_max - len(a1)), mode='constant', constant_values=0)
    a2 = np.pad(a2, (0, n_max - len(a2)), mode='constant', constant_values=0)

    # Calcular proporciones
    total1 = a1.sum()
    total2 = a2.sum()
    p1 = a1 / total1 if total1 > 0 else np.zeros_like(a1)
    p2 = a2 / total2 if total2 > 0 else np.zeros_like(a2)

    resultados = []
    for i in range(n_max):
        if p1[i] == 0 and p2[i] > 0:
            cambio = "Nueva clase"
        elif p1[i] > 0 and p2[i] == 0:
            cambio = "Clase eliminada"
        elif p1[i] == 0 and p2[i] == 0:
            cambio = 0.0
        else:
            cambio = round(((p2[i] - p1[i]) / p1[i]) * 100, 2)

        resultados.append({
            "Clase": i,  # identificador numérico
            "Conteo_1": a1[i],
            "Conteo_2": a2[i],
            "Proporcion_1": round(p1[i], 6),
            "Proporcion_2": round(p2[i], 6),
            "Cambio_%": cambio
        })

    return resultados
def comparar_cambios(resultados, tolerancia=20):
    """
    Compara las variaciones (Cambio_%) de clases coincidentes entre sí.
    Devuelve una lista con pares de clases y un flag global.

    flag = 0 → todas las clases varían de forma similar (dentro del margen)
    flag = 1 → hay clases que varían de forma distinta
    """
    # Filtrar solo las clases comparables (nada de 'nueva clase' o 'eliminada')
    variaciones = [r["Cambio_%"] for r in resultados if isinstance(r["Cambio_%"], (int, float))]

    comparaciones = []
    flag_global = 0  # asumimos que varían igual hasta encontrar lo contrario

    for i in range(len(variaciones) - 1):
        diff = abs(variaciones[i] - variaciones[i + 1])
        flag = 0 if diff <= tolerancia else 1
        comparaciones.append({
            "Clases_comparadas": (i, i + 1),
            "Cambio_Clase_i_%": variaciones[i],
            "Cambio_Clase_i+1_%": variaciones[i + 1],
            "Diferencia_%": round(diff, 2),
            "flag": flag
        })
        if flag == 1:
            flag_global = 1  # al menos una pareja difiere

    return {
        "comparaciones": comparaciones,
        "flag_global": flag_global
    }
def comparar_arrays(a, b):
    """
    Compara dos arrays a partir del segundo elemento (índice 1).
    Retorna:
        - flag=1 "a > b" si a es mayor en la mayoría de posiciones
        - flag=-1 "b > a" si b es mayor en la mayoría de posiciones
        - flag=0 "iguales" si no hay diferencia
    """
    # Determinar longitud mínima común
    n = min(len(a), len(b))

    mayor_a = 0
    mayor_b = 0
    flag=22

    for i in range(1, n):  # desde el segundo elemento
        if a[i] > b[i]:
            mayor_a += 1
        elif b[i] > a[i]:
            mayor_b += 1

    if mayor_a > mayor_b:
        flag = 1
    elif mayor_b > mayor_a:
        flag = -1
    else:
        flag = 0
    return flag

def puntos_mas_lejanos(a, b):
    # Calcula para cada punto de b su distancia mínima a a
    distancias = []
    for p in b:
        d_min = min(distancia(p, q) for q in a)
        distancias.append((d_min, p))

    # Ordena por distancia descendente
    distancias.sort(reverse=True, key=lambda x: x[0])

    # Devuelve tantos puntos como tenga a
    return [p for _, p in distancias[:len(a)]]

def asignacion_clusters_clases(flag_sup,X,Y,flag,relacion,n_c,n_clu,dim,inclu_p,centr=[],asignacion_previa=[]):
    labs_transformados=[]
    cambio = 0

    if flag_sup==1:
        kmeansc = KMeans(n_clusters=n_clu, max_iter=MAX_ITERATIONS, random_state=0, tol=CONVERGENCE_TOLERANCE)
        kmeansc.fit(X)  # kmeansc.fit(X,Y)
        t_c = kmeansc.cluster_centers_
        tmp_centro = t_c.tolist()
        Y_CLU = kmeansc.labels_
        lab = [i for i in range(n_clu)]
        asignacion = np.zeros((n_clu, 2))
        proporcion = np.zeros((n_clu, 2))
        for i in range(n_clu):
            c = Counter(Y[np.where(Y_CLU == i)])
            value, count = c.most_common()[0]
            if len(c) > 1:
                value2, count2 = c.most_common()[1]
                if (count - count2) < count / 10:
                    centroides_referencia = np.array(centr)[:, 0:dim]
                    indice_buscado = closest_node(tmp_centro[i], centroides_referencia)
                    if value != (int)(asignacion_previa[indice_buscado][1]):
                        value = (int)(asignacion_previa[indice_buscado][1])
            asignacion[i] = (i, value)
            proporcion[i] = (i, count)
        indice = 0
        while ((asignacion[indice][1] != asignacion[indice + 1][1]) & (indice + 2 <= n_clu - 1)):
            if ((asignacion[indice][1] != asignacion[indice + 2][1])):
                new_centroides = swapPositions(tmp_centro, indice, -1)
                kmeansc = KMeans(n_clusters=n_clu, init=np.array(new_centroides), max_iter=MAX_ITERATIONS,
                                 random_state=0, tol=CONVERGENCE_TOLERANCE)
                kmeansc.fit(X)  # kmeansc.fit(X, Y)
                tmp_cent = kmeansc.cluster_centers_
                tmp_centro = tmp_cent.tolist()
                Y_CLU = kmeansc.labels_
                lab = [i for i in range(n_clu)]
                asignacion = np.zeros((n_clu, 2))
                proporcion = np.zeros((n_clu, 2))
                for i in range(n_clu):
                    c = Counter(Y[np.where(Y_CLU == i)])
                    value, count = c.most_common()[0]
                    if len(c) > 1:
                        value2, count2 = c.most_common()[1]
                        if (count - count2) < count / 10:
                            centroides_referencia = np.array(centr)[:, 0:dim]
                            indice_buscado = closest_node(tmp_centro[i], centroides_referencia)
                            if value != (int)(asignacion_previa[indice_buscado][1]):
                                value = (int)(asignacion_previa[indice_buscado][1])
                    asignacion[i] = (i, value)
                    proporcion[i] = (i, count)
            else:
                if type(asignacion)==np.ndarray:
                    asignacion = swapPositions(asignacion.tolist(), indice + 1, indice + 2)
                else:
                    asignacion = swapPositions(asignacion, indice + 1, indice + 2)
                tmp_centro = swapPositions(tmp_centro, indice + 1, indice + 2)
            indice += 1
        asignacion = np.array(asignacion)

        kmeanscD = KMeans(n_clusters=n_clu, max_iter=MAX_ITERATIONS, random_state=0, init=np.array(tmp_centro),
                         tol=CONVERGENCE_TOLERANCE)
        kmeanscD.fit(X)
        Y_CLU = kmeanscD.labels_
        t_c = kmeanscD.cluster_centers_
        labs_transformados = Y_CLU

    else:
        if len(asignacion_previa)==0:
            if len(centr)==0:
                kmeansc = KMeans(n_clusters=n_clu, max_iter=MAX_ITERATIONS, random_state=0, tol=CONVERGENCE_TOLERANCE)
                kmeansc.fit(X)
                tmp_cent = kmeansc.cluster_centers_
                tmp_centro = tmp_cent.tolist()
                Y_CLU=kmeansc.labels_
                asignacion = np.zeros((n_clu, 2))
                proporcion = np.zeros((n_clu, 2))

                if len(Y)==0:
                    print('UNO')
                    for i in range(n_clu):
                        c = Counter(Y_CLU[np.where(Y_CLU == i)])
                        value, count = c.most_common()[0]
                        if len(c) > 1:
                            value2, count2 = c.most_common()[1]
                            if (count - count2) < count / 10:
                                centroides_referencia = np.array(centr)[:, 0:dim]
                                indice_buscado = closest_node(tmp_centro[i], centroides_referencia)
                                if value != (int)(asignacion_previa[indice_buscado][1]):
                                    value = (int)(asignacion_previa[indice_buscado][1])
                        proporcion[i] = (i, count)
                else:
                    print('DOS')
                    for i in range(n_clu):
                        c = Counter(Y[np.where(Y_CLU == i)])
                        value, count = c.most_common()[0]
                        if len(c) > 1:
                            value2, count2 = c.most_common()[1]
                            if (count - count2) < count / 10:
                                centroides_referencia = np.array(centr)[:, 0:dim]
                                indice_buscado = closest_node(tmp_centro[i], centroides_referencia)
                                if value != (int)(asignacion_previa[indice_buscado][1]):
                                    value = (int)(asignacion_previa[indice_buscado][1])
                        asignacion[i] = (i, value)
                        proporcion[i] = (i, count)
                    indice = 0
                    while ((asignacion[indice][1] != asignacion[indice + 1][1]) & (indice + 2 <= n_clu - 1)):
                        if ((asignacion[indice][1] != asignacion[indice + 2][1])):
                            new_centroides = swapPositions(tmp_centro, indice, -1)
                            kmeansc = KMeans(n_clusters=n_clu, init=np.array(new_centroides), max_iter=MAX_ITERATIONS,
                                             random_state=0, tol=CONVERGENCE_TOLERANCE)
                            kmeansc.fit(X)  # kmeansc.fit(X, Y)
                            tmp_cent = kmeansc.cluster_centers_
                            tmp_centro = tmp_cent.tolist()
                            Y_CLU = kmeansc.labels_
                            lab = [i for i in range(n_clu)]
                            asignacion = np.zeros((n_clu, 2))
                            proporcion = np.zeros((n_clu, 2))
                            for i in range(n_clu):
                                c = Counter(Y[np.where(Y_CLU == i)])
                                value, count = c.most_common()[0]
                                if len(c) > 1:
                                    value2, count2 = c.most_common()[1]
                                    if (count - count2) < count / 10:
                                        centroides_referencia = np.array(centr)[:, 0:dim]
                                        indice_buscado = closest_node(tmp_centro[i], centroides_referencia)
                                        if value != (int)(asignacion_previa[indice_buscado][1]):
                                            value = (int)(asignacion_previa[indice_buscado][1])
                                asignacion[i] = (i, value)
                                proporcion[i] = (i, count)
                        else:
                            asignacion = swapPositions(asignacion.tolist(), indice + 1, indice + 2)#asignacion.tolist()
                            tmp_centro = swapPositions(tmp_centro, indice + 1, indice + 2)
                        indice += 1

                asignacion = np.array(asignacion)

                kmeanscD = KMeans(n_clusters=n_clu, max_iter=MAX_ITERATIONS, random_state=0, init=np.array(tmp_centro),
                                  tol=CONVERGENCE_TOLERANCE)
                kmeanscD.fit(X)
                Y_CLU = kmeanscD.labels_
                t_c = kmeanscD.cluster_centers_
                labs_transformados = Y_CLU

            else:
                kmeansc = KMeans(n_clusters=n_clu, max_iter=MAX_ITERATIONS, random_state=0, tol=CONVERGENCE_TOLERANCE)
                kmeansc.fit(X)
                tmp_cent = kmeansc.cluster_centers_
                asignacion = np.zeros((n_clu, 2))
                proporcion = np.zeros((n_clu, 2))
                centroides_ordenados = []

                asg, centr_reordenado, new_class=comparar_conjuntos(np.array(centr)[:, :dim] ,tmp_cent, distancia_limite=2.0)

                for i in range(len(asg)):
                    asignacion[i] = (asg[i][0], asg[i][1])
                    centroides_ordenados.append(centr_reordenado[i])
                for j in range (len(new_class)):
                    asignacion[len(asg)+j] = (len(asg)+j, new_class[j])
                    centroides_ordenados.append(tmp_cent[new_class[j]])

                cambio = 2
                kmeansc = KMeans(n_clusters=n_clu, max_iter=MAX_ITERATIONS, random_state=0,
                                 init=np.array(centroides_ordenados), tol=CONVERGENCE_TOLERANCE)
                kmeansc.fit(X)
                labs_transformados = kmeansc.labels_
                t_c = kmeansc.cluster_centers_
                for i in range(n_clu):
                    count = (int)(len(X[np.where(labs_transformados == i)]))#CAMBIO 2025 NO TENGO Y
                    proporcion[i] = (i, count)
        else:
            centroides_referencia = np.array(centr)[:, 0:dim]

            if n_clu==np.array(centr).shape[0]:#
                cambio = 0
                kmeansc = KMeans(n_clusters=n_clu, max_iter=MAX_ITERATIONS, random_state=0, init=centroides_referencia,tol=CONVERGENCE_TOLERANCE)
            elif (n_clu>np.array(centr).shape[0]):
                cambio=1
                kmeansc = KMeans(n_clusters=n_clu, init=INITIALIZE_CLUSTERS, max_iter=MAX_ITERATIONS, random_state=0,tol=CONVERGENCE_TOLERANCE)  # init=(np.array(centr)[:, 0:dim])
                xtrain_kmeans = np.concatenate((np.array(centr)[:, 0:dim], X))
                kmeansc.fit(xtrain_kmeans)#kmeansc.fit(X)
                tmp_cent = kmeansc.cluster_centers_
                tmp_centro = tmp_cent.tolist()

                dist_matrix = distance_matrix(centroides_referencia, tmp_centro)
                row_ind, col_ind = linear_sum_assignment(dist_matrix)
                indices_new = np.arange(n_clu)
                indices_fuera = []
                centroides_ordenados = []

                cont = 0
                for ix in col_ind:
                    centroides_ordenados.append(tmp_centro[ix])
                    cont += 1
                for c in indices_new:
                    if (c not in col_ind):# or ((c not in row_ind))):#((c not in col_ind) or ((c not in row_ind))):
                        indices_fuera.append(c)
                for f in indices_fuera:
                    ind = closest_node(tmp_centro[f], centroides_referencia)
                if ind not in col_ind:
                    centroides_ordenados.append(tmp_centro[ind])
                else:
                    centroides_ordenados.append(tmp_centro[f])
                kmeansc = KMeans(n_clusters=n_clu, max_iter=MAX_ITERATIONS, random_state=0,init=np.array(centroides_ordenados), tol=CONVERGENCE_TOLERANCE)
            else:
                cambio = 1
                kmeansc = KMeans(n_clusters=n_clu, init=INITIALIZE_CLUSTERS, max_iter=MAX_ITERATIONS, random_state=0,
                                 tol=CONVERGENCE_TOLERANCE)  # init=(np.array(centr)[:, 0:dim])
                xtrain_kmeans = np.concatenate((centroides_referencia, X))
                kmeansc.fit(xtrain_kmeans)#kmeansc.fit(X)#
                tmp_cent = kmeansc.cluster_centers_
                tmp_centro = tmp_cent.tolist()

                dist_matrix = distance_matrix(centroides_referencia, tmp_centro)
                row_ind, col_ind = linear_sum_assignment(dist_matrix)
                indices_new = np.arange(n_clu)
                indices_fuera = []
                centroides_ordenados = []

                cont = 0
                for ix in col_ind:
                    centroides_ordenados.append(tmp_centro[ix])
                    cont += 1
                for c in indices_new:
                    if c not in col_ind:
                        indices_fuera.append(c)
                kmeansc = KMeans(n_clusters=n_clu, max_iter=MAX_ITERATIONS, random_state=0,
                                 init=np.array(centroides_ordenados), tol=CONVERGENCE_TOLERANCE)

            kmeansc.fit(X)
            Y_CLU = kmeansc.labels_
            t_c = kmeansc.cluster_centers_
            labs_transformados=Y_CLU

            asignacion = np.zeros((n_clu, 2))
            proporcion = np.zeros((n_clu, 2))

            if centroides_referencia.shape[0]==n_c:
                if n_clu < n_c:
                    asg_comp, centr_reordenado, new_class = comparar_conjuntos(t_c,centroides_referencia,distancia_limite=2.0)
                else:
                    asg_comp, centr_reordenado, new_class = comparar_conjuntos(centroides_referencia, t_c,distancia_limite=2.0)

                if (len(new_class)==0):
                    if(centr_reordenado==t_c).all():
                        asignacion=asignacion_previa[:n_clu]
                        for i in range(n_clu):
                            count = (int)(len(X[np.where(labs_transformados == i)]))#CAMBIO 2025 NO TENGO Y
                            proporcion[i] = (i, count)
                    else:
                        if n_clu > n_c:
                            asignacion[:asignacion_previa.shape[0]] = asignacion_previa
                            indice_centrode_cercano, asignacion[-1] = punto_mas_cercano_con_label(centr_reordenado[-1], centroides_referencia, asignacion_previa)
                        else:
                            for i in range(len(asg_comp)):
                                asignacion[i] = (asg_comp[i][0], asg_comp[i][1])

                        for i in range(n_clu):
                            count = (int)(len(X[np.where(labs_transformados == i)]))#CAMBIO 2025 NO TENGO Y
                            proporcion[i] = (i, count)
                else:
                    if n_clu > n_c:
                        asignacion[:asignacion_previa.shape[0]] = asignacion_previa
                        indice_centrode_cercano, asignacion[-1][1] = punto_mas_cercano_con_label(centr_reordenado[-1],centroides_referencia, asignacion_previa)
                        asignacion[n_clu-1][0]=n_clu-1
                    else:
                        for i in range(len(asg_comp)):
                            asignacion[i] = (asg_comp[i][0], asg_comp[i][1])
                    for i in range(n_clu):
                        count_old = (int)(len(X[np.where(labs_transformados == i)]))
                        proporcion[i] = (i, count_old)
            else:
                if len(Y)==0:
                    if n_clu < centroides_referencia.shape[0]:
                        asg_comp, centr_reordenado, new_class = comparar_conjuntos(t_c, centroides_referencia,
                                                                                   distancia_limite=2.0)
                    else:
                        asg_comp, centr_reordenado, new_class = comparar_conjuntos(centroides_referencia, t_c,
                                                                                   distancia_limite=2.0)
                    if (len(new_class) == 0):
                        if (centr_reordenado == t_c).all():
                            asignacion = asignacion_previa[:n_clu]
                            for i in range(n_clu):
                                count = (int)(len(X[np.where(labs_transformados == i)]))  # CAMBIO 2025 NO TENGO Y
                                proporcion[i] = (i, count)
                        else:
                            if n_clu > n_c:
                                asignacion[:asignacion_previa.shape[0]] = asignacion_previa
                                indice_centrode_cercano, asignacion[-1] = punto_mas_cercano_con_label(
                                    centr_reordenado[-1], centroides_referencia, asignacion_previa)
                            else:
                                for i in range(len(asg_comp)):
                                    asignacion[i] = (asg_comp[i][0], asg_comp[i][1])

                            for i in range(n_clu):
                                count = (int)(len(X[np.where(labs_transformados == i)]))  # CAMBIO 2025 NO TENGO Y
                                proporcion[i] = (i, count)
                    else:
                        if n_clu > n_c:
                            asignacion[:asignacion_previa.shape[0]] = asignacion_previa
                            indice_centrode_cercano, asignacion[-1][1] = punto_mas_cercano_con_label(
                                centr_reordenado[-1], centroides_referencia, asignacion_previa)
                            asignacion[n_clu - 1][0] = n_clu - 1
                        else:
                            for i in range(len(asg_comp)):
                                asignacion[i] = (asg_comp[i][0], asg_comp[i][1])
                        for i in range(n_clu):
                            count_old = (int)(len(X[np.where(labs_transformados == i)]))
                            proporcion[i] = (i, count_old)

                    for i in range(n_clu):
                        c = Counter(Y_CLU[np.where(Y_CLU == i)])
                        value, count = c.most_common()[0]
                        if len(c) > 1:
                            value2, count2 = c.most_common()[1]
                            if (count - count2) < count / 10:
                                centroides_referencia = np.array(centr)[:, 0:dim]
                                indice_buscado = closest_node(t_c[i], centroides_referencia)
                                if value != (int)(asignacion_previa[indice_buscado][1]):
                                    value = (int)(asignacion_previa[indice_buscado][1])
                        proporcion[i] = (i, count)
                else:
                    for i in range(n_clu):
                        c = Counter(Y[np.where(Y_CLU == i)])
                        value, count = c.most_common()[0]
                        if len(c) > 1:
                            value2, count2 = c.most_common()[1]
                            if (count - count2) < count / 10:
                                centroides_referencia = np.array(centr)[:, 0:dim]
                                indice_buscado = closest_node(tmp_centro[i], centroides_referencia)
                                if value != (int)(asignacion_previa[indice_buscado][1]):
                                    value = (int)(asignacion_previa[indice_buscado][1])
                        asignacion[i] = (i, value)
                        proporcion[i] = (i, count)
                    indice = 0
                    while ((asignacion[indice][1] != asignacion[indice + 1][1]) & (indice + 2 <= n_clu - 1)):
                        if ((asignacion[indice][1] != asignacion[indice + 2][1])):
                            new_centroides = swapPositions(tmp_centro, indice, -1)
                            kmeansc = KMeans(n_clusters=n_clu, init=np.array(new_centroides), max_iter=MAX_ITERATIONS,
                                             random_state=0, tol=CONVERGENCE_TOLERANCE)
                            kmeansc.fit(X)  # kmeansc.fit(X, Y)
                            tmp_cent = kmeansc.cluster_centers_
                            tmp_centro = tmp_cent.tolist()
                            Y_CLU = kmeansc.labels_
                            lab = [i for i in range(n_clu)]
                            asignacion = np.zeros((n_clu, 2))
                            proporcion = np.zeros((n_clu, 2))
                            for i in range(n_clu):
                                c = Counter(Y[np.where(Y_CLU == i)])
                                value, count = c.most_common()[0]
                                if len(c) > 1:
                                    value2, count2 = c.most_common()[1]
                                    if (count - count2) < count / 10:
                                        centroides_referencia = np.array(centr)[:, 0:dim]
                                        indice_buscado = closest_node(tmp_centro[i], centroides_referencia)
                                        if value != (int)(asignacion_previa[indice_buscado][1]):
                                            value = (int)(asignacion_previa[indice_buscado][1])
                                asignacion[i] = (i, value)
                                proporcion[i] = (i, count)
                        else:
                            asignacion = swapPositions(asignacion.tolist(), indice + 1, indice + 2)#asignacion.tolist()
                            tmp_centro = swapPositions(tmp_centro, indice + 1, indice + 2)
                        indice += 1
            asignacion = np.array(asignacion)

            kmeanscD = KMeans(n_clusters=n_clu, max_iter=MAX_ITERATIONS, random_state=0, init=np.array(t_c),
                              tol=CONVERGENCE_TOLERANCE)
            kmeanscD.fit(X)
            Y_CLU = kmeanscD.labels_
            t_c = kmeanscD.cluster_centers_
            labs_transformados = Y_CLU

    return labs_transformados,t_c,asignacion,cambio,proporcion

def asignacion_clusters_clasesNEW(flag_sup,X,Y,flag,relacion,n_c,n_clu,dim,inclu_p,centr=[],asignacion_previa=[]):
    labs_transformados=[]
    cambio = 0

    if flag_sup==1:
        if len(asignacion_previa) != 0:
            centroides_referencia = np.array(centr)[:, 0:dim]
            cambio = 0
            kmeansc = KMeans(n_clusters=n_clu, max_iter=MAX_ITERATIONS, random_state=0, init=centroides_referencia,
                             tol=CONVERGENCE_TOLERANCE)
            kmeansc.fit(X)
            Y_CLU = kmeansc.labels_
            t_c = kmeansc.cluster_centers_
            labs_transformados = Y_CLU
            proporcion = np.zeros((n_clu, 2))
            asignacion = asignacion_previa[:n_clu]
            for i in range(n_clu):
                count = (int)(len(X[np.where(labs_transformados == i)]))  # CAMBIO 2025 NO TENGO Y
                proporcion[i] = (i, count)
        else:
            kmeansc = KMeans(n_clusters=n_clu, max_iter=MAX_ITERATIONS, random_state=0, tol=CONVERGENCE_TOLERANCE)
            kmeansc.fit(X)  # kmeansc.fit(X,Y)
            t_c = kmeansc.cluster_centers_
            tmp_centro = t_c.tolist()
            Y_CLU = kmeansc.labels_
            lab = [i for i in range(n_clu)]
            asignacion = np.zeros((n_clu, 2))
            proporcion = np.zeros((n_clu, 2))
            for i in range(n_clu):
                c = Counter(Y[np.where(Y_CLU == i)])
                value, count = c.most_common()[0]
                if len(c) > 1:
                    value2, count2 = c.most_common()[1]
                    if (count - count2) < count / 10:
                        centroides_referencia = np.array(centr)[:, 0:dim]
                        indice_buscado = closest_node(tmp_centro[i], centroides_referencia)
                        if value != (int)(asignacion_previa[indice_buscado][1]):
                            value = (int)(asignacion_previa[indice_buscado][1])
                asignacion[i] = (i, value)
                proporcion[i] = (i, count)
            indice = 0
            while ((asignacion[indice][1] != asignacion[indice + 1][1]) & (indice + 2 <= n_clu - 1)):
                if ((asignacion[indice+1][1] != asignacion[indice + 2][1])):
                    if type(asignacion) == np.ndarray:
                        asignacion = swapPositions(asignacion.tolist(), indice + 1, indice + 2)
                    else:
                        asignacion = swapPositions(asignacion, indice + 1, indice + 2)
                    tmp_centro = swapPositions(tmp_centro, indice + 1, indice + 2)

                    kmeansc = KMeans(n_clusters=n_clu, init=np.array(tmp_centro), max_iter=MAX_ITERATIONS,
                                     random_state=0, tol=CONVERGENCE_TOLERANCE)
                    kmeansc.fit(X)  # kmeansc.fit(X, Y)
                    tmp_cent = kmeansc.cluster_centers_
                    tmp_centro = tmp_cent.tolist()
                    Y_CLU = kmeansc.labels_

                    proporcion = np.zeros((n_clu, 2))
                    for i in range(n_clu):
                        c = Counter(Y[np.where(Y_CLU == i)])
                        value, count = c.most_common()[0]
                        proporcion[i] = (i, count)
                indice += 1
            asignacion = np.array(asignacion)
            kmeanscD = KMeans(n_clusters=n_clu, max_iter=MAX_ITERATIONS, random_state=0, init=np.array(tmp_centro),
                             tol=CONVERGENCE_TOLERANCE)
            kmeanscD.fit(X)
            Y_CLU = kmeanscD.labels_
            t_c = kmeanscD.cluster_centers_
            labs_transformados = Y_CLU
    else:
        if len(asignacion_previa)==0:
            if len(centr)==0:
                kmeansc = KMeans(n_clusters=n_clu, max_iter=MAX_ITERATIONS, random_state=0, tol=CONVERGENCE_TOLERANCE)
                kmeansc.fit(X)
                tmp_cent = kmeansc.cluster_centers_
                tmp_centro = tmp_cent.tolist()
                Y_CLU=kmeansc.labels_
                asignacion = np.zeros((n_clu, 2))
                proporcion = np.zeros((n_clu, 2))

                if len(Y)==0:
                    print('UNO')
                    for i in range(n_clu):
                        c = Counter(Y_CLU[np.where(Y_CLU == i)])
                        value, count = c.most_common()[0]
                        if len(c) > 1:
                            value2, count2 = c.most_common()[1]
                            if (count - count2) < count / 10:
                                centroides_referencia = np.array(centr)[:, 0:dim]
                                indice_buscado = closest_node(tmp_centro[i], centroides_referencia)
                                if value != (int)(asignacion_previa[indice_buscado][1]):
                                    value = (int)(asignacion_previa[indice_buscado][1])
                        proporcion[i] = (i, count)
                else:
                    print('DOS')
                    for i in range(n_clu):
                        c = Counter(Y[np.where(Y_CLU == i)])
                        value, count = c.most_common()[0]
                        if len(c) > 1:
                            value2, count2 = c.most_common()[1]
                            if (count - count2) < count / 10:
                                centroides_referencia = np.array(centr)[:, 0:dim]
                                indice_buscado = closest_node(tmp_centro[i], centroides_referencia)
                                if value != (int)(asignacion_previa[indice_buscado][1]):
                                    value = (int)(asignacion_previa[indice_buscado][1])
                        asignacion[i] = (i, value)
                        proporcion[i] = (i, count)
                    indice = 0
                    while ((asignacion[indice][1] != asignacion[indice + 1][1]) & (indice + 2 <= n_clu - 1)):
                        if ((asignacion[indice+1][1] != asignacion[indice + 2][1])):#(asignacion[indice][1] != asignacion[indice + 2][1])
                            new_centroides = swapPositions(tmp_centro, indice, -1)
                            kmeansc = KMeans(n_clusters=n_clu, init=np.array(new_centroides), max_iter=MAX_ITERATIONS,
                                             random_state=0, tol=CONVERGENCE_TOLERANCE)
                            kmeansc.fit(X)  # kmeansc.fit(X, Y)
                            tmp_cent = kmeansc.cluster_centers_
                            tmp_centro = tmp_cent.tolist()
                            Y_CLU = kmeansc.labels_
                            lab = [i for i in range(n_clu)]
                            asignacion = np.zeros((n_clu, 2))
                            proporcion = np.zeros((n_clu, 2))
                            for i in range(n_clu):
                                c = Counter(Y[np.where(Y_CLU == i)])
                                value, count = c.most_common()[0]
                                if len(c) > 1:
                                    value2, count2 = c.most_common()[1]
                                    if (count - count2) < count / 10:
                                        centroides_referencia = np.array(centr)[:, 0:dim]
                                        indice_buscado = closest_node(tmp_centro[i], centroides_referencia)
                                        if value != (int)(asignacion_previa[indice_buscado][1]):
                                            value = (int)(asignacion_previa[indice_buscado][1])
                                asignacion[i] = (i, value)
                                proporcion[i] = (i, count)
                        indice += 1

                asignacion = np.array(asignacion)
                kmeanscD = KMeans(n_clusters=n_clu, max_iter=MAX_ITERATIONS, random_state=0, init=np.array(tmp_centro),
                                  tol=CONVERGENCE_TOLERANCE)
                kmeanscD.fit(X)
                Y_CLU = kmeanscD.labels_
                t_c = kmeanscD.cluster_centers_
                labs_transformados = Y_CLU
            else:
                kmeansc = KMeans(n_clusters=n_clu, max_iter=MAX_ITERATIONS, random_state=0, tol=CONVERGENCE_TOLERANCE)
                kmeansc.fit(X)
                tmp_cent = kmeansc.cluster_centers_
                asignacion = np.zeros((n_clu, 2))
                proporcion = np.zeros((n_clu, 2))
                centroides_ordenados = []
                asg, centr_reordenado, new_class=comparar_conjuntos(np.array(centr)[:, :dim] ,tmp_cent, distancia_limite=2.0)

                for i in range(len(asg)):
                    asignacion[i] = (asg[i][0], asg[i][1])
                    centroides_ordenados.append(centr_reordenado[i])
                for j in range (len(new_class)):
                    asignacion[len(asg)+j] = (len(asg)+j, new_class[j])
                    centroides_ordenados.append(tmp_cent[new_class[j]])

                cambio = 2
                kmeansc = KMeans(n_clusters=n_clu, max_iter=MAX_ITERATIONS, random_state=0,
                                 init=np.array(centroides_ordenados), tol=CONVERGENCE_TOLERANCE)
                kmeansc.fit(X)
                labs_transformados = kmeansc.labels_
                t_c = kmeansc.cluster_centers_
                for i in range(n_clu):
                    count = (int)(len(X[np.where(labs_transformados == i)]))#CAMBIO 2025 NO TENGO Y
                    proporcion[i] = (i, count)
        else:
            centroides_referencia = np.array(centr)[:, 0:dim]

            if n_clu==np.array(centr).shape[0]:#
                cambio = 0
                kmeansc = KMeans(n_clusters=n_clu, max_iter=MAX_ITERATIONS, random_state=0, init=centroides_referencia,tol=CONVERGENCE_TOLERANCE)
            elif (n_clu>np.array(centr).shape[0]):
                cambio=1
                kmeansc = KMeans(n_clusters=n_clu, init=INITIALIZE_CLUSTERS, max_iter=MAX_ITERATIONS, random_state=0,tol=CONVERGENCE_TOLERANCE)  # init=(np.array(centr)[:, 0:dim])
                xtrain_kmeans = np.concatenate((np.array(centr)[:, 0:dim], X))
                kmeansc.fit(xtrain_kmeans)#kmeansc.fit(X)
                tmp_cent = kmeansc.cluster_centers_
                tmp_centro = tmp_cent.tolist()

                dist_matrix = distance_matrix(centroides_referencia, tmp_centro)
                row_ind, col_ind = linear_sum_assignment(dist_matrix)
                indices_new = np.arange(n_clu)
                indices_fuera = []
                centroides_ordenados = []

                cont = 0
                for ix in col_ind:
                    centroides_ordenados.append(tmp_centro[ix])
                    cont += 1
                for c in indices_new:
                    if (c not in col_ind):# or ((c not in row_ind))):#((c not in col_ind) or ((c not in row_ind))):
                        indices_fuera.append(c)
                for f in indices_fuera:
                    ind = closest_node(tmp_centro[f], centroides_referencia)
                if ind not in col_ind:
                    centroides_ordenados.append(tmp_centro[ind])
                else:
                    centroides_ordenados.append(tmp_centro[f])
                kmeansc = KMeans(n_clusters=n_clu, max_iter=MAX_ITERATIONS, random_state=0,init=np.array(centroides_ordenados), tol=CONVERGENCE_TOLERANCE)
            else:
                cambio = 1
                kmeansc = KMeans(n_clusters=n_clu, init=INITIALIZE_CLUSTERS, max_iter=MAX_ITERATIONS, random_state=0,
                                 tol=CONVERGENCE_TOLERANCE)  # init=(np.array(centr)[:, 0:dim])
                xtrain_kmeans = np.concatenate((centroides_referencia, X))
                kmeansc.fit(xtrain_kmeans)#kmeansc.fit(X)#
                tmp_cent = kmeansc.cluster_centers_
                tmp_centro = tmp_cent.tolist()

                dist_matrix = distance_matrix(centroides_referencia, tmp_centro)
                row_ind, col_ind = linear_sum_assignment(dist_matrix)
                indices_new = np.arange(n_clu)
                indices_fuera = []
                centroides_ordenados = []

                cont = 0
                for ix in col_ind:
                    centroides_ordenados.append(tmp_centro[ix])
                    cont += 1
                for c in indices_new:
                    if c not in col_ind:
                        indices_fuera.append(c)
                kmeansc = KMeans(n_clusters=n_clu, max_iter=MAX_ITERATIONS, random_state=0,
                                 init=np.array(centroides_ordenados), tol=CONVERGENCE_TOLERANCE)

            kmeansc.fit(X)
            Y_CLU = kmeansc.labels_
            t_c = kmeansc.cluster_centers_
            labs_transformados=Y_CLU

            asignacion = np.zeros((n_clu, 2))
            proporcion = np.zeros((n_clu, 2))

            if centroides_referencia.shape[0]==n_c:
                if n_clu < n_c:
                    asg_comp, centr_reordenado, new_class = comparar_conjuntos(t_c,centroides_referencia,distancia_limite=2.0)
                else:
                    asg_comp, centr_reordenado, new_class = comparar_conjuntos(centroides_referencia, t_c,distancia_limite=2.0)

                if (len(new_class)==0):
                    if(centr_reordenado==t_c).all():
                        asignacion=asignacion_previa[:n_clu]
                        for i in range(n_clu):
                            count = (int)(len(X[np.where(labs_transformados == i)]))#CAMBIO 2025 NO TENGO Y
                            proporcion[i] = (i, count)
                    else:
                        if n_clu > n_c:
                            asignacion[:asignacion_previa.shape[0]] = asignacion_previa
                            indice_centrode_cercano, asignacion[-1] = punto_mas_cercano_con_label(centr_reordenado[-1], centroides_referencia, asignacion_previa)
                        else:
                            for i in range(len(asg_comp)):
                                asignacion[i] = (asg_comp[i][0], asg_comp[i][1])

                        for i in range(n_clu):
                            count = (int)(len(X[np.where(labs_transformados == i)]))#CAMBIO 2025 NO TENGO Y
                            proporcion[i] = (i, count)
                else:
                    if n_clu > n_c:
                        asignacion[:asignacion_previa.shape[0]] = asignacion_previa
                        indice_centrode_cercano, asignacion[-1][1] = punto_mas_cercano_con_label(centr_reordenado[-1],centroides_referencia, asignacion_previa)
                        asignacion[n_clu-1][0]=n_clu-1
                    else:
                        for i in range(len(asg_comp)):
                            asignacion[i] = (asg_comp[i][0], asg_comp[i][1])
                    for i in range(n_clu):
                        count_old = (int)(len(X[np.where(labs_transformados == i)]))
                        proporcion[i] = (i, count_old)
            else:
                if len(Y)==0:
                    if n_clu < centroides_referencia.shape[0]:
                        asg_comp, centr_reordenado, new_class = comparar_conjuntos(t_c, centroides_referencia,
                                                                                   distancia_limite=2.0)
                    else:
                        asg_comp, centr_reordenado, new_class = comparar_conjuntos(centroides_referencia, t_c,
                                                                                   distancia_limite=2.0)
                    if (len(new_class) == 0):
                        if (centr_reordenado == t_c).all():
                            asignacion = asignacion_previa[:n_clu]
                            for i in range(n_clu):
                                count = (int)(len(X[np.where(labs_transformados == i)]))  # CAMBIO 2025 NO TENGO Y
                                proporcion[i] = (i, count)
                        else:
                            if n_clu > n_c:
                                asignacion[:asignacion_previa.shape[0]] = asignacion_previa
                                indice_centrode_cercano, asignacion[-1] = punto_mas_cercano_con_label(
                                    centr_reordenado[-1], centroides_referencia, asignacion_previa)
                            else:
                                for i in range(len(asg_comp)):
                                    asignacion[i] = (asg_comp[i][0], asg_comp[i][1])

                            for i in range(n_clu):
                                count = (int)(len(X[np.where(labs_transformados == i)]))  # CAMBIO 2025 NO TENGO Y
                                proporcion[i] = (i, count)
                    else:
                        if n_clu > n_c:
                            asignacion[:asignacion_previa.shape[0]] = asignacion_previa
                            indice_centrode_cercano, asignacion[-1][1] = punto_mas_cercano_con_label(
                                centr_reordenado[-1], centroides_referencia, asignacion_previa)
                            asignacion[n_clu - 1][0] = n_clu - 1
                        else:
                            for i in range(len(asg_comp)):
                                asignacion[i] = (asg_comp[i][0], asg_comp[i][1])
                        for i in range(n_clu):
                            count_old = (int)(len(X[np.where(labs_transformados == i)]))
                            proporcion[i] = (i, count_old)

                    for i in range(n_clu):
                        c = Counter(Y_CLU[np.where(Y_CLU == i)])
                        value, count = c.most_common()[0]
                        if len(c) > 1:
                            value2, count2 = c.most_common()[1]
                            if (count - count2) < count / 10:
                                centroides_referencia = np.array(centr)[:, 0:dim]
                                indice_buscado = closest_node(t_c[i], centroides_referencia)
                                if value != (int)(asignacion_previa[indice_buscado][1]):
                                    value = (int)(asignacion_previa[indice_buscado][1])
                        proporcion[i] = (i, count)
                else:
                    for i in range(n_clu):
                        c = Counter(Y[np.where(Y_CLU == i)])
                        value, count = c.most_common()[0]
                        if len(c) > 1:
                            value2, count2 = c.most_common()[1]
                            if (count - count2) < count / 10:
                                centroides_referencia = np.array(centr)[:, 0:dim]
                                indice_buscado = closest_node(tmp_centro[i], centroides_referencia)
                                if value != (int)(asignacion_previa[indice_buscado][1]):
                                    value = (int)(asignacion_previa[indice_buscado][1])
                        asignacion[i] = (i, value)
                        proporcion[i] = (i, count)
                    indice = 0
                    while ((asignacion[indice][1] != asignacion[indice + 1][1]) & (indice + 2 <= n_clu - 1)):
                        if ((asignacion[indice+1][1] != asignacion[indice + 2][1])):#(asignacion[indice][1] != asignacion[indice + 2][1])
                            new_centroides = swapPositions(tmp_centro, indice, -1)
                            kmeansc = KMeans(n_clusters=n_clu, init=np.array(new_centroides), max_iter=MAX_ITERATIONS,
                                             random_state=0, tol=CONVERGENCE_TOLERANCE)
                            kmeansc.fit(X)  # kmeansc.fit(X, Y)
                            tmp_cent = kmeansc.cluster_centers_
                            tmp_centro = tmp_cent.tolist()
                            Y_CLU = kmeansc.labels_
                            lab = [i for i in range(n_clu)]
                            asignacion = np.zeros((n_clu, 2))
                            proporcion = np.zeros((n_clu, 2))
                            for i in range(n_clu):
                                c = Counter(Y[np.where(Y_CLU == i)])
                                value, count = c.most_common()[0]
                                if len(c) > 1:
                                    value2, count2 = c.most_common()[1]
                                    if (count - count2) < count / 10:
                                        centroides_referencia = np.array(centr)[:, 0:dim]
                                        indice_buscado = closest_node(tmp_centro[i], centroides_referencia)
                                        if value != (int)(asignacion_previa[indice_buscado][1]):
                                            value = (int)(asignacion_previa[indice_buscado][1])
                                asignacion[i] = (i, value)
                                proporcion[i] = (i, count)

                        indice += 1
            asignacion = np.array(asignacion)

            kmeanscD = KMeans(n_clusters=n_clu, max_iter=MAX_ITERATIONS, random_state=0, init=np.array(t_c),
                              tol=CONVERGENCE_TOLERANCE)
            kmeanscD.fit(X)
            Y_CLU = kmeanscD.labels_
            t_c = kmeanscD.cluster_centers_
            labs_transformados = Y_CLU

    return labs_transformados,t_c,asignacion,cambio,proporcion

def merge_hungarian(num_clases,lista_puntos_1, lista_puntos_2):
    lista_1 = copy.deepcopy(lista_puntos_1)
    lista_2 = copy.deepcopy(lista_puntos_2)

    lista_patron_utilizar = [[] for i in range(num_clases)]
    lista_2_ordenada = [[] for i in range(num_clases)]

    lista_1_indices_no_usados = [[] for i in range(num_clases)]
    lista_1_no_usados = [[] for i in range(num_clases)]

    lista_2_no_usados = [[] for i in range(num_clases)]
    lista_2_indices_no_usados = [[] for i in range(num_clases)]

    for j in range(num_clases):
        dist_matrix = distance_matrix(lista_1[j], lista_2[j])
        # Apply the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        tam1 = len(lista_puntos_1[j])
        tam2 = len(lista_puntos_2[j])
        if tam1>tam2:
            indices_listapatron = np.arange(tam1)
            cont = 0
            for ix in col_ind:
                lista_2_ordenada[j].append(lista_2[j][ix])
                cont += 1
            for c in indices_listapatron:
                if c not in row_ind:
                    lista_1_indices_no_usados[j].append(c)
                    lista_1_no_usados[j].append(lista_1[j][c])

            lista_indices=[]
            for elemento in indices_listapatron:
                if elemento not in lista_1_indices_no_usados[j]:
                    lista_indices.append(elemento)
            lista_patron_utilizar[j] = np.array(lista_1[j])[lista_indices].tolist()
        else:
            indices_lista2 = np.arange(tam2)
            cont = 0
            for ix in col_ind:
                lista_2_ordenada[j].append(lista_2[j][ix])
                cont += 1
            for c in indices_lista2:
                if c not in row_ind:
                    lista_2_indices_no_usados[j].append(c)
                    lista_2_no_usados[j].append(lista_2[j][c])
            lista_indices = [elemento for indice, elemento in enumerate(indices_lista2) if indice not in lista_2_indices_no_usados[j]]
            lista_patron_utilizar[j] = np.array(lista_1[j])[lista_indices].tolist()

    return lista_patron_utilizar,lista_2_ordenada, lista_1_no_usados
def merge_hungarianGNG(num_clases, lista_puntos_1, lista_puntos_2):
    lista_1= copy.deepcopy(lista_puntos_1)
    lista_2 = copy.deepcopy(lista_puntos_2)

    lista_patron_utilizar = [[] for i in range(num_clases)]
    lista_2_ordenada = [[] for i in range(num_clases)]

    lista_1_indices_no_usados = [[] for i in range(num_clases)]
    lista_1_no_usados = [[] for i in range(num_clases)]

    lista_2_no_usados = [[] for i in range(num_clases)]
    lista_2_indices_no_usados = [[] for i in range(num_clases)]

    for j in range(num_clases):
        dist_matrix = distance_matrix(lista_1[j], lista_2[j])
        # Apply the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        tam1 = len(lista_puntos_1[j])
        tam2 = len(lista_puntos_2[j])
        if tam1>tam2:
            indices_listapatron = np.arange(tam1)
            cont = 0
            for ix in col_ind:
                lista_2_ordenada[j].append(lista_2[j][ix])
                cont += 1
            for c in indices_listapatron:
                if c not in row_ind:
                    lista_1_indices_no_usados[j].append(c)
                    lista_1_no_usados[j].append(lista_1[j][c])

            lista_indices=[]
            for elemento in indices_listapatron:
                if elemento not in lista_1_indices_no_usados[j]:
                    lista_indices.append(elemento)
            lista_patron_utilizar[j] = np.array(lista_1[j])[lista_indices].tolist()
        else:
            indices_lista2 = np.arange(tam2)
            cont = 0
            for ix in col_ind:
                lista_2_ordenada[j].append(lista_2[j][ix])
                cont += 1
            for c in indices_lista2:
                if c not in row_ind:
                    lista_2_indices_no_usados[j].append(c)
                    lista_2_no_usados[j].append(lista_2[j][c])
            lista_indices = [elemento for indice, elemento in enumerate(indices_lista2) if indice not in lista_2_indices_no_usados[j]]
            lista_patron_utilizar[j] = np.array(lista_1[j])[lista_indices].tolist()
    return lista_patron_utilizar, lista_2_ordenada, lista_1_no_usados, lista_1_indices_no_usados,lista_2_no_usados

def funcion_comparativa(Xfeatures,Xlabels,test,metrica,XTRAIN=[],YTRAIN=[],n_c=2,flag=0,Tol_DENSITY=1,dim=0,centroides=0,TOL_radio_clase=0,ident_Z=0,feat_imp=0,ZONA=[]):
    '''
    :param Xfeatures: La lista de puntos ( en el espacio dd las features) de las que tengo certeza de estar bien etiquetadas
    :param Xlabels:  La etiquetas de las instancias anteriores
    :param Xtest:  La lista de nuesvas instancias de las que quiero obtener su etiqueta
    :param metrica: Metrica comparativa, KNN,closest point, ...
    :param Tol_DENSITY: Parametro necesario para la mentrica de densidad. Marca la densidad por clase.
    :return prediccion: Prediccion de las etiquetas de Xtest.
    '''
    prediccion=[]

    if metrica=='KNN5_GNG':
        knn = KNeighborsClassifier(n_neighbors=num_vecinos_entropia,algorithm='ball_tree').fit(Xfeatures,Xlabels)
        prediccion=knn.predict(test).astype(int)
    elif metrica == 'KNN7':
        knn = KNeighborsClassifier(n_neighbors=7,weights = 'distance',algorithm='kd_tree').fit(Xfeatures, Xlabels)
        prediccion = knn.predict(test).astype(int)
    elif metrica == 'KNN13':
        knn = KNeighborsClassifier(n_neighbors=13, weights='distance', algorithm='kd_tree').fit(Xfeatures, Xlabels)
        prediccion = knn.predict(test).astype(int)
    elif metrica=='KNN5':
        if flag != 0:
            knn = KNeighborsClassifier(n_neighbors=num_vecinos_entropia,algorithm='ball_tree').fit(Xfeatures, Xlabels)
            predicciontmp =knn.predict(test).astype(int)
            nbrs = NearestNeighbors(n_neighbors=num_vecinos_entropia, algorithm='ball_tree').fit(Xfeatures)

            conti = 0
            for pto in test:
                distances_centroides = distances_nodes(dim, pto, np.array(centroides)[:, :dim])
                numCLA=len(distances_centroides)

                num_vecinos_con_misma_etiqueta=np.zeros(numCLA)
                distances, indices = nbrs.kneighbors(pto.reshape(1, -1))
                etiquetas_knn = [Xlabels[ind] for ind in (indices)]
                etiquetas_knn = np.array(etiquetas_knn).flatten()
                for j in range(numCLA):
                    num_vecinos_con_misma_etiqueta[j] = np.count_nonzero(etiquetas_knn == j)

                if (num_vecinos_con_misma_etiqueta<=1).any():
                    claseMAYORITARIA=np.argmax(num_vecinos_con_misma_etiqueta)
                    prediccion.append(claseMAYORITARIA)
                else:
                    clase=np.argmin(distances_centroides)
                    prediccion.append(clase)
                conti+=1
        else:
            knn = KNeighborsClassifier(n_neighbors=num_vecinos_entropia,weights = 'distance', algorithm = 'kd_tree').fit(Xfeatures, Xlabels)
            prediccion = knn.predict(test).astype(int)


    return prediccion
def funcion_comparativaREAL(num,Xfeatures,Xlabels,test,metrica,XTRAIN=[],YTRAIN=[],n_c=2,flag=0,Tol_DENSITY=1,dim=0,centroides=0,TOL_radio_clase=0,ident_Z=0,feat_imp=0,ZONA=[]):
    '''
    :param Xfeatures: La lista de puntos ( en el espacio dd las features) de las que tengo certeza de estar bien etiquetadas
    :param Xlabels:  La etiquetas de las instancias anteriores
    :param Xtest:  La lista de nuesvas instancias de las que quiero obtener su etiqueta
    :param metrica: Metrica comparativa, KNN,closest point, ...
    :param Tol_DENSITY: Parametro necesario para la mentrica de densidad. Marca la densidad por clase.
    :return prediccion: Prediccion de las etiquetas de Xtest.
    '''
    prediccion=[]

    if metrica=='KNN5_GNG':
        knn = KNeighborsClassifier(n_neighbors=num,algorithm='ball_tree').fit(Xfeatures,Xlabels)
        prediccion=knn.predict(test).astype(int)
    elif metrica == 'KNN7':
        knn = KNeighborsClassifier(n_neighbors=7,weights = 'distance',algorithm='kd_tree').fit(Xfeatures, Xlabels)
        prediccion = knn.predict(test).astype(int)
    elif metrica == 'KNN13':
        knn = KNeighborsClassifier(n_neighbors=13, weights='distance', algorithm='kd_tree').fit(Xfeatures, Xlabels)
        prediccion = knn.predict(test).astype(int)
    elif metrica=='KNN5':
        if flag != 0:
            knn = KNeighborsClassifier(n_neighbors=num_vecinos_entropia,algorithm='ball_tree').fit(Xfeatures, Xlabels)
            predicciontmp =knn.predict(test).astype(int)
            nbrs = NearestNeighbors(n_neighbors=num_vecinos_entropia, algorithm='ball_tree').fit(Xfeatures)

            conti = 0
            for pto in test:
                distances_centroides = distances_nodes(dim, pto, np.array(centroides)[:, :dim])
                numCLA=len(distances_centroides)

                num_vecinos_con_misma_etiqueta=np.zeros(numCLA)
                distances, indices = nbrs.kneighbors(pto.reshape(1, -1))
                etiquetas_knn = [Xlabels[ind] for ind in (indices)]
                etiquetas_knn = np.array(etiquetas_knn).flatten()
                for j in range(numCLA):
                    num_vecinos_con_misma_etiqueta[j] = np.count_nonzero(etiquetas_knn == j)

                if (num_vecinos_con_misma_etiqueta<=1).any():
                    claseMAYORITARIA=np.argmax(num_vecinos_con_misma_etiqueta)
                    prediccion.append(claseMAYORITARIA)
                else:
                    clase=np.argmin(distances_centroides)
                    prediccion.append(clase)
                conti+=1
        else:
            knn = KNeighborsClassifier(n_neighbors=num_vecinos_entropia,weights = 'distance', algorithm = 'kd_tree').fit(Xfeatures, Xlabels)
            prediccion = knn.predict(test).astype(int)


    return prediccion

def encontrar_codo(distancias):
    puntos = np.array(list(enumerate(distancias)))
    inicio = puntos[0]
    fin = puntos[-1]
    linea = fin - inicio
    linea_norm = linea / np.linalg.norm(linea)
    vectores = puntos - inicio
    proyecciones = np.dot(vectores, linea_norm)
    proyecciones_vector = np.outer(proyecciones, linea_norm)
    distancias_perpendiculares = np.linalg.norm(vectores - proyecciones_vector, axis=1)
    indice_codo = np.argmax(distancias_perpendiculares)
    return indice_codo, distancias_perpendiculares
def funcion_predecir(Xfeatures,Xlabels,test,metrica,XTRAIN=[],YTRAIN=[],n_c=2,num_k=5,flag=0,Tol_DENSITY=1,dim=0,centroides=0,TOL_radio_clase=0,ident_Z=0,feat_imp=0,ZONA=[]):
    '''
        :param Xfeatures: La lista de puntos ( en el espacio dd las features) de las que tengo certeza de estar bien etiquetadas
        :param Xlabels:  La etiquetas de las instancias anteriores
        :param Xtest:  La lista de nuesvas instancias de las que quiero obtener su etiqueta
        :param metrica: Metrica comparativa, KNN,closest point, ...
        :param Tol_DENSITY: Parametro necesario para la mentrica de densidad. Marca la densidad por clase.
        :return prediccion: Prediccion de las etiquetas de Xtest.
        '''
    prediccion = []
    if metrica == 'CENTROIDES':
        if len(ZONA)>0:
            knn = KNeighborsClassifier(n_neighbors=7, weights='distance', algorithm='kd_tree').fit(Xfeatures, Xlabels)
            prediccion_knn = knn.predict(test).astype(int)
            cont_knn = 0
            for elemento in test:
                indice_centroide = closest_node(elemento, np.array(centroides)[:, 0:dim])
                indice_cercano = closest_node(elemento, Xfeatures)
                closest = Xfeatures[indice_cercano]
                predclosest = Xlabels[indice_cercano]

                if ident_Z != 9:
                    if ((prediccion_knn[cont_knn] != indice_centroide) and (elemento[feat_imp] > ZONA[0]) and (
                            elemento[feat_imp] < ZONA[1])):
                        prediccion.append(predclosest)
                    else:
                        prediccion.append(indice_centroide)
                else:
                    prediccion.append((prediccion_knn[cont_knn]))

                cont_knn += 1
            prediccion = np.array(prediccion)
        else:
            knn = KNeighborsClassifier(n_neighbors=13, weights='distance', algorithm='kd_tree').fit(Xfeatures, Xlabels)
            prediccion = knn.predict(test).astype(int)
    elif metrica == 'KMEANS':
        kmeansc = KMeans(n_clusters=n_c, max_iter=MAX_ITERATIONS, random_state=0, init=(np.array(centroides)[:, 0:dim]),
                         tol=CONVERGENCE_TOLERANCE)
        kmeansc.fit(test)
        prediccion = kmeansc.labels_
    elif metrica == '1_NN':
        knnGNG = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree').fit(Xfeatures, Xlabels)
        prediccionGNG = knnGNG.predict(test).astype(int)

        kmeanscCAMBIO = KMeans(n_clusters=n_c, max_iter=MAX_ITERATIONS, random_state=0,
                          init=(np.array(centroides)[:, 0:dim]),
                          tol=CONVERGENCE_TOLERANCE)
        xtrain_kmeans = np.concatenate((np.array(centroides)[:, 0:dim], XTRAIN))
        lab = [i for i in range(n_c)]
        my_y_train_KMEANS = np.concatenate((lab, YTRAIN))
        kmeanscCAMBIO.fit(xtrain_kmeans, my_y_train_KMEANS)
        prediccion_kmeansCAMBIO = kmeanscCAMBIO.predict(test)

        kmeansc = KMeans(n_clusters=n_c, max_iter=MAX_ITERATIONS, random_state=0,
                          init=(np.array(centroides)[:, 0:dim]),
                          tol=CONVERGENCE_TOLERANCE)

        _asociacion_clusters = kmeansc.fit(test).labels_
        new_centr=kmeansc.fit(test).cluster_centers_
        new_lab = kmeansc.predict(new_centr)

        knn = KNeighborsClassifier(n_neighbors=1).fit(new_centr, new_lab)
        prediccion_knn = knn.predict(test).astype(int)
        predic_TMP=[]
        for h in range (len(prediccionGNG)):
            if prediccionGNG[h]==prediccion_knn[h]:
                predic_TMP.append(prediccionGNG[h])
            else:
                predic_TMP.append(prediccion_knn[h])

        if len(set(prediccion_kmeansCAMBIO.tolist())) > 1:
            #prediccion=prediccion_knn
            prediccion = predic_TMP#CAMBIO 2025
        else:
            prediccion=prediccion_kmeansCAMBIO
    elif metrica == 'DBSCAN_old':
        kmeanscCAMBIO = KMeans(n_clusters=n_c, max_iter=MAX_ITERATIONS, random_state=0,
                          init=(np.array(centroides)[:, 0:dim]),
                          tol=CONVERGENCE_TOLERANCE)
        xtrain_kmeans = np.concatenate((np.array(centroides)[:, 0:dim], XTRAIN))
        lab = [i for i in range(n_c)]
        my_y_train_KMEANS = np.concatenate((lab, YTRAIN))
        kmeanscCAMBIO.fit(xtrain_kmeans, my_y_train_KMEANS)
        prediccion_kmeansCAMBIO = kmeanscCAMBIO.predict(test)

        kmeansc = KMeans(n_clusters=n_c, max_iter=MAX_ITERATIONS, random_state=0,
                          init=(np.array(centroides)[:, 0:dim]),
                          tol=CONVERGENCE_TOLERANCE)

        _asociacion_clusters = kmeansc.fit(test).labels_

        tmp=[]
        for l in range(n_c):
            tmp.append(np.sum(_asociacion_clusters == l))

        centroides_new,contorno_new=obtencion_centroide(n_c,test,_asociacion_clusters)

        neighbors = NearestNeighbors(n_neighbors=n_c)
        neighbors_fit = neighbors.fit(test)
        distances, indices = neighbors_fit.kneighbors(test)

        # Sort and plot the distances to find the "elbow"
        distances = np.sort(distances[:, (n_c-1)])  # 4 = min_samples - 1
        indice_codo, dis_codo= encontrar_codo(distances)
        #print(dis_codo[indice_codo],distances[indice_codo])

        '''plt.plot(distances)
        plt.xlabel("Points sorted by distance")
        plt.ylabel("5th Nearest Neighbor Distance")
        plt.title("k-distance Graph")
        plt.grid(True)
        plt.show()'''

        if len(set(prediccion_kmeansCAMBIO.tolist())) > 1:
            prediccion=_asociacion_clusters
        else:
            prediccion=prediccion_kmeansCAMBIO
    elif metrica == 'DBSCAN':
        knn = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree').fit(Xfeatures, Xlabels)
        prediccionGNG = knn.predict(test).astype(int)

        # 1. Estimar densidad usando Kernel Density
        kde = KernelDensity(bandwidth=1.0)#bandwich
        kde.fit(test)

        # ==============================
        # 1.1. densidad rho
        # ==============================
        log_dens = kde.score_samples(test)  # log-densidad en cada punto
        densities = np.exp(log_dens)  # pasamos de log a densidad real

        # ==============================
        # 1.2.  Calcular delta (igual que antes)
        # ==============================
        dists = pairwise_distances(test)

        idx_sorted = np.argsort(-densities)
        N = len(test)
        delta = np.zeros(N)

        delta[idx_sorted[0]] = np.max(dists[idx_sorted[0]])

        for i in idx_sorted[1:]:
            mask = densities > densities[i]
            # delta[i] = np.min(dists[i][mask])
            if np.any(mask):
                delta[i] = np.min(dists[i][mask])
            else:
                # caso donde NO hay ningún punto con densidad mayor
                # asigna la distancia máxima como fallback
                if mask.sum() == 0:
                    print(f"El punto {i} tiene la densidad máxima (ningún vecino con mayor densidad).")
                delta[i] = np.max(dists[i])

        # Los puntos que se distinguen (cercanos a ser centroides)
        cond = delta > 3.0
        # Puntos filtrados
        X_fitrados = test[cond]
        indices_filtrados = np.where(cond)[0]
        dens_filtrados = densities[indices_filtrados]
        delta_filtrados = delta[indices_filtrados]
        print(X_fitrados, indices_filtrados, dens_filtrados, delta_filtrados)

        #draw_densityPLot(densities, delta)

        # 2. Asignar cada punto al centro más cercano
        distances = np.linalg.norm(test[:, None, :] - np.array(centroides)[None, :,0:dim], axis=2)
        nearest_centroid = np.argmin(distances, axis=1)

        # 3. Opcional: Filtrar puntos de baja densidad
        density_threshold = np.percentile(densities, 2)  #percent_num # por ejemplo, 2% más baja
        # print("Threshold density 2 ", density_threshold)
        labels_incertibumbre = nearest_centroid.copy()
        prediccion = nearest_centroid.copy()
        labels_incertibumbre[densities < density_threshold] = -1  # etiqueta -1 = ruido/baja densidad

        # Aqui es donde utilizo los nodos GNG
        for elem in range(len(labels_incertibumbre)):
            if labels_incertibumbre[elem] == -1:
                if nearest_centroid[elem] != prediccionGNG[elem]:
                    #print(distances[elem], prediccionGNG[elem], nearest_centroid[elem])
                    prediccion[elem] = prediccionGNG[elem]
    elif metrica == 'COMP':
        prediccion_centroide = []
        prediccion_close = []

        kmeansc1 = KMeans(n_clusters=n_c, max_iter=MAX_ITERATIONS, random_state=0,
                          init=(np.array(centroides)[:, 0:dim]),
                          tol=CONVERGENCE_TOLERANCE)
        kmeansc2 = KMeans(n_clusters=n_c, max_iter=MAX_ITERATIONS, random_state=0,
                          init=(np.array(centroides)[:, 0:dim]),
                          tol=CONVERGENCE_TOLERANCE)

        xtrain_kmeans = np.concatenate((np.array(centroides)[:, 0:dim], XTRAIN))
        lab = [i for i in range(n_c)]
        my_y_train_KMEANS = np.concatenate((lab, YTRAIN))
        kmeansc1.fit(xtrain_kmeans, my_y_train_KMEANS)
        prediccion_kmeans1 = kmeansc1.predict(test)
        Xlabels_train = np.concatenate((lab, Xlabels))
        kmeansc2.fit(Xfeatures, Xlabels_train)
        prediccion_kmeans2 = kmeansc2.predict(test)

        knn = KNeighborsClassifier(n_neighbors=num_vecinos_entropia, algorithm='ball_tree').fit(Xfeatures, Xlabels)
        prediccion_knn = knn.predict(test).astype(int)

        for elemento in test:
            indice_centroide = closest_node(elemento, np.array(centroides)[:, 0:dim])
            prediccion_centroide.append(indice_centroide)
            indice = closest_node(elemento, Xfeatures)
            closest = Xfeatures[indice]
            predclosest = Xlabels[indice]
            prediccion_close.append(predclosest)

        tamaño_muestra = test.shape[0]
        for i in range(tamaño_muestra):
            if prediccion_kmeans1[i] != prediccion_kmeans2[i]:

                if prediccion_knn[i] != prediccion_kmeans1[i]:
                    if ((prediccion_knn[i] == prediccion_kmeans2[i]) & (prediccion_knn[i] != prediccion_close[i])):
                        prediccion.append(prediccion_knn[i])
                    else:
                        prediccion.append(prediccion_kmeans1[i])
                else:
                    prediccion.append(prediccion_knn[i])
            else:
                prediccion.append(prediccion_kmeans1[i])

    elif metrica == 'KNN5_GNG':
        knn = KNeighborsClassifier(n_neighbors=num_vecinos_entropia, algorithm='ball_tree').fit(Xfeatures, Xlabels)
        prediccion = knn.predict(test).astype(int)
    elif metrica == 'KNN7':
        knn = KNeighborsClassifier(n_neighbors=7, weights='distance', algorithm='kd_tree').fit(Xfeatures, Xlabels)
        prediccion = knn.predict(test).astype(int)
    elif metrica == 'KNN13':
        knn = KNeighborsClassifier(n_neighbors=13, weights='distance', algorithm='kd_tree').fit(Xfeatures, Xlabels)
        prediccion = knn.predict(test).astype(int)
    elif metrica == 'KNN5':
        if flag != 0:
            knn = KNeighborsClassifier(n_neighbors=num_vecinos_entropia, algorithm='ball_tree').fit(Xfeatures, Xlabels)
            predicciontmp = knn.predict(test).astype(int)
            nbrs = NearestNeighbors(n_neighbors=num_vecinos_entropia, algorithm='ball_tree').fit(Xfeatures)

            conti = 0
            for pto in test:
                distances_centroides = distances_nodes(dim, pto, np.array(centroides)[:, :dim])
                numCLA = len(distances_centroides)

                num_vecinos_con_misma_etiqueta = np.zeros(numCLA)
                distances, indices = nbrs.kneighbors(pto.reshape(1, -1))
                etiquetas_knn = [Xlabels[ind] for ind in (indices)]
                etiquetas_knn = np.array(etiquetas_knn).flatten()
                for j in range(numCLA):
                    num_vecinos_con_misma_etiqueta[j] = np.count_nonzero(etiquetas_knn == j)

                if (num_vecinos_con_misma_etiqueta <= 1).any():
                    claseMAYORITARIA = np.argmax(num_vecinos_con_misma_etiqueta)
                    prediccion.append(claseMAYORITARIA)
                else:
                    clase = np.argmin(distances_centroides)
                    prediccion.append(clase)
                conti += 1
        else:
            knn = KNeighborsClassifier(n_neighbors=num_vecinos_entropia, weights='distance', algorithm='kd_tree').fit(
                Xfeatures, Xlabels)
            prediccion = knn.predict(test).astype(int)
    elif metrica == 'UNBALANCE':
        global tolerancia_traslacion
        numCLA = len(TOL_radio_clase)
        for u in range(numCLA - 1):
            tmp = distances_nodes(dim, np.array(centroides)[u, :dim], np.array(centroides)[u + 1:, :dim]).min()
            if u == 0:
                dist = tmp
            else:
                dist = np.min((np.min(dist), np.min(tmp)), axis=0)
        if ((dist / 4 < tolerancia_traslacion * 4) & (dist / 4 >= 0.75)):
            tolerancia_traslacion = dist / 4
        else:
            tolerancia_traslacion = 0.75

        flag_diferencia_GR = 0
        minimo_radio = np.array(TOL_radio_clase).min()
        clase_pequeña = np.argmin(TOL_radio_clase)
        clase_grande = np.argmax(TOL_radio_clase)
        maximo_radio = np.array(TOL_radio_clase).max()

        if (maximo_radio - minimo_radio) > minimo_radio:
            flag_diferencia_GR = 1

        if flag == 0:
            knn = KNeighborsClassifier(n_neighbors=num_vecinos_entropia, algorithm='ball_tree').fit(Xfeatures, Xlabels)
            predicciontmp = knn.predict(test).astype(int)
            nbrs = NearestNeighbors(n_neighbors=num_vecinos_entropia, algorithm='ball_tree').fit(Xfeatures)
            nbrs0 = NearestNeighbors(n_neighbors=num_vecinos_entropia + 1, algorithm='ball_tree').fit(test)

            minimo = np.array(Tol_DENSITY).min()
            maximo = np.array(Tol_DENSITY).max()

            conti = 0
            for pto in test:
                distances_centroides = distances_nodes(dim, pto, np.array(centroides)[:, :dim])

                num_vecinos_con_misma_etiqueta = np.zeros(numCLA)
                distances, indices = nbrs.kneighbors(pto.reshape(1, -1))
                etiquetas_knn = [Xlabels[ind] for ind in (indices)]
                etiquetas_knn = np.array(etiquetas_knn).flatten()
                for j in range(numCLA):
                    num_vecinos_con_misma_etiqueta[j] = np.count_nonzero(etiquetas_knn == j)

                if (num_vecinos_con_misma_etiqueta <= 1).any():
                    claseMAYORITARIA = np.argmax(num_vecinos_con_misma_etiqueta)
                    prediccion.append(claseMAYORITARIA)
                else:
                    if (distances_centroides[clase_pequeña] <= TOL_radio_clase[clase_pequeña] + tolerancia_traslacion):
                        prediccion.append(clase_pequeña)
                    else:
                        minimos_dim = np.array(centroides)[:, 0:dim].min(axis=0)
                        maximos_dim = np.array(centroides)[:, 0:dim].max(axis=0)

                        distances0, indices0 = nbrs0.kneighbors(pto.reshape(1, -1))
                        distances0 = np.array(distances0).flatten()
                        distancia_media_vecinos = distances0[1:].mean()
                        clase = -1
                        if (distances0[1:] <= minimo).all():
                            clase = np.argmin(Tol_DENSITY)
                        elif (distances0[1:] >= maximo).all():
                            clase = np.argmax(Tol_DENSITY)
                        else:
                            clase = closest_node_dim(1, distancia_media_vecinos, Tol_DENSITY)

                        for i in range(dim - 1):
                            if ((minimos_dim[i] < pto[i] < maximos_dim[i]) & (
                                    minimos_dim[i + 1] < pto[i + 1] < maximos_dim[i + 1])):
                                prediccion.append(clase)
                                break
                            else:
                                prediccion.append(predicciontmp[conti])
                                break

                conti += 1
                if conti > test.shape[0]:
                    print('ERROR')
        else:
            knn = KNeighborsClassifier(n_neighbors=num_vecinos_entropia, algorithm='ball_tree').fit(Xfeatures, Xlabels)
            prediccionTMP = knn.predict(test).astype(int)

            minimo = np.array(Tol_DENSITY).min()
            maximo = np.array(Tol_DENSITY).max()
            nbrs = NearestNeighbors(n_neighbors=num_vecinos_entropia + 1, algorithm='ball_tree').fit(test)
            cont2 = 0
            for pto in test:
                indiceclose = closest_node(pto, Xfeatures)
                closest = Xfeatures[indiceclose]
                predclosest = Xlabels[indiceclose].astype(int)

                distances, indices = nbrs.kneighbors(pto.reshape(1, -1))
                distances = np.array(distances).flatten()
                distancia_media_vecinos = distances[1:].mean()
                if (distances[1:] <= minimo).all():
                    clase = np.argmin(Tol_DENSITY)
                elif (distances[1:] >= maximo).all():
                    clase = np.argmax(Tol_DENSITY)
                else:
                    clase = closest_node_dim(1, distancia_media_vecinos, Tol_DENSITY)

                if (clase == prediccionTMP[cont2]):
                    prediccion.append(clase)
                else:
                    if dim == 2:
                        x_point = pto[0] - centroides[clase_pequeña][0]
                        y_point = pto[1] - centroides[clase_pequeña][1]
                        hypotenuse = math.sqrt(x_point ** 2 + y_point ** 2)
                    elif dim == 3:
                        x_point = pto[0] - centroides[clase_pequeña][0]
                        y_point = pto[1] - centroides[clase_pequeña][1]
                        z_point = pto[2] - centroides[clase_pequeña][2]
                        hypotenuse = math.sqrt(x_point ** 2 + y_point ** 2 + z_point ** 2)
                    else:
                        hypotenuse = distance.euclidean(pto, centroides[clase_pequeña].flatten())

                    if hypotenuse <= TOL_radio_clase[clase_pequeña] + tolerancia_traslacion:

                        prediccion.append(clase_pequeña)
                    elif (flag_diferencia_GR == 1) & (prediccionTMP[cont2] != clase_pequeña):
                        prediccion.append(clase)
                    else:
                        prediccion.append(prediccionTMP[cont2])
                cont2 += 1

    return prediccion
def calcular_centroidesREAL(X, y):
    centroides = {}
    for clase in np.unique(y):
        centroides[clase] = X[y == clase].mean(axis=0)
    return centroides

from scipy.spatial.distance import euclidean
def funcion_predecirREAL(Xfeatures,Xlabels,test,metrica,XTRAIN=[],YTRAIN=[],n_c=2,num_k=5,flag=0,flagR=0,Tol_DENSITY=1,dim=0,centroides=0,TOL_radio_clase=0,ident_Z=0,feat_imp=0,ZONA=[]):
    '''
        :param Xfeatures: La lista de puntos ( en el espacio dd las features) de las que tengo certeza de estar bien etiquetadas
        :param Xlabels:  La etiquetas de las instancias anteriores
        :param Xtest:  La lista de nuesvas instancias de las que quiero obtener su etiqueta
        :param metrica: Metrica comparativa, KNN,closest point, ...
        :param Tol_DENSITY: Parametro necesario para la mentrica de densidad. Marca la densidad por clase.
        :return prediccion: Prediccion de las etiquetas de Xtest.
        '''
    prediccion = []
    if metrica == 'CENTROIDES':
        if len(ZONA)>0:
            knn = KNeighborsClassifier(n_neighbors=7, weights='distance', algorithm='kd_tree').fit(Xfeatures, Xlabels)
            prediccion_knn = knn.predict(test).astype(int)
            cont_knn = 0
            for elemento in test:
                indice_centroide = closest_node(elemento, np.array(centroides)[:, 0:dim])
                indice_cercano = closest_node(elemento, Xfeatures)
                closest = Xfeatures[indice_cercano]
                predclosest = Xlabels[indice_cercano]

                if ident_Z != 9:
                    if ((prediccion_knn[cont_knn] != indice_centroide) and (elemento[feat_imp] > ZONA[0]) and (
                            elemento[feat_imp] < ZONA[1])):
                        prediccion.append(predclosest)
                    else:
                        prediccion.append(indice_centroide)
                else:
                    prediccion.append((prediccion_knn[cont_knn]))

                cont_knn += 1
            prediccion = np.array(prediccion)
        else:
            knn = KNeighborsClassifier(n_neighbors=13, weights='distance', algorithm='kd_tree').fit(Xfeatures, Xlabels)
            prediccion = knn.predict(test).astype(int)
    elif metrica == 'KMEANS':
        kmeansc = KMeans(n_clusters=n_c, max_iter=MAX_ITERATIONS, random_state=0, init=(np.array(centroides)[:, 0:dim]),
                         tol=CONVERGENCE_TOLERANCE)
        kmeansc.fit(test)
        prediccion = kmeansc.labels_
    elif metrica == '1_NN':
        scaler = StandardScaler()
        Xfeatures = scaler.fit_transform(Xfeatures)
        test = scaler.transform(test)

        knnGNG = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree').fit(Xfeatures, Xlabels)

        distancias, indices = knnGNG.kneighbors(test)
        dist_media = distancias.mean(axis=1)

        dist_train, _ = knnGNG.kneighbors(Xfeatures)
        dist_train_media = dist_train.mean(axis=1)

        umbral_global = np.mean(dist_train_media) + 2 * np.std(dist_train_media)

        # OPT1:
        # prediccion = knnGNG.predict(test)
        # prediccion[dist_media > umbral_global] = 100
        # OPT2:
        prediccion_base = knnGNG.predict(test)
        probs = knnGNG.predict_proba(test)
        confianza = probs.max(axis=1)

        umbral_conf = 0.55  # puedes ajustarlo:no varia

        mask_rechazo = (dist_media > umbral_global) | (confianza < umbral_conf) # &

        prediccion = prediccion_base.copy()
        prediccion[mask_rechazo] = 100

    elif metrica == 'DBSCAN_old':
        kmeanscCAMBIO = KMeans(n_clusters=n_c, max_iter=MAX_ITERATIONS, random_state=0,
                          init=(np.array(centroides)[:, 0:dim]),
                          tol=CONVERGENCE_TOLERANCE)
        xtrain_kmeans = np.concatenate((np.array(centroides)[:, 0:dim], XTRAIN))
        lab = [i for i in range(n_c)]
        my_y_train_KMEANS = np.concatenate((lab, YTRAIN))
        kmeanscCAMBIO.fit(xtrain_kmeans, my_y_train_KMEANS)
        prediccion_kmeansCAMBIO = kmeanscCAMBIO.predict(test)

        kmeansc = KMeans(n_clusters=n_c, max_iter=MAX_ITERATIONS, random_state=0,
                          init=(np.array(centroides)[:, 0:dim]),
                          tol=CONVERGENCE_TOLERANCE)

        _asociacion_clusters = kmeansc.fit(test).labels_

        tmp=[]
        for l in range(n_c):
            tmp.append(np.sum(_asociacion_clusters == l))

        centroides_new,contorno_new=obtencion_centroide(n_c,test,_asociacion_clusters)

        neighbors = NearestNeighbors(n_neighbors=n_c)
        neighbors_fit = neighbors.fit(test)
        distances, indices = neighbors_fit.kneighbors(test)

        # Sort and plot the distances to find the "elbow"
        distances = np.sort(distances[:, (n_c-1)])  # 4 = min_samples - 1
        indice_codo, dis_codo= encontrar_codo(distances)
        #print(dis_codo[indice_codo],distances[indice_codo])

        '''plt.plot(distances)
        plt.xlabel("Points sorted by distance")
        plt.ylabel("5th Nearest Neighbor Distance")
        plt.title("k-distance Graph")
        plt.grid(True)
        plt.show()'''

        if len(set(prediccion_kmeansCAMBIO.tolist())) > 1:
            prediccion=_asociacion_clusters
        else:
            prediccion=prediccion_kmeansCAMBIO
    elif metrica == 'DBSCAN':
        knn = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree').fit(Xfeatures, Xlabels)
        prediccionGNG = knn.predict(test).astype(int)

        # 1. Estimar densidad usando Kernel Density
        kde = KernelDensity(bandwidth=1.0)#bandwich
        kde.fit(test)

        # ==============================
        # 1.1. densidad rho
        # ==============================
        log_dens = kde.score_samples(test)  # log-densidad en cada punto
        densities = np.exp(log_dens)  # pasamos de log a densidad real

        # ==============================
        # 1.2.  Calcular delta (igual que antes)
        # ==============================
        dists = pairwise_distances(test)

        idx_sorted = np.argsort(-densities)
        N = len(test)
        delta = np.zeros(N)

        delta[idx_sorted[0]] = np.max(dists[idx_sorted[0]])

        for i in idx_sorted[1:]:
            mask = densities > densities[i]
            # delta[i] = np.min(dists[i][mask])
            if np.any(mask):
                delta[i] = np.min(dists[i][mask])
            else:
                # caso donde NO hay ningún punto con densidad mayor
                # asigna la distancia máxima como fallback
                if mask.sum() == 0:
                    print(f"El punto {i} tiene la densidad máxima (ningún vecino con mayor densidad).")
                delta[i] = np.max(dists[i])

        # Los puntos que se distinguen (cercanos a ser centroides)
        cond = delta > 3.0
        # Puntos filtrados
        X_fitrados = test[cond]
        indices_filtrados = np.where(cond)[0]
        dens_filtrados = densities[indices_filtrados]
        delta_filtrados = delta[indices_filtrados]
        print(X_fitrados, indices_filtrados, dens_filtrados, delta_filtrados)

        #draw_densityPLot(densities, delta)

        # 2. Asignar cada punto al centro más cercano
        distances = np.linalg.norm(test[:, None, :] - np.array(centroides)[None, :,0:dim], axis=2)
        nearest_centroid = np.argmin(distances, axis=1)

        # 3. Opcional: Filtrar puntos de baja densidad
        density_threshold = np.percentile(densities, 2)  #percent_num # por ejemplo, 2% más baja
        # print("Threshold density 2 ", density_threshold)
        labels_incertibumbre = nearest_centroid.copy()
        prediccion = nearest_centroid.copy()
        labels_incertibumbre[densities < density_threshold] = -1  # etiqueta -1 = ruido/baja densidad

        # Aqui es donde utilizo los nodos GNG
        for elem in range(len(labels_incertibumbre)):
            if labels_incertibumbre[elem] == -1:
                if nearest_centroid[elem] != prediccionGNG[elem]:
                    #print(distances[elem], prediccionGNG[elem], nearest_centroid[elem])
                    prediccion[elem] = prediccionGNG[elem]
    elif metrica == 'COMP':
        prediccion_centroide = []
        prediccion_close = []

        kmeansc1 = KMeans(n_clusters=n_c, max_iter=MAX_ITERATIONS, random_state=0,
                          init=(np.array(centroides)[:, 0:dim]),
                          tol=CONVERGENCE_TOLERANCE)
        kmeansc2 = KMeans(n_clusters=n_c, max_iter=MAX_ITERATIONS, random_state=0,
                          init=(np.array(centroides)[:, 0:dim]),
                          tol=CONVERGENCE_TOLERANCE)

        xtrain_kmeans = np.concatenate((np.array(centroides)[:, 0:dim], XTRAIN))
        lab = [i for i in range(n_c)]
        my_y_train_KMEANS = np.concatenate((lab, YTRAIN))
        kmeansc1.fit(xtrain_kmeans, my_y_train_KMEANS)
        prediccion_kmeans1 = kmeansc1.predict(test)
        Xlabels_train = np.concatenate((lab, Xlabels))
        kmeansc2.fit(Xfeatures, Xlabels_train)
        prediccion_kmeans2 = kmeansc2.predict(test)

        knn = KNeighborsClassifier(n_neighbors=num_vecinos_entropia, algorithm='ball_tree').fit(Xfeatures, Xlabels)
        prediccion_knn = knn.predict(test).astype(int)

        for elemento in test:
            indice_centroide = closest_node(elemento, np.array(centroides)[:, 0:dim])
            prediccion_centroide.append(indice_centroide)
            indice = closest_node(elemento, Xfeatures)
            closest = Xfeatures[indice]
            predclosest = Xlabels[indice]
            prediccion_close.append(predclosest)

        tamaño_muestra = test.shape[0]
        for i in range(tamaño_muestra):
            if prediccion_kmeans1[i] != prediccion_kmeans2[i]:

                if prediccion_knn[i] != prediccion_kmeans1[i]:
                    if ((prediccion_knn[i] == prediccion_kmeans2[i]) & (prediccion_knn[i] != prediccion_close[i])):
                        prediccion.append(prediccion_knn[i])
                    else:
                        prediccion.append(prediccion_kmeans1[i])
                else:
                    prediccion.append(prediccion_knn[i])
            else:
                prediccion.append(prediccion_kmeans1[i])

    elif metrica == 'KNN5_GNG':
        knn = KNeighborsClassifier(n_neighbors=num_vecinos_entropia, algorithm='ball_tree').fit(Xfeatures, Xlabels)
        prediccion = knn.predict(test).astype(int)
    elif metrica == 'KNN7':
        knn = KNeighborsClassifier(n_neighbors=7, weights='distance', algorithm='kd_tree').fit(Xfeatures, Xlabels)
        prediccion = knn.predict(test).astype(int)
    elif metrica == 'KNN13':
        knn = KNeighborsClassifier(n_neighbors=13, weights='distance', algorithm='kd_tree').fit(Xfeatures, Xlabels)
        prediccion = knn.predict(test).astype(int)
    elif metrica == 'KNN5':
        if flag != 0:
            knn = KNeighborsClassifier(n_neighbors=num_vecinos_entropia, algorithm='ball_tree').fit(Xfeatures, Xlabels)
            predicciontmp = knn.predict(test).astype(int)
            nbrs = NearestNeighbors(n_neighbors=num_vecinos_entropia, algorithm='ball_tree').fit(Xfeatures)

            conti = 0
            for pto in test:
                distances_centroides = distances_nodes(dim, pto, np.array(centroides)[:, :dim])
                numCLA = len(distances_centroides)

                num_vecinos_con_misma_etiqueta = np.zeros(numCLA)
                distances, indices = nbrs.kneighbors(pto.reshape(1, -1))
                etiquetas_knn = [Xlabels[ind] for ind in (indices)]
                etiquetas_knn = np.array(etiquetas_knn).flatten()
                for j in range(numCLA):
                    num_vecinos_con_misma_etiqueta[j] = np.count_nonzero(etiquetas_knn == j)

                if (num_vecinos_con_misma_etiqueta <= 1).any():
                    claseMAYORITARIA = np.argmax(num_vecinos_con_misma_etiqueta)
                    prediccion.append(claseMAYORITARIA)
                else:
                    clase = np.argmin(distances_centroides)
                    prediccion.append(clase)
                conti += 1
        else:
            knn = KNeighborsClassifier(n_neighbors=num_vecinos_entropia, weights='distance', algorithm='kd_tree').fit(
                Xfeatures, Xlabels)
            prediccion = knn.predict(test).astype(int)
    elif metrica == 'UNBALANCE':
        global tolerancia_traslacion
        numCLA = len(TOL_radio_clase)
        for u in range(numCLA - 1):
            tmp = distances_nodes(dim, np.array(centroides)[u, :dim], np.array(centroides)[u + 1:, :dim]).min()
            if u == 0:
                dist = tmp
            else:
                dist = np.min((np.min(dist), np.min(tmp)), axis=0)
        if ((dist / 4 < tolerancia_traslacion * 4) & (dist / 4 >= 0.75)):
            tolerancia_traslacion = dist / 4
        else:
            tolerancia_traslacion = 0.75

        flag_diferencia_GR = 0
        minimo_radio = np.array(TOL_radio_clase).min()
        clase_pequeña = np.argmin(TOL_radio_clase)
        clase_grande = np.argmax(TOL_radio_clase)
        maximo_radio = np.array(TOL_radio_clase).max()

        if (maximo_radio - minimo_radio) > minimo_radio:
            flag_diferencia_GR = 1

        if flag == 0:
            knn = KNeighborsClassifier(n_neighbors=num_vecinos_entropia, algorithm='ball_tree').fit(Xfeatures, Xlabels)
            predicciontmp = knn.predict(test).astype(int)
            nbrs = NearestNeighbors(n_neighbors=num_vecinos_entropia, algorithm='ball_tree').fit(Xfeatures)
            nbrs0 = NearestNeighbors(n_neighbors=num_vecinos_entropia + 1, algorithm='ball_tree').fit(test)

            minimo = np.array(Tol_DENSITY).min()
            maximo = np.array(Tol_DENSITY).max()

            conti = 0
            for pto in test:
                distances_centroides = distances_nodes(dim, pto, np.array(centroides)[:, :dim])

                num_vecinos_con_misma_etiqueta = np.zeros(numCLA)
                distances, indices = nbrs.kneighbors(pto.reshape(1, -1))
                etiquetas_knn = [Xlabels[ind] for ind in (indices)]
                etiquetas_knn = np.array(etiquetas_knn).flatten()
                for j in range(numCLA):
                    num_vecinos_con_misma_etiqueta[j] = np.count_nonzero(etiquetas_knn == j)

                if (num_vecinos_con_misma_etiqueta <= 1).any():
                    claseMAYORITARIA = np.argmax(num_vecinos_con_misma_etiqueta)
                    prediccion.append(claseMAYORITARIA)
                else:
                    if (distances_centroides[clase_pequeña] <= TOL_radio_clase[clase_pequeña] + tolerancia_traslacion):
                        prediccion.append(clase_pequeña)
                    else:
                        minimos_dim = np.array(centroides)[:, 0:dim].min(axis=0)
                        maximos_dim = np.array(centroides)[:, 0:dim].max(axis=0)

                        distances0, indices0 = nbrs0.kneighbors(pto.reshape(1, -1))
                        distances0 = np.array(distances0).flatten()
                        distancia_media_vecinos = distances0[1:].mean()
                        clase = -1
                        if (distances0[1:] <= minimo).all():
                            clase = np.argmin(Tol_DENSITY)
                        elif (distances0[1:] >= maximo).all():
                            clase = np.argmax(Tol_DENSITY)
                        else:
                            clase = closest_node_dim(1, distancia_media_vecinos, Tol_DENSITY)

                        for i in range(dim - 1):
                            if ((minimos_dim[i] < pto[i] < maximos_dim[i]) & (
                                    minimos_dim[i + 1] < pto[i + 1] < maximos_dim[i + 1])):
                                prediccion.append(clase)
                                break
                            else:
                                prediccion.append(predicciontmp[conti])
                                break

                conti += 1
                if conti > test.shape[0]:
                    print('ERROR')
        else:
            knn = KNeighborsClassifier(n_neighbors=num_vecinos_entropia, algorithm='ball_tree').fit(Xfeatures, Xlabels)
            prediccionTMP = knn.predict(test).astype(int)

            minimo = np.array(Tol_DENSITY).min()
            maximo = np.array(Tol_DENSITY).max()
            nbrs = NearestNeighbors(n_neighbors=num_vecinos_entropia + 1, algorithm='ball_tree').fit(test)
            cont2 = 0
            for pto in test:
                indiceclose = closest_node(pto, Xfeatures)
                closest = Xfeatures[indiceclose]
                predclosest = Xlabels[indiceclose].astype(int)

                distances, indices = nbrs.kneighbors(pto.reshape(1, -1))
                distances = np.array(distances).flatten()
                distancia_media_vecinos = distances[1:].mean()
                if (distances[1:] <= minimo).all():
                    clase = np.argmin(Tol_DENSITY)
                elif (distances[1:] >= maximo).all():
                    clase = np.argmax(Tol_DENSITY)
                else:
                    clase = closest_node_dim(1, distancia_media_vecinos, Tol_DENSITY)

                if (clase == prediccionTMP[cont2]):
                    prediccion.append(clase)
                else:
                    if dim == 2:
                        x_point = pto[0] - centroides[clase_pequeña][0]
                        y_point = pto[1] - centroides[clase_pequeña][1]
                        hypotenuse = math.sqrt(x_point ** 2 + y_point ** 2)
                    elif dim == 3:
                        x_point = pto[0] - centroides[clase_pequeña][0]
                        y_point = pto[1] - centroides[clase_pequeña][1]
                        z_point = pto[2] - centroides[clase_pequeña][2]
                        hypotenuse = math.sqrt(x_point ** 2 + y_point ** 2 + z_point ** 2)
                    else:
                        hypotenuse = distance.euclidean(pto, centroides[clase_pequeña].flatten())

                    if hypotenuse <= TOL_radio_clase[clase_pequeña] + tolerancia_traslacion:

                        prediccion.append(clase_pequeña)
                    elif (flag_diferencia_GR == 1) & (prediccionTMP[cont2] != clase_pequeña):
                        prediccion.append(clase)
                    else:
                        prediccion.append(prediccionTMP[cont2])
                cont2 += 1

    return prediccion
# =====================================================================
# PREDICCIÓN OSR SOBRE LOS NODOS GNG
#   '1_NN'    : k-NN sobre nodos + umbral de distancia y de confianza
#   'KDE_EVT' : densidad con kernel RBF sobre nodos + calibración EVT
# =====================================================================
from scipy.special import logsumexp
from scipy.stats import weibull_min

# Mismo k en validación y en predicción: los umbrales se ajustan con este k
K_VECINOS_OSR = 5
# Fracción de muestras conocidas de validación que los umbrales deben aceptar.
# Sustituye a max/min, que dejaban los umbrales en manos de un solo outlier.
COBERTURA_CONOCIDAS = 0.95
# Factores de ancho de banda probados en validación para 'KDE_EVT'
H_CANDIDATOS_KDE = (0.5, 1.0, 2.0)


def _escalar(Xfeatures, test, scaler):
    """Aplica el scaler común. Sin scaler se ajusta sobre los nodos (comportamiento antiguo)."""
    if scaler is None:
        scaler = StandardScaler().fit(Xfeatures)
    return scaler.transform(Xfeatures), scaler.transform(test)


def _kkc_numericas(kkc):
    """Letter: 'A'..'Z' -> 1..26, como target_to_numeric. Pendigits: sin cambios."""
    kkc_eval = []
    for k in np.array(kkc).flatten():
        s = str(k).strip()
        if len(s) == 1 and s.isalpha():
            kkc_eval.append(ord(s.lower()) - ord('a') + 1)
        else:
            kkc_eval.append(k)
    return kkc_eval


class ModeloKDE_EVT:
    '''
    Score de pertenencia por clase con kernel RBF centrado en los nodos GNG:

        log s_c(x) = logsumexp_{j en c} ( -||x - w_j||^2 / (2 (h*sigma_j)^2) )

    sigma_j es la distancia media del nodo j a sus m nodos más cercanos de su
    misma clase (aproxima la longitud de sus aristas GNG), así que el ancho del
    kernel se adapta a la densidad local de cada zona.

    Calibración EVT (como W-SVM / EVM): para cada clase c se ajusta una Weibull
    a la cola de u = -log s_c(x) sobre las muestras de entrenamiento de c.
    P_incl(x, c) = 1 - F_Weibull(u_c(x)). La clase predicha es argmax_c s_c(x)
    y su P_incl es el score de "conocida" (rechazo si P_incl < delta).
    '''
    def __init__(self, h=1.0, m_vecinos=3, cola=0.2, min_cola=20, bloque=2048):
        self.h = h
        self.m_vecinos = m_vecinos
        self.cola = cola
        self.min_cola = min_cola
        self.bloque = bloque

    def fit(self, nodos, etiquetas, Xtrain, ytrain, scaler=None):
        nodos = np.asarray(nodos, dtype=float)
        Xtrain = np.asarray(Xtrain, dtype=float)
        if scaler is None:
            scaler = StandardScaler().fit(Xtrain)
        self.scaler = scaler
        W = scaler.transform(nodos)
        et = np.asarray(etiquetas).astype(int)
        self.clases = np.unique(et)

        # Anchos por nodo
        sigmas = {}
        for c in self.clases:
            Wc = W[et == c]
            if len(Wc) > 1:
                m = min(self.m_vecinos, len(Wc) - 1)
                D = cdist(Wc, Wc)
                np.fill_diagonal(D, np.inf)
                sigmas[c] = np.sort(D, axis=1)[:, :m].mean(axis=1)
            else:
                sigmas[c] = None
        conocidos = np.concatenate([s for s in sigmas.values() if s is not None]) \
            if any(s is not None for s in sigmas.values()) else np.array([1.0])
        sigma_def = float(np.median(conocidos))
        self.W, self.inv2s2 = {}, {}
        for c in self.clases:
            s = sigmas[c] if sigmas[c] is not None else np.array([sigma_def])
            s = np.maximum(s, 1e-6) * self.h
            self.W[c] = W[et == c]
            self.inv2s2[c] = 1.0 / (2.0 * s ** 2)

        # Calibración EVT con las muestras de entrenamiento de cada clase
        S = self._log_scores(scaler.transform(Xtrain))
        ytrain = np.asarray(ytrain).astype(int)
        self.weibull = {}
        for i, c in enumerate(self.clases):
            u = -S[ytrain == c, i]
            if len(u) == 0:
                u = -S[:, i]
            n_cola = int(min(len(u), max(self.min_cola, np.ceil(self.cola * len(u)))))
            tail = np.sort(u)[-n_cola:]
            t0 = tail.min() - 1e-9
            try:
                forma, _, escala = weibull_min.fit(tail - t0, floc=0)
                if not (np.isfinite(forma) and np.isfinite(escala) and escala > 0):
                    raise ValueError
            except Exception:
                forma, escala = 1.0, max(float(np.std(tail)), 1e-6)
            self.weibull[c] = (t0, forma, escala)
        return self

    def _log_scores(self, Xs):
        S = np.empty((len(Xs), len(self.clases)))
        for ini in range(0, len(Xs), self.bloque):
            blq = Xs[ini:ini + self.bloque]
            for i, c in enumerate(self.clases):
                d2 = cdist(blq, self.W[c], 'sqeuclidean')
                S[ini:ini + self.bloque, i] = logsumexp(-d2 * self.inv2s2[c][None, :], axis=1)
        return S

    def predecir(self, X):
        S = self._log_scores(self.scaler.transform(np.asarray(X, dtype=float)))
        idx = np.argmax(S, axis=1)
        pred = self.clases[idx].astype(float)
        p_incl = np.empty(len(X))
        for i, c in enumerate(self.clases):
            sel = idx == i
            if np.any(sel):
                t0, forma, escala = self.weibull[c]
                u = np.clip(-S[sel, i] - t0, 0, None)
                p_incl[sel] = 1.0 - weibull_min.cdf(u, forma, loc=0, scale=escala)
        return pred, p_incl


def funcion_predecir_scoresREAL(Xfeatures,Xlabels,test,metrica,openess,umbralDIST,umbralCONF,
                                scaler=None,XTRAIN=None,YTRAIN=None):
    '''
        :param Xfeatures: La lista de puntos ( en el espacio dd las features) de las que tengo certeza de estar bien etiquetadas
        :param Xlabels:  La etiquetas de las instancias anteriores
        :param Xtest:  La lista de nuesvas instancias de las que quiero obtener su etiqueta
        :param metrica: '1_NN' o 'KDE_EVT'
        :param umbralDIST: '1_NN' -> umbral de distancia; 'KDE_EVT' -> delta (umbral de P_incl)
        :param umbralCONF: '1_NN' -> umbral de confianza; 'KDE_EVT' -> h (factor de ancho de banda)
        :param scaler: StandardScaler común ajustado sobre el train (el mismo que en validación)
        :param XTRAIN, YTRAIN: muestras de entrenamiento (necesarias para la calibración EVT)
        :return prediccion, score de "conocida" para AUROC
        '''
    prediccion = []
    if metrica == 'KDE_EVT':
        modelo = ModeloKDE_EVT(h=umbralCONF).fit(Xfeatures, Xlabels, XTRAIN, YTRAIN, scaler)
        prediccion_base, p_incl = modelo.predecir(test)
        prediccion = prediccion_base.copy()
        if openess > 0:
            prediccion[p_incl < umbralDIST] = 100
        return prediccion, p_incl

    if metrica == '1_NN':
        Xfeatures, test = _escalar(Xfeatures, test, scaler)

        knnGNG = KNeighborsClassifier(n_neighbors=K_VECINOS_OSR, algorithm='ball_tree').fit(Xfeatures, Xlabels)

        distancias, indices = knnGNG.kneighbors(test)
        dist_media = distancias.mean(axis=1)

        dist_train, _ = knnGNG.kneighbors(Xfeatures)
        dist_train_media = dist_train.mean(axis=1)

        umbral_global = np.mean(dist_train_media) + 2 * np.std(dist_train_media)
        print(umbral_global,umbralDIST)

        prediccion_base = knnGNG.predict(test)
        probs = knnGNG.predict_proba(test)
        confianza = probs.max(axis=1)

        if openess>=18:
            mask_rechazo = (dist_media > umbralDIST) | (confianza < umbralCONF)#0.95
            prediccion = prediccion_base.copy()
            prediccion[mask_rechazo] = 100
        elif openess>0:
            mask_rechazo = (dist_media > umbralDIST) | (confianza < umbralCONF)#0.80
            prediccion = prediccion_base.copy()
            prediccion[mask_rechazo] = 100
        else:
            prediccion = prediccion_base.copy()

    return prediccion, confianza
def funcion_validar_scoresREALold(Xfeatures,Xlabels,testX,testY,metrica,kkc,openess):
    '''
        :param Xfeatures: La lista de puntos ( en el espacio dd las features) de las que tengo certeza de estar bien etiquetadas
        :param Xlabels:  La etiquetas de las instancias anteriores
        :param Xtest:  La lista de nuesvas instancias de las que quiero obtener su etiqueta
        :param metrica: Metrica comparativa, KNN,closest point, ...
        :param Tol_DENSITY: Parametro necesario para la mentrica de densidad. Marca la densidad por clase.
        :return prediccion: Prediccion de las etiquetas de Xtest.
        '''
    # Crea una instancia de la clase de métricas
    metrics = OSRMetrics()

    prediccion = []
    if metrica == '1_NN':
        scaler = StandardScaler()
        Xfeatures = scaler.fit_transform(Xfeatures)
        test = scaler.transform(testX)

        knnGNG = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree').fit(Xfeatures, Xlabels)

        distancias, indices = knnGNG.kneighbors(test)
        dist_media = distancias.mean(axis=1)

        dist_train, _ = knnGNG.kneighbors(Xfeatures)
        dist_train_media = dist_train.mean(axis=1)

        umbral_global = np.mean(dist_train_media) + 2 * np.std(dist_train_media)
        print(umbral_global)

        prediccion_base = knnGNG.predict(test)
        probs = knnGNG.predict_proba(test)
        confianza = probs.max(axis=1)
        # 2. BÚSQUEDA DEL UMBRAL ÓPTIMO
        # Vamos a probar diferentes combinaciones de umbral de distancia y confianza
        best_f1 = -1
        best_umbral_dist = 0
        best_umbral_conf = 0
        best_prediccion = None

        # Definimos rangos de búsqueda (Grid Search)
        # Umbral de distancia: Desde la media hasta media + 4*std
        dist_train, _ = knnGNG.kneighbors(test)
        dt_mean = dist_train.mean(axis=1).mean()
        dt_std = dist_train.mean(axis=1).std()

        candidates_dist = np.linspace(dt_mean, dt_mean + 4 * dt_std, 20)
        candidates_conf = np.linspace(0.5, 0.99, 20)  # Probar confianza entre 50% y 99%

        # Bucle de optimización
        for u_dist in candidates_dist:
            for u_conf in candidates_conf:
                # Creamos una máscara de rechazo con los candidatos actuales
                mask_rechazo = (dist_media > u_dist) | (confianza < u_conf)

                current_pred = prediccion_base.copy()
                current_pred[mask_rechazo] = 100  # Etiqueta para UUC

                # Evaluamos con Micro-F1 (o Youden si lo prefieres)
                score = metrics.micro_f1(testY, current_pred, kkc)

                if score > best_f1:
                    best_f1 = score
                    best_umbral_dist = u_dist
                    best_umbral_conf = u_conf
                    best_prediccion = current_pred

        print(f"Optimización completada. Best F1: {best_f1:.4f}")
        print(f"Umbral Distancia Óptimo: {best_umbral_dist:.4f}, Umbral Confianza Óptimo: {best_umbral_conf:.4f}")

    # Devolvemos la mejor predicción encontrada, la confianza y los umbrales para registro
    return best_prediccion,  best_umbral_dist, best_umbral_conf
    
def _validar_kde_evt(Xfeatures, Xlabels, testX, testY, kkc_eval, mask_conocida,
                     XTRAIN, YTRAIN, scaler, cobertura):
    '''Rejilla sobre h (ancho de banda) y delta (umbral de P_incl), con micro-F1 en validación.'''
    metrics = OSRMetrics()
    best_f1, best_delta, best_h, best_prediccion = -1, 0.0, H_CANDIDATOS_KDE[0], None
    for h in H_CANDIDATOS_KDE:
        modelo = ModeloKDE_EVT(h=h).fit(Xfeatures, Xlabels, XTRAIN, YTRAIN, scaler)
        prediccion_base, p_incl = modelo.predecir(testX)
        # Como mucho se rechaza un (1 - cobertura) de las conocidas
        if np.any(mask_conocida):
            delta_max = float(np.quantile(p_incl[mask_conocida], 1.0 - cobertura))
        else:
            delta_max = 1.0
        candidatos = np.unique(np.append(np.linspace(0.0, 1.0, 41), delta_max))
        for delta in candidatos[candidatos <= delta_max]:
            current_pred = prediccion_base.copy()
            current_pred[p_incl < delta] = 100
            score = metrics.micro_f1(testY, current_pred, kkc_eval)
            if score > best_f1:
                best_f1, best_delta, best_h, best_prediccion = score, float(delta), h, current_pred
    print(f"Optimización KDE_EVT completada. Best F1: {best_f1:.4f}")
    print(f"delta óptimo: {best_delta:.4f}, h óptimo: {best_h}")
    return best_prediccion, best_delta, best_h


def funcion_validar_scoresREAL(Xfeatures,Xlabels,testX,testY,metrica,kkc,openess,
                               scaler=None,XTRAIN=None,YTRAIN=None,cobertura=COBERTURA_CONOCIDAS):
    '''
        :param Xfeatures: La lista de puntos ( en el espacio dd las features) de las que tengo certeza de estar bien etiquetadas
        :param Xlabels:  La etiquetas de las instancias anteriores
        :param testX, testY: validación abierta (UUC = 100)
        :param metrica: '1_NN' o 'KDE_EVT'
        :param scaler: StandardScaler común ajustado sobre el train (el mismo que en predicción)
        :param XTRAIN, YTRAIN: muestras de entrenamiento (necesarias para 'KDE_EVT')
        :param cobertura: fracción mínima de conocidas de validación que se aceptan
        :return prediccion, umbral 1, umbral 2
                '1_NN'    -> (umbral de distancia, umbral de confianza)
                'KDE_EVT' -> (delta, h)
        '''
    # Crea una instancia de la clase de métricas
    metrics = OSRMetrics()

    # Las clases conocidas de Letter llegan como letras y las
    # predicciones como numeros. Sin esta conversion el micro-F1
    # de la rejilla vale 0 y se guarda el umbral mas pequeno.
    kkc_eval = _kkc_numericas(kkc)
    y_true_s, _, known_s = OSRMetrics._unify_types(testY, testY, kkc_eval)
    mask_conocida = np.isin(y_true_s, known_s)

    if metrica == 'KDE_EVT':
        return _validar_kde_evt(Xfeatures, Xlabels, testX, testY, kkc_eval, mask_conocida,
                                XTRAIN, YTRAIN, scaler, cobertura)

    prediccion = []
    if metrica == '1_NN':
        Xfeatures, test = _escalar(Xfeatures, testX, scaler)

        knnGNG = KNeighborsClassifier(n_neighbors=K_VECINOS_OSR, algorithm='ball_tree').fit(Xfeatures, Xlabels)

        distancias, indices = knnGNG.kneighbors(test)
        dist_media = distancias.mean(axis=1)

        dist_train, _ = knnGNG.kneighbors(Xfeatures)
        dist_train_media = dist_train.mean(axis=1)

        umbral_global = np.mean(dist_train_media) + 2 * np.std(dist_train_media)
        print(umbral_global)

        prediccion_base = knnGNG.predict(test)
        probs = knnGNG.predict_proba(test)
        confianza = probs.max(axis=1)
        # 2. BÚSQUEDA DEL UMBRAL ÓPTIMO
        # Vamos a probar diferentes combinaciones de umbral de distancia y confianza
        best_f1 = -1
        best_umbral_dist = 0
        best_umbral_conf = 0
        best_prediccion = None

        # Definimos rangos de búsqueda (Grid Search)
        # Umbral de distancia: Desde la media hasta media + 4*std
        dist_train, _ = knnGNG.kneighbors(test)
        dt_mean = dist_train.mean(axis=1).mean()
        dt_std = dist_train.mean(axis=1).std()

        candidates_dist = np.linspace(dt_mean, dt_mean + 4 * dt_std, 20)
        candidates_conf = np.linspace(0.5, 0.99, 20)  # Probar confianza entre 50% y 99%

        # Cada umbral rechaza como mucho un (1 - cobertura) de las conocidas
        # de validación (cuantiles en vez de max/min, robusto a outliers).
        if np.any(mask_conocida):
            suelo_dist = float(np.quantile(dist_media[mask_conocida], cobertura))
            techo_conf = float(np.quantile(confianza[mask_conocida], 1.0 - cobertura))
        else:
            suelo_dist = float(candidates_dist[0])
            techo_conf = float(candidates_conf[0])
        candidates_dist = np.unique(np.append(candidates_dist, suelo_dist))
        candidates_conf = np.unique(np.append(candidates_conf, techo_conf))

        # Bucle de optimización
        for u_dist in candidates_dist:
            if u_dist < suelo_dist:
                continue
            for u_conf in candidates_conf:
                if u_conf > techo_conf:
                    continue
                # Creamos una máscara de rechazo con los candidatos actuales
                mask_rechazo = (dist_media > u_dist) | (confianza < u_conf)

                current_pred = prediccion_base.copy()
                current_pred[mask_rechazo] = 100  # Etiqueta para UUC

                # Evaluamos con Micro-F1 (o Youden si lo prefieres)
                score = metrics.micro_f1(testY, current_pred, kkc_eval)

                if score > best_f1:
                    best_f1 = score
                    best_umbral_dist = u_dist
                    best_umbral_conf = u_conf
                    best_prediccion = current_pred

        if best_prediccion is None:
            best_umbral_dist = suelo_dist
            best_umbral_conf = techo_conf
            mask_rechazo = (dist_media > suelo_dist) | (confianza < techo_conf)
            best_prediccion = prediccion_base.copy()
            best_prediccion[mask_rechazo] = 100
            best_f1 = metrics.micro_f1(testY, best_prediccion, kkc_eval)

        print(f"Optimización completada. Best F1: {best_f1:.4f}")
        print(f"Umbral Distancia Óptimo: {best_umbral_dist:.4f}, Umbral Confianza Óptimo: {best_umbral_conf:.4f}")

    # Devolvemos la mejor predicción encontrada, la confianza y los umbrales para registro
    return best_prediccion,  best_umbral_dist, best_umbral_conf
    
def sorted_cluster(x, model=None):
    if model == None:
        model = KMeans()
    model = sorted_cluster_centers_(model, x)
    model = sorted_labels_(model, x)
    return model

def sorted_cluster_centers_(model, x):
    model.fit(x)
    new_centroids = []
    magnitude = []
    for center in model.cluster_centers_:
        magnitude.append(np.sqrt(center.dot(center)))
    idx_argsort = np.argsort(magnitude)
    model.cluster_centers_ = model.cluster_centers_[idx_argsort]
    return model

def sorted_labels_(sorted_model, x):
    sorted_model.labels_ = sorted_model.predict(x)
    return sorted_model
def funcion_definicion_DENSIDAD(X, Y,volumenB,num_vecinos,num_clases,dim,centroides_clase):
    densidad_clase= [[]for i in range (num_clases)]
    radio_clase= [[]for i in range (num_clases)]
    volumetria= [[]for i in range (num_clases)]

    tamaño_TOTAL=len(X)
    if tamaño_TOTAL>volumenB:
        numero_iteraciones=(int)(tamaño_TOTAL/volumenB)
        pequeño=volumenB
    else:
        numero_iteraciones = 1
        pequeño=tamaño_TOTAL

    densidad_clase_tmp = [[]for i in range (num_clases)]
    radio_clase_tmp = [[]for i in range (num_clases)]
    volumetria_tmp = [[]for i in range (num_clases)]
    for j in range(numero_iteraciones+1):
        if j<numero_iteraciones:
            Xtmp=X[(int)(j*pequeño):(int)((j+1)*pequeño)]
            Ytmp=Y[(int)(j*pequeño):(int)((j+1)*pequeño)]
        else:
            Xtmp = X[(int)(j * pequeño):]
            Ytmp = Y[(int)(j * pequeño):]
        for i in range (num_clases):
            conj=Xtmp[Ytmp==i]
            tamaño = len(conj)
            if tamaño>=num_vecinos+1:
                tmp1=[]
                tmp2=[]
                for pto in conj:
                    dist_centroide=np.linalg.norm(pto - centroides_clase[i])
                    nbrs = NearestNeighbors(n_neighbors=num_vecinos+1, algorithm='ball_tree').fit( conj)
                    distances, indices = nbrs.kneighbors(pto.reshape(1, -1))
                    distances = np.array(distances).flatten()
                    tmp1.append(distances[1:].mean())
                    tmp2.append(dist_centroide)
                densidad_clase_tmp[i].append(np.array(tmp1).mean() )
                radio_tmp=np.percentile(tmp2,85)

                perimetro=2*np.pi*radio_tmp
                area = np.pi*radio_tmp*radio_tmp
                volumen=(4/3)*np.pi*radio_tmp*radio_tmp*radio_tmp
                unidad = tamaño/volumen

                radio_clase_tmp[i].append(radio_tmp)
                volumetria_tmp[i].append(unidad)

    for i in range (num_clases):
        densidad_clase[i]=np.mean(densidad_clase_tmp[i])
        radio_clase[i]=np.mean(radio_clase_tmp[i])
        volumetria[i]=np.mean(volumetria_tmp[i])

    return densidad_clase,radio_clase,volumetria
def funcion_definicion_DENSIDADREAL(X, Y,volumenB,num_vecinos,num_clases,dim,centroides_clase):
    densidad_clase= [[]for i in range (num_clases)]
    radio_clase= [[]for i in range (num_clases)]
    volumetria= [[]for i in range (num_clases)]

    tamaño_TOTAL=len(X)
    if tamaño_TOTAL>volumenB:
        numero_iteraciones=(int)(tamaño_TOTAL/volumenB)
        pequeño=volumenB
    else:
        numero_iteraciones = 1
        pequeño=tamaño_TOTAL

    densidad_clase_tmp = [[]for i in range (num_clases)]
    radio_clase_tmp = [[]for i in range (num_clases)]
    volumetria_tmp = [[]for i in range (num_clases)]
    for j in range(numero_iteraciones+1):
        if j<numero_iteraciones:
            Xtmp=X[(int)(j*pequeño):(int)((j+1)*pequeño)]
            Ytmp=Y[(int)(j*pequeño):(int)((j+1)*pequeño)]
        else:
            Xtmp = X[(int)(j * pequeño):]
            Ytmp = Y[(int)(j * pequeño):]
        clases_train = np.unique(Y)
        cont=0
        for i in clases_train:
            conj=Xtmp[Ytmp==i]
            tamaño = len(conj)
            if tamaño>=num_vecinos+1:
                tmp1=[]
                tmp2=[]
                for pto in conj:
                    dist_centroide=np.linalg.norm(pto - centroides_clase[cont])
                    nbrs = NearestNeighbors(n_neighbors=num_vecinos+1, algorithm='ball_tree').fit( conj)
                    distances, indices = nbrs.kneighbors(pto.reshape(1, -1))
                    distances = np.array(distances).flatten()
                    tmp1.append(distances[1:].mean())
                    tmp2.append(dist_centroide)
                densidad_clase_tmp[cont].append(np.array(tmp1).mean() )
                radio_tmp=np.percentile(tmp2,85)

                perimetro=2*np.pi*radio_tmp
                area = np.pi*radio_tmp*radio_tmp
                volumen=(4/3)*np.pi*radio_tmp*radio_tmp*radio_tmp
                unidad = tamaño/volumen

                radio_clase_tmp[cont].append(radio_tmp)
                volumetria_tmp[cont].append(unidad)
            cont+=1

    for i in range (num_clases):
        densidad_clase[i]=np.mean(densidad_clase_tmp[i])
        radio_clase[i]=np.mean(radio_clase_tmp[i])
        volumetria[i]=np.mean(volumetria_tmp[i])

    return densidad_clase,radio_clase,volumetria

def draw_image(centroides,graph,nodos_sin_trasladar,nodos_trasladados,etiquetas,num_cla,num_dim,contador2,test,etiqueta_test,id_zona,ZONA,id_feature,data_dir,fichero):#id_zona,ZONA_LABELIZADA,feature_imp_num, i_name
    #plt.figure(figsize=(5.5, 6))
    plt.ion()

    for node_1, node_2 in graph.edges:
        weights = np.concatenate([node_1.weight, node_2.weight])
        line, = plt.plot(*weights.T, color='black', alpha=0.9)#.T
        plt.setp(line, linewidth=0.2, color='black')

    hh = pd.DataFrame()
    df_test = pd.DataFrame()
    pp = pd.DataFrame()

    for c in range(num_dim):
        hh[coordenadas[c]] = nodos_sin_trasladar[:, c]
        df_test[coordenadas[c]] = test[:, c]
        pp[coordenadas[c]] = nodos_trasladados[:, c]

    hh['LAB'] = etiquetas
    max_sin_X = hh['C1'].max(axis=0)
    min_sin_X = hh['C1'].min(axis=0)
    max_sin_Y = hh['C2'].max(axis=0)
    min_sin_Y = hh['C2'].min(axis=0)

    df_test['LAB'] = etiqueta_test
    max_test_X = df_test['C1'].max(axis=0)
    min_test_X = df_test['C1'].min(axis=0)
    max_test_Y = df_test['C2'].max(axis=0)
    min_test_Y = df_test['C2'].min(axis=0)

    pp['LAB'] = etiquetas
    max_tras_X = pp['C1'].max(axis=0)
    min_tras_X = pp['C1'].min(axis=0)
    max_tras_Y = pp['C2'].max(axis=0)
    min_tras_Y = pp['C2'].min(axis=0)

    for p in range(num_cla):
        plt.scatter(hh[hh['LAB'] == p]['C1'], hh[hh['LAB'] == p]['C2'], alpha=0.5,color=colors_SINGULARES[p])
        plt.scatter(df_test[df_test['LAB'] == p]['C1'], df_test[df_test['LAB'] == p]['C2'], alpha=0.5, color=colors_TEST[p])
        plt.scatter(pp[pp['LAB'] == p]['C1'], pp[pp['LAB'] == p]['C2'],  color=colors_trasladados[p])

        plt.scatter(centroides[p][0], centroides[p][1], marker='o', color='k')

    min_total_X=np.array([min_tras_X,min_sin_X,min_test_X]).min()-1
    max_total_X=np.array([max_tras_X,max_sin_X,max_test_X]).max()+1
    min_total_Y =np.array([min_tras_Y,min_sin_Y,min_test_Y]).min()-1
    max_total_Y =np.array([max_tras_Y,max_sin_Y,max_test_Y]).max()+1

    plt.axis([min_total_X,max_total_X,min_total_Y,max_total_Y])

    if id_zona!=9:
        if id_feature == 0:
            plt.axvline(x=ZONA[0], linestyle=':', color='k', alpha=0.5)
            plt.axvline(x=ZONA[1], linestyle=':', color='k', alpha=0.5)
        else:
            plt.axhline(y=ZONA[0], linestyle=':', color='k', alpha=0.5)
            plt.axhline(y=ZONA[1], linestyle=':', color='k', alpha=0.5)
    else:
        for p in range(num_cla):
            plt.axvline(x=ZONA[p][0], linestyle=':', color='r', alpha=0.5)
            plt.axvline(x=ZONA[p][1], linestyle=':', color='r', alpha=0.5)

    plt.title('Contador ' + str(contador2))

    '''
    # VIDEOS FARADAI
    # Fijar los límites del eje x
    plt.xlim(-5, 30)

    # Fijar los límites del eje y
    plt.ylim(-5, 30)
    # VIDEOS FARADAI
    '''
    path=settings_osr.path
    path=path+data_dir+'/'
    plt.savefig(path + fichero + '_' + str(contador2) + '_.jpg')
    #plt.show(block=False)
    plt.pause(0.4)
    plt.close('all')

def draw_image_divergente( graph,  num_cla, num_dim, contador2,test, etiqueta_test, data_dir,fichero):
    #plt.figure(figsize=(5.5, 6))
    for node_1, node_2 in graph.edges:
        weights = np.concatenate([node_1.weight, node_2.weight])
        line, = plt.plot(*weights.T, color='black', alpha=0.9)  # .T
        plt.setp(line, linewidth=0.2, color='black')

    df_test = pd.DataFrame()

    for c in range(num_dim):
        df_test[coordenadas[c]] = test[:, c]

    df_test['LAB'] = etiqueta_test

    for p in range(num_cla):
        plt.scatter(df_test[df_test['LAB'] == p]['C1'], df_test[df_test['LAB'] == p]['C2'], alpha=0.5,
                    color=colors_TEST[p])

    plt.title('Contador ' + str(contador2))

    path = settings_osr.path
    path = path + data_dir + '/'
    plt.savefig(path + fichero + '_' + str(contador2) + '_.jpg')
    plt.show(block=False)
    plt.pause(0.4)
    plt.close('all')

def calculo_distancia_centroides(lista1,lista2,cla):
    nodes1 = np.asarray(lista1)
    nodes2 = np.asarray(lista2)
    dst=[]
    if ((len(nodes1)!=cla)or(len(nodes2)!=cla)):
        num=np.array([len(nodes1),len(nodes2),cla]).min()
        for c in range(num):
            dst.append(distance.euclidean(nodes2[c].flatten(), nodes1[c].flatten()))
    else:
        for c in range(cla):
            dst.append(distance.euclidean(nodes2[c].flatten(), nodes1[c].flatten()))
    return dst
def calculo_TOLERANCE_centroides(centroides,num_clases,radio_clase):
    Distancias=[]
    RelacionRadios=[]
    flag_dens=0
    for c in range(num_clases-1):
        j=c+1
        tmpDIST=[]
        tmpRADIO=[]
        while j<num_clases:
            tmpDIST.append(distance.euclidean(centroides[c].flatten(), centroides[j].flatten()))
            tmpRADIO.append((c,j))
            j +=1
        Distancias.append(tmpDIST)
        RelacionRadios.append(tmpRADIO)

    if num_clases==2:
        for c in range(num_clases-1):
            tol=np.sum(radio_clase)
            if( (tol>Distancias[c])):
                flag_dens=1
    else:
        for c in range(num_clases):
            tol=(radio_clase[c])
            if c<num_clases-1:
                if( (tol>np.min(Distancias[c]))):
                    flag_dens=1
            else:
                if ((tol > np.min(Distancias[c-1]))):
                    flag_dens = 1

    return Distancias,flag_dens
def coord_to_list(data,dim,tamaño):
    if dim==2:
        coordenadas_x = data[-tamaño:, 0]
        coordenadas_y = data[-tamaño:, 1]
        coordenadas_z = np.zeros(tamaño)
        A = [np.array(coordenadas_x), np.array(coordenadas_y), np.array(coordenadas_z)]
    elif dim==3:
        coordenadas_x = data[-tamaño:, 0]
        coordenadas_y = data[-tamaño:, 1]
        coordenadas_z = data[-tamaño:, 2]
        A = [np.array(coordenadas_x), np.array(coordenadas_y), np.array(coordenadas_z)]
    else:
        A = [data[-tamaño:, j] for j in range(dim)]

    return A
def merge_puntos(lista_patron, lista_2,dim,num_clases):
    lista_ordenada=[]
    for j in range(num_clases):
        list1=lista_patron[j]
        list2 = lista_2[j]
        num_list1 = len(lista_patron[j])
        num_list2 = len(lista_2[j])
        num_menor = np.min(np.array((num_list1, num_list2)))
        tmp_list=[]
        conti = 0
        for node in list1:
            if conti < num_menor:
                my_indice=closest_node(node, list2)
                tmp_list.append(list2[my_indice])
                list2.pop(my_indice)
            conti += 1
        lista_ordenada.append(tmp_list)

    return lista_ordenada

def concateno_listas(lista1,lista2,indices_no_encontrados,proporciones,num_clases):
    lista_concatenada=[[]  for i in range(num_clases)]
    for i in range(num_clases):
        cont1=0
        cont2 = 0
        for j in range ((int)(proporciones[i])):
            if indices_no_encontrados[i]:
                if j not in indices_no_encontrados[i]:
                    if lista1[i]:
                        lista_concatenada[i].append(lista1[i][cont1])
                        cont1 += 1
                else:
                    if lista2[i]:
                        lista_concatenada[i].append(lista2[i][cont2])
                        cont2+=1
            else:
                lista_concatenada[i]=lista1[i]
                break

    return lista_concatenada
def list_to_coord(list_x,list_y,list_z,lista_multiple=[],dimen=2):
    if dimen == 2:
        v=list(zip(list_x,list_y))
    elif dimen ==3:
        v = list(zip(list_x, list_y, list_z))
    else:
        volumen_lista=len(lista_multiple[0])
        v = list(zip(lista_multiple[:,j] for j in range(volumen_lista)))

    return v
def filtrar(puntos,labels,dim, numero_clases):
    df = pd.DataFrame()
    orden=[]
    if dim==2:
        df['C1']=np.array(puntos)[:,0]
        df['C2']=np.array(puntos)[:,1]
        df['lab']=labels

        for lb in range(numero_clases):
            tmp = df[df['lab'] == lb]
            tmp_coords = [(tmp['C1'].iloc[w], tmp['Y'].iloc[w]) for w in range(tmp.shape[0])]
            orden.append(tmp_coords)
    elif dim==3:
        df['C1']=puntos[:,0]
        df['C2']=puntos[:,1]
        df['C3'] = puntos[:, 2]
        df['lab']=labels
        for lb in range(numero_clases):
            tmp = df[df['lab'] == lb]
            tmp_coords = [(tmp['C1'].iloc[w],tmp['C2'].iloc[w],tmp['C3'].iloc[w]) for w in range(tmp.shape[0])]
            orden.append(tmp_coords)
    else:
        for j in range(dim):
            tmp_name='Coord'+str(j)
            df[tmp_name] = puntos[:, j]
        df['lab'] = labels
        for lb in range(numero_clases):
            tmp = df[df['lab'] == lb]
            for w in range(tmp.shape[0]):
                tmp_coords = [(tmp['Coord'+str(j)].iloc[w]) for j in range(dim)]
            orden.append(tmp_coords)
    return orden
def filtrarGNGREALES(puntos,labels, numero_clases):
    tamaño = len(puntos)
    proporciones=[]

    puntos_ordenados = defaultdict(list)
    lista_indices = defaultdict(list)

    puntos_ordenados_lista = []
    lista_entr_indices=[]

    total_labels=np.unique(labels)
    for i in range(tamaño):
        puntos_ordenados[(int)(labels[i])].append(puntos[i])
        lista_indices[(int)(labels[i])].append(i)
    '''for label in sorted(puntos_ordenados):
        print(label, puntos_ordenados[label])'''

    labels_ordenados = sorted(puntos_ordenados.keys())
    lista_puntos = [puntos_ordenados[label] for label in labels_ordenados]
    lista_labels = [lista_indices[label] for label in labels_ordenados]

    for k in range(numero_clases):
        num=len(lista_puntos[k])
        proporciones.append(num)

    return lista_puntos,lista_labels,labels_ordenados,proporciones
def filtrarGNG(puntos,labels, numero_clases):
    tamaño = len(puntos)
    proporciones=[]

    puntos_ordenados_lista = [[] for k in range(numero_clases)]
    lista_indices = [[] for k in range(numero_clases)]

    for i, label in enumerate(labels):
        puntos_ordenados_lista[int(label)].append(puntos[i])
        lista_indices[int(label)].append(i)
    for k in range(numero_clases):
        num=len(puntos_ordenados_lista[k])
        proporciones.append(num)

    return puntos_ordenados_lista,lista_indices,proporciones
def get_permutations(arr):
    return list(itertools.permutations(arr))
def check_list(lista,num,centroides,toreraces,DIM,radio):

    for p in range(num):
        puntos = lista[p]
        num_list = 0
        for point in puntos:
            distance = ((point - centroides[p][:DIM]) ** 2 + (point - centroides[p][:DIM]) ** 2) ** 0.5
            # math.sqrt((point - centroides_transformados[p][:DIMENSION]) ** 2 + (point - centroides_transformados[p][:DIMENSION]) ** 2)
            if (distance > radio +  toreraces[p]).any():
                for f in range(DIM):
                    point_tmp = list(point)
                    if (point_tmp[f] > centroides[p][f]):
                        point_tmp[f] = point[f] - toreraces[p]
                        lista[p][num_list] = point_tmp
                    elif (point_tmp[f] < centroides[p][f]):
                        point_tmp[f] = point[f] + toreraces[p]
                        lista[p][num_list] = point_tmp
            num_list += 1
    return lista
def obtencion_ceniodes(n_c,DIM,Xtrain,ytrain,Xtest,ytest):
    t_c_TRAIN=[]
    cent_TRAIN=[]
    labelstrain=np.unique(ytrain)
    labels_test=np.unique(ytest)
    for i in labelstrain:
        res = np.array(centeroidpython(Xtrain[ytrain == i]))
        cent_TRAIN.append(res)
        if DIM < 3:
            t_c_TRAIN.append(np.array([centeroidpython(Xtrain[ytrain == i])[0],
                                 centeroidpython(Xtrain[ytrain == i])[1], 0]))
        else:
            t_c_TRAIN.append(res)
    t_c_TEST = []
    cent_TEST = []
    for i in labels_test:
        res = np.array(centeroidpython(Xtest[ytest == i]))
        cent_TEST.append(res)
        if DIM < 3:
            t_c_TEST.append(np.array([centeroidpython(Xtest[ytest == i])[0],
                                       centeroidpython(Xtest[ytest == i])[1], 0]))
        else:
            t_c_TEST.append(res)

    return t_c_TRAIN,t_c_TEST
def obtencion_centroidesC(n_c,DIM,Xtrain,ytrain,Xtest,ytest):
    flag_cambio_cl=0
    t_c_TRAIN=[]
    cent_TRAIN=[]
    for i in range(n_c):
        if len(Xtrain[ytrain == i])>0:
            res = np.array(centeroidpython(Xtrain[ytrain == i]))
            cent_TRAIN.append(res)
            if DIM < 3:
                t_c_TRAIN.append(np.array([centeroidpython(Xtrain[ytrain == i])[0],
                                     centeroidpython(Xtrain[ytrain == i])[1], 0]))
            else:
                t_c_TRAIN.append(res)
        else:
            cent_TRAIN.append(-1)
            t_c_TRAIN.append(-1)
            flag_cambio_cl = -1
    t_c_TEST = []
    cent_TEST = []
    for i in range(n_c):
        if len(Xtest[ytest == i]) > 0:
            res = np.array(centeroidpython(Xtest[ytest == i]))
            cent_TEST.append(res)
            if DIM < 3:
                t_c_TEST.append(np.array([centeroidpython(Xtest[ytest == i])[0],
                                           centeroidpython(Xtest[ytest == i])[1], 0]))
            else:
                t_c_TEST.append(res)
        else:
            t_c_TEST.append(-1)
            cent_TEST.append(-1)
            flag_cambio_cl = -1

    return flag_cambio_cl,t_c_TRAIN,t_c_TEST

def from_list_t_array(mylista,dim,num_cla):
    n=0
    for i in range (num_cla):
        if  mylista[i]:
            n+=len( mylista[i])

    miarray_POSICIONES=np.zeros((n,dim))
    miarray__etiquetasGNG = np.zeros((n))

    cont_ext=0
    for i in range(num_cla):
        cont = 0
        while cont<len( mylista[i]):
            miarray_POSICIONES[cont_ext] = mylista[i][cont]
            miarray__etiquetasGNG[cont_ext]=i
            cont+=1
            cont_ext+=1
    return miarray_POSICIONES,miarray__etiquetasGNG
def from_list_t_arrayREAL(mylista,dim,labs,num_cla):
    n=0
    tamaño=[]
    for i in range (num_cla):
        if  mylista[i]:
            n+=len( mylista[i])
        tamaño.append(len( mylista[i]))

    miarray_POSICIONES=np.zeros((n,dim))
    miarray__etiquetasGNG = np.zeros((n))

    cont_ext=0
    acum=0
    for i in labs:
        cont = 0
        while cont<tamaño[cont_ext]:
            miarray_POSICIONES[cont+acum] = mylista[cont_ext][cont]
            miarray__etiquetasGNG[cont+acum]=i
            cont+=1
        acum+=tamaño[cont_ext]
        cont_ext += 1
    return miarray_POSICIONES,miarray__etiquetasGNG
def inversa_filtroGNG(tamaño_gng,dimension,numero_clases,GNG_lista,indices0):
    sh=(tamaño_gng, dimension)
    puntos_transformados = np.zeros(tamaño_gng * dimension ).reshape( *sh)
    etiquetas_transformados = np.zeros(tamaño_gng )
    for k in range(numero_clases):
        tamaño_l=len(GNG_lista[k])
        for i in range (tamaño_l):
            puntos_transformados[indices0[k][i]]=GNG_lista[k][i]
            etiquetas_transformados[indices0[k][i]]=k
    return  puntos_transformados,etiquetas_transformados

def swapPositions(list, pos1, pos2):
    list[pos1], list[pos2] = list[pos2], list[pos1]
    return list
def funcion_ordenar(indice_origen,origen, lista_centroides_1,lista_centroides_2, lista_tras, lista_rot, lista_ang_b, lista_ang_cl):
    indice_buscado = closest_node(origen, lista_centroides_1)
    #print(indice_origen,indice_buscado)
    swapPositions(lista_centroides_1, indice_origen, indice_buscado)
    swapPositions(lista_centroides_2,indice_origen, indice_buscado)
    swapPositions(lista_tras,indice_origen, indice_buscado)
    swapPositions(lista_rot,indice_origen, indice_buscado)
    swapPositions(lista_ang_b,indice_origen, indice_buscado)
    swapPositions(lista_ang_cl,indice_origen, indice_buscado)

    return lista_centroides_1,lista_centroides_2, lista_tras, lista_rot, lista_ang_b, lista_ang_cl
def funcion_ordenarGNG_points(num_clases,origen, lista_puntos_1):
    for j in range(num_clases):
        indice_buscado = closest_node(origen[j], lista_puntos_1)
        swapPositions(lista_puntos_1, j, indice_buscado)
    return lista_puntos_1
def SeleccionByClass(instances, labels_predicted, classes,parametro_balance):
    indexesByClass = {}

    for c in range(classes):
        indexesByClass[c] = [i for i in range(len(labels_predicted)) if labels_predicted[i] == c]

    pdfsByClass = {}

    for c, indexes in indexesByClass.items():
        if len(indexes) > 0:
            pdfs = [-1] * len(instances)
            points = instances[indexes]
            pdfsByPoints=np.zeros(points.shape[0])

            kernel = KernelDensity(kernel='gaussian', bandwidth=0.4).fit(instances)
            pdfsByPoints = np.exp(kernel.score_samples(points))

            a = 0
            for i in indexes:
                if pdfsByPoints[a] != -1:
                    pdfs[i] = pdfsByPoints[a]
                a += 1
            pdfsByClass[c] = pdfs

    selectedIndexes = []

    for k in pdfsByClass:
        arrPdf = np.array(pdfsByClass[k])
        numSelected = int(np.floor(parametro_balance * len(arrPdf)))#parametro_balance[k]
        numSelected_P = int(np.floor(parametro_balance * len(indexesByClass[k])))

        if numSelected>len(indexesByClass[k]):
            numSelected=len(indexesByClass[k])

        ind = (-arrPdf).argsort()[:numSelected_P]
        selectedIndexes.append(ind)
    stackedIndexes = selectedIndexes[0]

    for i in range(1, len(selectedIndexes)):
        stackedIndexes = np.hstack([stackedIndexes, selectedIndexes[i]])

    newX = instances[stackedIndexes]
    newy = labels_predicted[stackedIndexes]

    return newX, newy
def SeleccionBy1Class(instances, centroid,parametro_balance):
    points=np.array(instances)
    distances = distance.cdist([centroid], points)[0]
    #distances = np.sqrt(np.sum((points - centroid) ** 2, axis=1))
    tam=instances.shape[0]
    sel=(int)(tam*parametro_balance)

    # Get the indices of the nearest points (for example, the 3 nearest points)
    nearest_indices = distances.argsort()[:sel]

    # Get the nearest points
    nearest_points = instances[nearest_indices]

    return nearest_points

def indices_mayores(arr, tol=1e-3):
    # (indice, valor)
    enumerados = list(enumerate(arr))
    enumerados.sort(key=lambda x: x[1], reverse=True)
    i1, v1 = enumerados[0]
    i2, v2 = enumerados[1]
    # Comparación primero vs segundo
    if abs(v1 - v2) <= tol:
        m1= max(i1, i2)
    else:
        m1=i1
    # Comparación segundo vs tercero
    i3, v3 = enumerados[2]
    if abs(v2 - v3) <= tol:
        m2= max(i2, i3)
    else:
        m2=i2
    return m1, m2
def distancia(p1, p2):
    """Calcula la distancia euclidiana entre dos puntos."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
def distancias_al_centroide_INTRA_CLUSTER(puntos,centroide):
    """Calcula la distancia mínima y máxima entre los puntos y el centroide."""
    distancias = [(p, distancia(p, centroide)) for p in puntos]
    punto_min = min(distancias, key=lambda x: x[1])
    punto_max = max(distancias, key=lambda x: x[1])

    ''' punto_min[1] #distancia_minima
     punto_min[0] #punto_mas_cercano
     punto_max[1] #distancia_maxima
     punto_max[0] # punto_mas_lejano'''

    return punto_max[1]
def distancias_MINIMAal_centroide(puntos,centroide):
    distancias = [distancia(p, centroide) for p in puntos]
    distancia_min = np.min(distancias)
    return distancia_min
def distancia_media_al_centroide_INTRA_CLUSTER(puntos,centroide):
    distancias = [distancia(p, centroide) for p in puntos]
    punto_mean = np.mean(distancias)
    punto_std=np.std(distancias)
    return punto_mean,punto_std
def punto_mas_cercano_con_label(punto, puntos, labels):
    """
    punto: tupla (x, y)
    puntos: lista de tuplas [(x1, y1), (x2, y2), ...]
    labels: lista de etiquetas paralela a puntos
    """
    px, py = punto

    min_dist = float("inf")
    indice_mas_cercano = -1

    for i, (x, y) in enumerate(puntos):
        dist = math.dist((px, py), (x, y))
        if dist < min_dist:
            min_dist = dist
            indice_mas_cercano = i

    punto_mas_cercano = puntos[indice_mas_cercano]
    label_asociada = labels[indice_mas_cercano]

    return indice_mas_cercano,  label_asociada[-1]
def punto_mas_cercano(punto, puntos):
    """
    punto: tupla (x, y)
    puntos: lista de tuplas [(x1, y1), (x2, y2), ...]
    """
    px, py = punto

    min_dist = float("inf")
    indice_mas_cercano = -1

    for i, (x, y) in enumerate(puntos):
        dist = math.dist((px, py), (x, y))
        if dist < min_dist:
            min_dist = dist
            indice_mas_cercano = i

    punto_mas_cercano = puntos[indice_mas_cercano]

    return indice_mas_cercano

def distancias_extremas(puntos):
    """Calcula la distancia mínima y máxima entre pares de puntos."""
    min_dist = float('inf')
    max_dist = 0
    par_min = par_max = (None, None)

    for p1, p2 in combinations(puntos, 2):
        d = np.linalg.norm(p1 - p2)#distancia(p1, p2)
        if d < min_dist:
            min_dist = d
            par_min = (p1, p2)
        if d > max_dist:
            max_dist = d
            par_max = (p1, p2)

    return min_dist,max_dist
def mitad_menores_que_ref(arr, REF):
    """
    Retorna True si al menos la mitad de los elementos de arr
    son menores que REF. De lo contrario, retorna False.
    """
    if not arr:  # evitar división por cero
        return False

    count_menores = sum(1 for x in arr if x < REF)
    return count_menores >= len(arr) / 2
def comprobar_arrays(A, B):
    # A tiene 3 elementos, B tiene 4
    numa=len(A)
    a1, a2, a3 = A

    # Probar todas las combinaciones posibles:
    # • elegir 2 elementos de A que deben igualar (±1) a 2 de B
    # • el 3º elemento de A debe ser suma exacta de los 2 B restantes
    for Ai, Aj, Ak in permutations(A, 3):  # Ai y Aj son los que comparan con B, Ak es el que verifica la suma
        for Bk, Bl, Bm, Bn in permutations(B, 4):  # Bk y Bl comparan con Ai y Aj, Bm y Bn suman

            cond1 = abs(Ai - Bk) <= 1 and abs(Aj - Bl) <= 1
            cond2 = Ak == Bm + Bn

            if cond1 and cond2:
                return 0  # verdadero

    return 1  # falso
def comprobar_arrays_general(A, B):
    n = len(A)
    if len(B) != n + 1:
        raise ValueError("B debe tener exactamente un elemento más que A.")

    # Elegimos n-1 elementos de A que deben coincidir (±1) con n-1 elementos de B
    for A_coinciden_indices in combinations(range(n), n - 1):

        # Los índices que NO están en A_coinciden_indices son el A_restante
        A_restante_index = list(set(range(n)) - set(A_coinciden_indices))[0]
        A_restante = A[A_restante_index]

        # Ahora elegimos n-1 elementos de B para comparar con los de A
        for B_coinciden_indices in combinations(range(len(B)), n - 1):

            # Los dos elementos restantes de B deben sumar A_restante
            B_restantes_indices = list(set(range(len(B))) - set(B_coinciden_indices))

            # Probar todas las permutaciones para emparejar correctamente
            for permA in permutations(A_coinciden_indices):
                for permB in permutations(B_coinciden_indices):

                    # Condición 1: coincidencia ±1
                    cond1 = all(abs(A[i] - B[j]) <= 1 for i, j in zip(permA, permB))

                    # Condición 2: suma exacta de los otros dos elementos de B
                    Br1, Br2 = B[B_restantes_indices[0]], B[B_restantes_indices[1]]
                    cond2 = (A_restante == Br1 + Br2)

                    if cond1 and cond2:
                        return 0  # verdadero

    return 1  # falso
def comparar_centroides(centroides_a, centroides_b,dim):
    """
    Calcula distancia mínima y máxima entre los centroides de dos grupos.
    Funciona incluso si las listas tienen distinta longitud.
    """

    A = np.array(centroides_a, dtype=float)
    B = np.array(centroides_b, dtype=float)

    A = A[:, :dim]
    B = B[:, :dim]

    # Comprobación mínima: misma dimensión por punto
    if A.shape[1] != B.shape[1]:
        raise ValueError(
            f"Los centroides no tienen la misma dimensión: {A.shape[1]} vs {B.shape[1]}"
        )

    # Todas las diferencias entre todos los pares (n x m x d)
    diff = A[:, None, :] - B[None, :, :]

    # Normas euclídeas (n x m)
    dist = np.linalg.norm(diff, axis=2)

    return dist.min(), dist.max()
def aumenta_distancia(centroides_antes, centroides_despues,dist):
    """
    Compara la separación entre centroides antes y después.
    Retorna:
      - True si las distancias aumentan
      - False si se mantienen o disminuyen
      - dist_media_antes, dist_media_despues
    """

    A = np.array(centroides_antes, dtype=float)
    B = np.array(centroides_despues, dtype=float)

    # Validar número y dimensión
    if A.shape != B.shape:
        raise ValueError("Las dos listas deben tener misma forma (mismos centroides y misma dimensión).")

    # Distancias entre centroides en el mismo índice
    dist_antes = np.linalg.norm(A - A.mean(axis=0), axis=1)
    dist_despues = np.linalg.norm(B - B.mean(axis=0), axis=1)

    dist_media_antes = dist_antes.mean()
    dist_media_despues = dist_despues.mean()
    print( dist_media_antes, dist_media_despues)

    return dist_media_antes,dist_media_despues,np.abs(dist_media_despues-dist_media_antes)>dist
def estadisticas_diferencias(lista1, lista2):
    n = min(len(lista1), len(lista2))

    # Recortar ambas listas
    l1 = lista1[:n]
    l2 = lista2[:n]

    # Diferencias elemento a elemento
    diferencias = [a - b for a, b in zip(l1, l2)]

    # Media
    media = mean(diferencias)

    # Mediana
    mediana = median(diferencias)
    maxdeif= np.max(diferencias)

    # Moda (maneja el caso de múltiples modas)
    conteo = Counter(diferencias)
    max_freq = max(conteo.values())
    modas = [k for k, v in conteo.items() if v == max_freq]

    return diferencias,maxdeif,media,mediana,modas

def comparacionhistorica(TESTACT,TRAINACT,TESTANT,TRAINANT):
    pr_te, se_te=indices_mayores(TESTACT)
    pr_tr, se_tr = indices_mayores(TRAINACT)
    difstest,max_test,mediatest,medianatest,modatest=estadisticas_diferencias(TESTACT, TESTANT)
    difstrain,max_train,mediatrain,medianatrain,modatrain=estadisticas_diferencias(TRAINACT, TRAINANT)

    if (se_te==pr_tr):
        if (np.abs(max_test-max_train)>np.abs(max_test-mediatest))&(medianatrain<=np.abs(max_train-mediatrain)):
            max_index = max(pr_te,se_te)
            if max_index!=pr_te:
                tmp=pr_te
                pr_te=se_te
                se_te=tmp

    return pr_te,se_te,pr_tr,se_tr
def comparacionhistoricaI(TESTACT,TRAINACT,TESTANT,TRAINANT,TOLHIST):
    pr_te, se_te=indices_mayores(TESTACT)
    pr_tr, se_tr = indices_mayores(TESTANT)
    difstest,max_test,mediatest,medianatest,modatest=estadisticas_diferencias(TESTACT, TESTANT)
    difstrain,max_train,mediatrain,medianatrain,modatrain=estadisticas_diferencias(TRAINACT, TRAINANT)

    if (np.abs(max_test-max_train)<TOLHIST):
        max_index = max(pr_te,se_te)
        if max_index!=pr_te:
            tmp=pr_te
            pr_te=se_te
            se_te=tmp

    return pr_te,se_te,pr_tr,se_tr
def comparacionhistoricaII(pr_te, se_te,pr_tr, se_tr,TESTACT,TRAINACT,TESTANT,TRAINANT,TOLHIST):
    difstest,max_test,mediatest,medianatest,modatest=estadisticas_diferencias(TESTACT, TESTANT)
    difstrain,max_train,mediatrain,medianatrain,modatrain=estadisticas_diferencias(TRAINACT, TRAINANT)

    if (np.abs(max_test-max_train)<TOLHIST):
        max_index = max(pr_te,se_te)
        if max_index!=pr_te:
            tmp=pr_te
            pr_te=se_te
            se_te=tmp

    return pr_te,se_te,pr_tr,se_tr

def analisis_morfologico(flag,flag_m,num1,num2,puntos,centroides_anteriores,prop_ant,tol,nodos,radio_max,radio_min,DIM,bal):
    indice=0
    tmp_DOBLE=-1
    modelt1 = KMeans(n_clusters=num1)
    modelt1.fit(puntos)
    cent_test1 = modelt1.cluster_centers_
    lab1 = modelt1.labels_
    disttesttmp1=[]
    distteststd1 = []
    proporciones1 = []

    for d in  range(num1):
        tmp,std = distancia_media_al_centroide_INTRA_CLUSTER(puntos[lab1==d],cent_test1[d])
        disttesttmp1.append(tmp)
        distteststd1.append(std)
        pr1 = len(puntos[lab1 == d])
        proporciones1.append(pr1)
    modelt2 = KMeans(n_clusters=num2)
    modelt2.fit(puntos)
    cent_test2 = modelt2.cluster_centers_
    lab2 = modelt2.labels_
    disttesttmp2 = []
    distteststd2 = []
    proporciones2 = []
    for d in range(num2):
        tmp2,std2 = distancia_media_al_centroide_INTRA_CLUSTER(puntos[lab2 == d], cent_test2[d])
        disttesttmp2.append(tmp2)
        distteststd2.append(std2)
        pr2 = len(puntos[lab2 == d])
        proporciones2.append(pr2)

    if num2!=2:
        distMIN2, distMAX2 = distancias_extremas(cent_test2)
    else:
        distMIN2,distMAX2=distancias_extremas(cent_test2)
        distMIN2=radio_min

    if num1!=2:
        distMIN1, distMAX1 = distancias_extremas(cent_test1)
    else:
        distMIN1,distMAX1=distancias_extremas(cent_test1)
        distMIN1=radio_min

    MIN1, MAX1 = comparar_centroides(centroides_anteriores, cent_test1,DIM)
    MIN2, MAX2 = comparar_centroides(centroides_anteriores, cent_test2,DIM)

    min_tot=np.min((distMIN2,distMIN1))
    min_ind=np.min((num2,num1))
    max_ind = np.max((num2, num1))

    if flag==0:
        min=np.min((num2,num1))/2
        max = np.round(np.max((num2, num1)) / 2)
        if min!=max:
            print('DUDA')
            max = np.round(np.max((num2, num1)) / 2) + 1
        comparattiva = (nodos / max) - (min-1)
        comparattiva=comparattiva/radio_min
    else:
        if num2 > num1:
            comparattiva=np.mean(proporciones1)/num1
            comparattiva = comparattiva / radio_min
        else:
            comparattiva=np.mean(proporciones2)/num2
            comparattiva = comparattiva / radio_min

    if num2>num1:
        count_menores = sum(1 for x in proporciones2 if x < comparattiva)
        print(count_menores >= len(proporciones2) / 2)
        tmp=num2
    else:
        count_menores = sum(1 for x in proporciones1 if x < comparattiva)
        print(count_menores >= (int)(len(proporciones1) / 2))
        tmp = num1

    if distMIN1<distMIN2:
        if (distMIN1 * 2 > distMIN2):
            tmp_DOBLE = num1

        if ((distMIN1 < tol / 2) ):
            indice = num2
            print('RESULTADO analisis morfologico',num2, count_menores,tmp,tmp_DOBLE,distMIN2)
        else:
            indice = num1
            print('RESULTADO analisis morfologico',num1,count_menores,tmp,tmp_DOBLE, distMIN1)
    else:
        if (distMIN2 * 2 > distMIN1):
            tmp_DOBLE = num2

        if( (distMIN2<tol/2)):
            indice=num1
            print('RESULTADO analisis morfologico',num1,count_menores,tmp,tmp_DOBLE,distMIN1)
        else:
            indice=num2
            print('RESULTADO analisis morfologico',num2,count_menores,tmp,tmp_DOBLE,distMIN2)

    if num1 < num2:
        res = comprobar_arrays_general(proporciones1, proporciones2)
        mapping,distintos=match_centroids(cent_test1, cent_test2, split_threshold=0.5)
        distancia_distintoACT=distancias_MINIMAal_centroide(cent_test1, cent_test2[distintos[0]])
        if np.array(centroides_anteriores)[:, :DIM].shape[0]< num2:
            mappANT2,distANT2=match_centroids(np.array(centroides_anteriores)[:, :DIM],cent_test2 , split_threshold=0.5)
            distancia_distintoANT=distancias_MINIMAal_centroide(np.array(centroides_anteriores)[:, :DIM], cent_test2[distANT2[0]])
        else:
            distancia_distintoANT=MAX2
    else:
        res = comprobar_arrays_general(proporciones2, proporciones1)#contador 30
        mapping, distintos = match_centroids(cent_test2, cent_test1, split_threshold=0.5)  # min_tot
        distancia_distintoACT = distancias_MINIMAal_centroide(cent_test2, cent_test1[distintos[0]])
        if np.array(centroides_anteriores)[:, :DIM].shape[0]< num1:
            mappANT1,distANT1=match_centroids(np.array(centroides_anteriores)[:, :DIM],cent_test1 , split_threshold=0.5)
            distancia_distintoANT=distancias_MINIMAal_centroide(np.array(centroides_anteriores)[:, :DIM], cent_test1[distANT1[0]])
        else:
            distancia_distintoANT = MAX1

    print(distancia_distintoANT,tol)
    if ((tmp_DOBLE!=-1)&(indice!=tmp_DOBLE)):#&(distancia_distintoANT<tol)
        indice = tmp_DOBLE
    elif ((tmp_DOBLE==-1)&(count_menores!=0)):
        indice = tmp
    elif ((tmp_DOBLE!=-1)&(tmp>indice)):
        indice = tmp

    print(num1,proporciones1,comparattiva)
    print(num2,proporciones2,comparattiva)
    print(distancia_distintoANT, MAX2)

    if flag_m==False:#True cambio 11/12
        if flag==0:
            if(bal==0):
                media = np.mean(prop_ant,axis=0)
                if num1==len(centroides_anteriores):

                    if (np.abs(np.array(proporciones1)-media)<(1.5*comparattiva)).all():
                        indice = len(centroides_anteriores)
                    else:
                        indice = num2
                elif num2==len(centroides_anteriores):
                    if (np.abs(np.array(proporciones2)-media)<(1.5*comparattiva)).all():
                        indice = len(centroides_anteriores)
                    else:
                        indice = num1
            else:
                print('SIN EXPLORAR')
                indice=len(centroides_anteriores)
        else:
            print('TRAIN INMUTABLE')
            indice = len(centroides_anteriores)
    else:
        if(( res==0 )&(np.abs(distancia_distintoACT-radio_max)<SILHOUTE_TOLERANCE)):#&(np.abs(distancia_distintoANT-MAX2)<MAX2)
            indice=min_ind
        elif ((indice==num1)&(proporciones1<comparattiva).any()):
             min_ind
        elif ((indice == num2) & (proporciones2 < comparattiva).any()):
            indice = min_ind
        elif (np.abs(distancia_distintoANT-MAX2)<SILHOUTE_TOLERANCE):
            indice = max_ind
        elif ((indice == num2) & (proporciones2 < comparattiva).any()):
            indice = max_ind
        else:
            print('MORFO ',res,distancia_distintoACT,radio_max)

    return indice
def analisis_clusters_TEST(fl_osr,n_c,in_1_te,in_2_te,puntostest,in_1_tr,in_2_tr,puntostrain,
                           disttest,disttrain,ref_radios,number_instances,tamaño_dataset,
                           flag_movimientot,centroide_ANT,prop_ANT,array_dist_ANT,num_nodos_cl,diame,dim,flag_b,
                           flag_movimiento,diffsTEST,diffsTRAIN,percentage,asig_ANT,
                           diferencias,distancia_limite,limite_clusters,LIMITE_S,
                           numero_ORIGINAL,cont,cont_camb):
    cambio_clu=-1
    flag_paso=-1
    flag_pasot=-1
    if (cont <= cont_camb + 5):
        in_1_te = in_1_tr = n_c
        cambio_clu = 0
    elif (np.abs(np.mean(disttest) - np.mean(disttrain)) < (ref_radios * 1.75)):
        if number_instances > tamaño_dataset:
            if ((fl_osr==1)&(np.abs(in_2_te-in_1_te)>1)or(np.abs(in_1_tr-in_2_tr)>1)):
                print('MG1')
                in_1_te = in_2_te
                if in_1_te > n_c:
                    cambio_clu = 2
                elif in_1_te< n_c:
                    cambio_clu = 3
                else:
                    cambio_clu = 0
            elif(fl_osr==1) &((np.abs(in_1_te-in_1_tr)>1)or(np.abs(in_2_te-in_2_tr)>1)):
                print('MG2')
                if in_1_te > in_2_te:
                    cambio_clu = 2
                else:
                    cambio_clu = 3
                in_1_te = in_2_te
            elif (fl_osr != 1) & ((np.abs(in_1_te - in_1_tr) > 1) or (np.abs(in_2_te - in_2_tr) > 1)):
                if in_1_te > in_2_te:
                    in_1_te = in_2_te
                cambio_clu = 3
            else:
                in_1_te = in_1_tr = n_c
                cambio_clu = 0
        else:
            if in_1_te == in_1_tr:
                if in_1_te == n_c:
                    cambio_clu = 0
                elif in_2_te==in_2_tr:
                    print('2ch')
                    if in_2_te == n_c:
                        cambio_clu = 0
                        in_1_te = in_2_te
                else:
                    if in_1_te < n_c:
                        indt = indtr = 0
                        if np.abs(in_1_te - in_2_te) == 1:
                            indt = analisis_morfologico(0, flag_movimientot, in_1_te,in_2_te, puntostest,
                                                                      centroide_ANT, prop_ANT,array_dist_ANT,
                                                                      num_nodos_cl, ref_radios,diame, dim, flag_b)  # ref_radios
                        if np.abs(in_1_tr - in_2_tr) == 1:
                            indtr = analisis_morfologico(1, flag_movimientot,in_1_tr,in_2_tr, puntostrain, centroide_ANT, prop_ANT,
                                                                       array_dist_ANT,num_nodos_cl, ref_radios,diame, dim, flag_b)  # ref_radios
                        if indt == in_1_te:
                            cambio_clu = 3
                        elif indtr == n_c:
                            in_1_te = indt
                            in_1_tr = indtr
                            cambio_clu = 0
                        elif indt==n_c:
                            in_1_te =in_1_tr= indt
                    else:
                        cambio_clu = 2
            else:
                if in_1_te < n_c:
                    print('cambio 3')
                    cambio_clu = 3
                else:
                    print('cambio2')
                    cambio_clu = 3
    elif ((np.mean(disttest) > np.mean(disttrain))&(in_1_te>n_c)):
        cambio_clu = 3
        print('ULTIMO CAMBIO  CRE 41')
    elif ((np.mean(disttest) < np.mean(disttrain))&(in_1_te<n_c)):
        cambio_clu = 3
        print('ULTIMO CAMBIO  CRE 42')

    elif ((flag_movimientot == True) & (np.mean(disttest) < np.mean(disttrain))):
        #if in_1_te<n_c:#cont < (7*tot/8):
        if in_1_te == n_c:  # in_1_tr
            if in_1_te > in_2_te:
                in_1_te = in_2_te
                cambio_clu = 3
            else:
                cambio_clu = 0
            print('ULTIMO CAMBIO 31 no1cdt, no2cdt')
        else:
            cambio_clu = -1
            print('ULTIMO CAMBIO 32 no1cdt, no2cdt')
        flag_paso = 1
    elif (3 * np.abs(np.mean(disttest)) < np.mean(disttrain)):
        if(in_1_te != in_1_tr):
            cambio_clu = 2
        elif ((in_1_te == in_1_tr) & (in_1_te < n_c)):
            cambio_clu = 3
    elif((flag_movimientot!=flag_movimiento)& (np.abs(np.mean(disttest)) >3 *np.mean(disttrain))):
        cambio_clu = 2
        if (in_1_te<in_2_te):
            if (in_1_te == in_1_tr):
                tmp=in_1_te
                in_1_te=in_2_te
                in_2_te=tmp

        print('CAMBIO ',cont,flag_movimientot,flag_movimiento,np.abs(np.mean(disttest)),np.mean(disttrain),cambio_clu)
    elif ((fl_osr!=1)&(flag_movimientot != flag_movimiento) & (np.abs(np.mean(disttest)) > 2 * np.mean(disttrain))):
        if in_1_te>n_c:
            cambio_clu = 2
            if (in_1_te < in_2_te):
                if (in_1_te == in_1_tr):
                    tmp = in_1_te
                    in_1_te = in_2_te
                    in_2_te = tmp
                print('CAMBIO MM', cont, flag_movimientot, flag_movimiento, np.abs(np.mean(disttest)), np.mean(disttrain),
                  cambio_clu)
        else:
            print('CAMBIO 1cdt')
    elif((fl_osr!=1)&(np.abs(np.mean(disttest)) > 2 * np.mean(disttrain))&(in_1_te < numero_ORIGINAL)&(in_1_te < in_2_te)):
        in_1_te=in_2_te
        cambio_clu = 2
        print('CAMBIO 5cvt')
    else:
        flag_paso = 0
        if (np.abs(in_1_te - in_2_te) == 1):
            print(np.mean(disttest))
            print(np.mean(disttrain))
            print(np.mean(disttest) > np.mean(disttrain))
            if ((flag_movimiento == True) & (np.mean(disttest) > 2 * np.mean(disttrain))):
                # Distinguirlo de RECURRING
                if in_1_te == n_c:
                    if (np.abs(np.abs(diffsTEST[0]) - np.abs(diffsTRAIN[0])) > TOL * 2):
                        in_1_te = in_2_te
                        cambio_clu = 3
                    else:
                        if ((in_2_te > n_c) & (
                                np.abs(np.abs(diffsTEST[0]) - np.abs(diffsTEST[1])) > TOL * 2)):
                            in_1_te = in_2_te
                            cambio_clu = 2
                            print('ULTIMO CAMBIO')
                        else:
                            indice_resultadotest = analisis_morfologico(0,flag_movimientot,in_1_te,in_2_te,puntostest,
                                                                                      centroide_ANT,prop_ANT, array_dist_ANT,num_nodos_cl,
                                                                                      ref_radios,diame,dim,flag_b)  # ref_radios
                            if indice_resultadotest == n_c:
                                in_1_te = indice_resultadotest
                                flag_paso = 1
                else:
                    in_1_te = in_2_te
                    print('ULTIMO CAMBIO M')
                flag_paso = 1
            elif ((flag_movimientot == True) & (np.mean(disttest) > np.mean(disttrain))):
                if in_1_te == n_c:
                    in_1_te = in_2_te
                cambio_clu = 3
                print('ULTIMO CAMBIO2')
                flag_paso = 1
            elif ((in_1_te != in_1_tr) & (2 * np.mean(disttest) < np.mean(disttrain))):
                cambio_clu = 3
                flag_paso = 1
            elif (((diferencias > distancia_limite).any()) & (limite_clusters > LIMITE_S) & (np.abs(diffsTEST[0]) > TOL)):

                indice_resultadotest = analisis_morfologico(0, flag_movimientot, in_1_te,in_2_te, puntostest,
                                                                          centroide_ANT,prop_ANT,array_dist_ANT,num_nodos_cl, ref_radios, diame,dim,flag_b)  # ref_radios

                in_1_te = indice_resultadotest
                if n_c != in_1_te:
                    flag = comparar_arrays(disttest, disttrain)
                    if flag == -1:
                        print(flag, 'nuevo nº clusters', in_1_te)
                        cambio_clu = 3
                    elif flag == 1:
                        if (not (np.array(disttrain)[1:] > distancia_limite).all()):
                            i = next(i for i, x in enumerate(diferencias) if x > distancia_limite)
                            print(flag, 'nuevo nº clusters', i + 2, in_2_te)
                            in_1_te = in_2_te  # distancia_limite invariante 3
                    print('flag ', flag, cambio_clu)
                    flag_paso = 1
            elif ((limite_clusters < TOL) & (np.abs(diffsTEST[0]) > TOL)):
                indice_resultadotest = analisis_morfologico(0, flag_movimientot, in_1_te,in_2_te, puntostest,
                                                                          centroide_ANT,prop_ANT,array_dist_ANT,num_nodos_cl, ref_radios, diame,dim,flag_b)  # ref_radios

                in_1_te = indice_resultadotest
                flag_paso = 1
            elif ((in_1_te < n_c) & (np.abs(diffsTEST[0]) > TOL)):
                indice_resultadotest = analisis_morfologico(0, flag_movimientot, in_1_te,in_2_te,puntostest,
                                                                          centroide_ANT,prop_ANT,array_dist_ANT,num_nodos_cl, ref_radios, diame,dim,flag_b)  # ref_radios

                in_1_te = indice_resultadotest
                flag_paso = 1
            if ((in_1_te < n_c) & (flag_paso == 0)):
                cambio_clu = 3
            elif (flag_paso == 0):
                indice_resultadotest = analisis_morfologico(0, flag_movimientot, in_1_te,in_2_te, puntostest,
                                                                          centroide_ANT,prop_ANT,array_dist_ANT,num_nodos_cl, ref_radios, diame,dim,flag_b)  # ref_radios
                if indice_resultadotest == n_c:
                    in_1_te = indice_resultadotest
                    flag_paso = 1
        elif (np.abs(in_1_tr - in_2_tr) == 1):#AQUI 17/12/25
            flag_pasot = 0
            print(np.abs(diffsTRAIN[0]))
            if (((diferencias > distancia_limite).any()) & (limite_clusters > LIMITE_S) & (
                    np.abs(diffsTRAIN[0]) > TOL)):
                print(np.mean(disttest))
                print(np.mean(disttrain))

                indice_resultadotrain = analisis_morfologico(1, flag_movimientot, in_1_tr,in_2_tr, puntostrain,
                                                                           centroide_ANT,prop_ANT,array_dist_ANT,num_nodos_cl, ref_radios, diame,dim,flag_b)  # ref_radios
                if (np.mean(disttest)>np.mean(disttrain))&(in_1_te<indice_resultadotrain):
                    in_1_te = indice_resultadotrain
                    flag_pasot = 1
            elif ( (in_1_tr < n_c) & (np.abs(diffsTRAIN[0]) > TOL)):
                indice_resultadotrain = analisis_morfologico(1, flag_movimientot,  in_1_tr,in_2_tr, puntostrain,
                                                                           centroide_ANT,prop_ANT,array_dist_ANT,num_nodos_cl, ref_radios, diame,dim,flag_b)

                in_1_te = indice_resultadotrain
                flag_pasot = 1
            elif ((limite_clusters < TOL) & (
                    np.abs(diffsTRAIN[0]) > TOL)):  # limite_clusters <tolerancia_coincidencia_punto
                indice_resultadotrain = analisis_morfologico(1, flag_movimientot, in_1_tr,in_2_tr, puntostrain,
                                                                           centroide_ANT,prop_ANT,array_dist_ANT,num_nodos_cl, ref_radios, diame,dim,flag_b)

                in_1_te = indice_resultadotrain
                flag_pasot = 1

        # ultimas comprobaciones
        if ((np.abs(in_1_te - in_2_te) > 1) & (in_1_te < n_c)):
            cambio_clu = 3
        if (in_1_tr > in_1_te) & (cambio_clu != 3):
            in_1_tr = in_1_te

        if len(asig_ANT) == 0:
            YtransTMP, cent0TMP, asignacionTMP, cambio_clustersTMP, prop_clustersTMP = asignacion_clusters_clases(
                0,
                X=puntostest, Y=[], flag=flag_b, relacion=[], n_c=numero_ORIGINAL,
                n_clu=in_1_te, dim=dim, inclu_p=percentage,
                centr=centroide_ANT)
        else:
            YtransTMP, centTMP, asignacionTMP, cambio_clustersTMP, prop_clustersTMP = asignacion_clusters_clases(
                0,
                X=puntostest, Y=[], flag=flag_b, relacion=[], n_c=numero_ORIGINAL,
                n_clu=in_1_te, dim=dim, inclu_p=percentage,
                centr=centroide_ANT, asignacion_previa=asig_ANT)

        cantidad_act = np.min(prop_clustersTMP[:, -1])
        print('cantidad minima actual ', cantidad_act)
        if ((cantidad_act == 0)):
            if in_1_te < n_c:
                in_1_te = in_1_te + 1
            else:
                in_1_te = in_1_te - 1
            if cont < cont_camb + 5:
                in_1_tr = in_1_te
            else:
                if in_1_tr != in_1_te:
                    indtr = analisis_morfologico(1, flag_movimientot,in_1_tr, in_1_te, puntostest,
                                                               centroide_ANT,prop_ANT,array_dist_ANT,num_nodos_cl, ref_radios, diame,dim,flag_b)
                    if indtr == n_c:
                        in_1_tr = indtr
                        cambio_clu = 0

    if (fl_osr==1):
        if(in_1_te==numero_ORIGINAL)or(in_1_te<n_c) :
            cambio_clu=-1
            in_1_te=n_c
    else:
        if in_1_te < numero_ORIGINAL:
            cambio_clu = -1
            in_1_te = n_c
    if (np.abs(in_1_te -n_c)>=2):
        cambio_clu = -1
        in_1_te = n_c
    if in_1_te==n_c:
        cambio_clu=-1
    print(cont,cambio_clu,in_1_te)
    return in_1_te,in_2_te,in_1_tr,in_2_tr,cambio_clu,flag_pasot,flag_paso,n_c
def dist_min_max(A, B):
    A = np.array(A, float)
    B = np.array(B, float)

    if A.shape[1] != B.shape[1]:
        raise ValueError("Los puntos no tienen misma dimensión.")

    diff = A[:, None, :] - B[None, :, :]
    dist = np.linalg.norm(diff, axis=2)

    return dist.min(), dist.max()
def separación_grupos(A_antes, B_antes, A_despues, B_despues):
    """
    Compara si dos grupos de centroides se separan entre sí entre dos instantes.
    A y B pueden tener distinta longitud.
    Retorna:
        aumenta_min  -> True si la distancia mínima aumenta
        aumenta_max  -> True si la distancia máxima aumenta
        (dmin_antes, dmin_despues, dmax_antes, dmax_despues)
    """

    dmin_antes, dmax_antes = dist_min_max(A_antes, B_antes)
    dmin_despues, dmax_despues = dist_min_max(A_despues, B_despues)

    aumenta_min = dmin_despues > dmin_antes
    aumenta_max = dmax_despues > dmax_antes

    return (aumenta_min, aumenta_max,
            dmin_antes, dmin_despues, dmax_antes, dmax_despues)
def match_centroids(prev_centroids, curr_centroids, split_threshold=0.5):
    """
    Empareja centroides entre batches y detecta splits.

    Parámetros:
        prev_centroids: np.array shape (n_prev, d)
        curr_centroids: np.array shape (n_curr, d)
        split_threshold: distancia máxima para considerar que un centroide se partió en 2

    Retorna:
        mapping: dict con { id_prev: [id_curr,...] }
    """
    n_prev = prev_centroids.shape[0]
    n_curr = curr_centroids.shape[0]
    distintos=[]

    # matriz de distancias
    D = distance.cdist(prev_centroids, curr_centroids)

    # Hungarian assignment para el emparejamiento óptimo
    row_ind, col_ind = linear_sum_assignment(D)

    mapping = {i: [] for i in range(n_prev)}

    # asignaciones iniciales
    for i, j in zip(row_ind, col_ind):
        mapping[i].append(j)

    # detectar splits (cuando hay más centroides nuevos que viejos)
    if n_curr > n_prev:
        # centroides nuevos no asignados
        assigned = set(col_ind)
        unassigned = [j for j in range(n_curr) if j not in assigned]

        for j in unassigned:
            # buscamos el centroide anterior más cercano
            i = np.argmin(D[:, j])
            if D[i, j] < split_threshold:  # si está suficientemente cerca
                mapping[i].append(j)

        distintos=unassigned

    return mapping,distintos
def aplicar_movimiento(listas_puntos,rotaciones,traslaciones,dim,numero_clases):
    transformados_tmp=[[] for i in range(numero_clases)]
    if dim==2:
        for i in range(numero_clases):
            if listas_puntos[i]:
                SINGULARES = [np.array(np.array(listas_puntos[i])[:, 0]),
                              np.array(np.array(listas_puntos[i])[:, 1]),
                              np.array(np.zeros(len(listas_puntos[i])))]

                SINGULARES_TRANSFORMADOS_STEP = (rotaciones[i] @ SINGULARES) +  traslaciones[i].reshape(3, 1)

                list_new_tmp = list_to_coord(list_x=SINGULARES_TRANSFORMADOS_STEP[0],
                                             list_y=SINGULARES_TRANSFORMADOS_STEP[1],
                                             list_z=SINGULARES_TRANSFORMADOS_STEP[2], lista_multiple=[], dimen=dim)
                transformados_tmp[i]=list_new_tmp

    elif dim==3:
        for i in range(numero_clases):
            if listas_puntos[i]:
                SINGULARES = [np.array(np.array(listas_puntos[i])[:, 0]),
                              np.array(np.array(listas_puntos[i])[:, 1]),
                              np.array(np.array(listas_puntos[i])[:, 2])]

                SINGULARES_TRANSFORMADOS_STEP = (rotaciones[i] @ SINGULARES) + traslaciones[i].reshape(3, 1)

                list_new_tmp = list_to_coord(list_x=SINGULARES_TRANSFORMADOS_STEP[0],
                                             list_y=SINGULARES_TRANSFORMADOS_STEP[1],
                                             list_z=SINGULARES_TRANSFORMADOS_STEP[2], lista_multiple=[],dimen=dim)
                transformados_tmp[i]=list_new_tmp

    else:
        for i in range(numero_clases):
            if listas_puntos[i]:
                SINGULARES = [np.array(np.array(listas_puntos[i])[:, j])for j in range(dim)]

                SINGULARES_TRANSFORMADOS_STEP = (rotaciones[i] @ SINGULARES) + traslaciones[i].reshape(dim, 1)

                # Asi los paso de vuelta
                list_new_tmp = list_to_coord(list_x=[], list_y=[], list_z=[],lista_multiple=SINGULARES_TRANSFORMADOS_STEP, dimen=dim)
                new_l=[]
                for p in range (len(list_new_tmp)):
                    new_l.append(list_new_tmp[p][0])

                transformados_tmp[i] = new_l

    return transformados_tmp
def comprobacion_movimiento_punto(punto,rotacion,traslacion,dim):
    if dim==2:
        SINGULAR = [punto[0], punto[1], [0]]
        SINGULAR_TRANSFORMADO = (rotacion @ SINGULAR) +  traslacion.reshape(3, 1)
        new_punto = list_to_coord(list_x=SINGULAR_TRANSFORMADO[0],
                                     list_y=SINGULAR_TRANSFORMADO[1],
                                     list_z=SINGULAR_TRANSFORMADO[2], lista_multiple=[],dimen=dim)
    elif dim==3:
        SINGULAR = [punto[0], punto[1], punto[2]]
        SINGULARES_TRANSFORMADO = (rotacion @ SINGULAR) + traslacion.reshape(3, 1)
        new_punto = list_to_coord(list_x=SINGULARES_TRANSFORMADO[0],
                                     list_y=SINGULARES_TRANSFORMADO[1],
                                     list_z=SINGULARES_TRANSFORMADO[2], lista_multiple=[],dimen=dim)
    else:
        SINGULAR_tmp=[]
        SINGULAR_TRANS_tmp = []
        for j in range(dim):
            SINGULAR_tmp.append(punto[j])

        SINGULAR =np.array(SINGULAR_tmp)
        SINGULARES_TRANSFORMADO = (rotacion @ SINGULAR) + traslacion.reshape(dim, 1)
        new_punto = list_to_coord(list_x=[],list_y=[],list_z=[], lista_multiple=SINGULARES_TRANSFORMADO,dimen=dim)

    return new_punto
def comprobacion_centroides( centroides_anteriores, centroides_transformados, centroides_sin_transformar,
                       rotaciones, traslaciones, angulos_base, angulos_clase, DIMENSION, numero_clases):
    centroides_comprobacion = []
    if DIMENSION == 2:
        for i in range(numero_clases-1):
            #print(i)
            CENTROIDE_INIT = centroides_anteriores[i]
            funcion_ordenar(0, CENTROIDE_INIT, centroides_sin_transformar[i:], centroides_transformados[i:],
                            traslaciones[i:], rotaciones[i:], angulos_base[i:], angulos_clase[i:])
            if (traslaciones[i] == 0.0).all():
                centroide_rotado = (rotaciones[i] @ CENTROIDE_INIT)
            else:
                CENTROIDE_TR_STEP = (rotaciones[i] @ CENTROIDE_INIT) + traslaciones[i].flatten()
                centroide_rotado = (CENTROIDE_TR_STEP[0], CENTROIDE_TR_STEP[1], CENTROIDE_TR_STEP[2])
            centroides_comprobacion.append(centroide_rotado)
    elif DIMENSION == 3:
        for i in range(numero_clases-1):
            CENTROIDE_INIT = centroides_anteriores[i]
            funcion_ordenar(0, CENTROIDE_INIT, centroides_sin_transformar[i:], centroides_transformados[i:],
                            traslaciones[i:], rotaciones[i:], angulos_base[i:], angulos_clase[i:])
            if (traslaciones[i] == 0.0).all():
                centroide_rotado = (rotaciones[i] @ CENTROIDE_INIT)
            else:
                CENTROIDE_TR_STEP = (rotaciones[i] @ CENTROIDE_INIT) + traslaciones[i].flatten()
                centroide_rotado = (CENTROIDE_TR_STEP[0], CENTROIDE_TR_STEP[1], CENTROIDE_TR_STEP[2])
            centroides_comprobacion.append(centroide_rotado)
    else:
        for i in range(numero_clases-1):
            CENTROIDE_INIT = centroides_anteriores[i]
            funcion_ordenar(0, CENTROIDE_INIT, centroides_sin_transformar[i:], centroides_transformados[i:],
                            traslaciones[i:], rotaciones[i:], angulos_base[i:], angulos_clase[i:])

            if (traslaciones[i] == 0.0).all():
                centroide_rotado = (rotaciones[i] @ CENTROIDE_INIT)
            else:
                CENTROIDE_TR_STEP = (rotaciones[i] @ CENTROIDE_INIT) + traslaciones[i].flatten()
                centroide_rotado = CENTROIDE_TR_STEP
            centroides_comprobacion.append(centroide_rotado)
    return  centroides_comprobacion

def comprobacion_angulos(id_clase,Rot,Tras,dim):
    if dim<=3:
        Rot = np.asarray(Rot, dtype=float)
        r00 = float(np.clip(Rot[0][0], -1.0, 1.0))
        angulo_base = (math.acos(r00))
        espejo_en_plano = np.linalg.det(Rot[:2, :2]) < -1e-8
        matriz_positiva = np.array([[math.cos(angulo_base), -math.sin(angulo_base), 0],
                                    [math.sin(angulo_base), math.cos(angulo_base), 0], [0, 0, 1]])
        matriz_negativa = np.array([[math.cos(angulo_base), math.sin(angulo_base), 0],
                                    [-math.sin(angulo_base), math.cos(angulo_base), 0], [0, 0, 1]])

        matriz_IMPROPIA = np.array([[math.cos(angulo_base), -math.sin(angulo_base), 0],
                                    [math.sin(angulo_base), math.cos(angulo_base), 0], [0, 0, -1]])

        angulo_clase = angulo_base + angulo_extra
        if espejo_en_plano:
            angulo_base = 0.0
            angulo_clase = 0.0
            matriz_reconstruida = np.array([[1., 0, 0],
                                            [0, 1., 0],
                                            [0, 0, 1.]])
        elif angulo_base > tolerancia_angulo:
            if (Rot - matriz_positiva < tolerancia_ROTACION).all():
                matriz_reconstruida = np.array(
                    [[math.cos(angulo_clase), -math.sin(angulo_clase), 0],
                     [math.sin(angulo_clase), math.cos(angulo_clase), 0], [0, 0, 1]])
            elif (Rot - matriz_negativa < tolerancia_ROTACION).all():
                matriz_reconstruida = np.array([[math.cos(angulo_clase), math.sin(angulo_clase), 0],
                                                [-math.sin(angulo_clase), math.cos(angulo_clase),
                                                 0], [0, 0, 1]])

            elif (Rot - matriz_IMPROPIA < tolerancia_ROTACION).all():
                matriz_reconstruida = np.array([[math.cos(angulo_clase), -math.sin(angulo_clase), 0],
                                                [math.sin(angulo_clase), math.cos(angulo_clase),
                                                 0], [0, 0, 1]])
            else:
                matriz_reconstruida=Rot
        else:
            matriz_reconstruida = np.array([[1, 0, 0],
                                            [0, 1, 0],
                                            [0, 0, 1]])
        for j in range(dim):
            if np.abs(Tras[j]) < tolerancia_traslacion_comparacion:
                Tras[j] = 0.
            else:
                if Tras[j]<0:
                    Tras[j]=Tras[j]-tolerancia_traslacion_comparacion
                else:
                    Tras[j] = Tras[j] + tolerancia_traslacion_comparacion
    else:
        Rot = np.asarray(Rot, dtype=float)
        r00 = float(np.clip(Rot[0][0], -1.0, 1.0))
        angulo_base = (math.acos(r00))
        angulo_clase = angulo_base + angulo_extra
        matriz_reconstruida=Rot

    return angulo_base,angulo_clase,matriz_reconstruida
def Wilcoxon_flag_NEW(samples1,samples2):
    flag=-1
    n1=check_normality(samples1)
    n2=check_normality(samples2)
    v=check_variance_homogeneity(samples1, samples2)

    if ((n1 >=0.05)&(n2 >=0.05)&(v >=0.05)):
        statistic, pvalue = ttest_ind(samples1, samples2)  # [:,0]
        #print('T Test Independiente,   statistics and p-value = {:<.3f}: {:<.3f}'.format(statistic.item(), pvalue.item()))
    else:
        statistic, pvalue = mannwhitneyu(samples1,samples2)#[:,0]
        #print('Man Whitney U,   statistics and p-value = {:<.3f}: {:<.3f}'.format(statistic.item(), pvalue.item()))
    #print(pvalue)
    if (pvalue<0.05):
        flag=0
        #print("Reject null hypothesis. >> The data is not the same.")
    else:
        flag=1
        #print("Fail to reject null hypothesis >> The data is the same.")
    return flag,pvalue
def check_normality(data):
    test_stat_normality, p_value_normality=shapiro(data)
    #print("p value:%.4f" % p_value_normality)
    if p_value_normality <0.05:
        #print("Reject null hypothesis >> The data is not normally distributed.")
        tmp=str("The data is not normally distributed.")
    else:
        #print("Fail to reject null hypothesis >> The data is normally distributed.")
        tmp=str("The data is normally distributed.")
    return p_value_normality
def check_variance_homogeneity(group1, group2):
    test_stat_var, p_value_var= levene(group1,group2)
    #print("p value:%.4f" % p_value_var)
    if p_value_var <0.05:
        #print("Reject null hypothesis >> The variances of the samples are different.")
        tmp=str("The variances of the samples are different.")
    else:
        #print("Fail to reject null hypothesis >> The variances of the samples are same.")
        tmp=str("The variances of the samples are same.")
    return(p_value_var)

def calcular_diametro(puntos):
    max_distancia = 0
    for i in range(len(puntos)):
        for j in range(i+1, len(puntos)):
            distancia = np.linalg.norm(puntos[i]-puntos[j])
            if distancia > max_distancia:
                max_distancia = distancia
    return max_distancia
def funcion_predecir_divergente(Xfeatures,Xlabels,test):
    prediccion = []
    knn = KNeighborsClassifier(n_neighbors=7, weights='distance', algorithm='kd_tree').fit(Xfeatures, Xlabels)
    prediccion = knn.predict(test).astype(int)
    return prediccion

def obtain_alpha_IOU_2025(cont,n_c,lista_points1,lista_points2,etiquetas_puntos2,percent,data_dir):
    contornos1 = [[] for p in range(n_c)]
    contornos2 = [[] for p in range(n_c)]
    coef_exp = 1.1+(1 - percent)

    name_iou=str('')

    valores_IOU = []
    intersect_c = [[] for p in range(n_c)]
    union_c = [[] for p in range(n_c)]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    for c in range(n_c):
        points1 = lista_points1[c]
        shaper1 = Alpha_Shaper(points1)
        lpha_opt1, alpha_shape1 = shaper1.optimize()#ALMACENADOS
        points2 = lista_points2[np.array(etiquetas_puntos2) == c]
        shaper2 = Alpha_Shaper(points2)
        lpha_opt2, alpha_shape2 = shaper2.optimize()#TEST
        minimo = min(lpha_opt1, (lpha_opt2))

        poly1 = shaper1.get_shape(alpha=minimo/2) # GNG  minimo/2 Para que no queden picos
        #print(lpha_opt1,lpha_opt2,minimo/2)
        contornos1[c].append(poly1)
        ax.scatter(*zip(*points1),alpha=0.35)
        plot_alpha_shape(ax, poly1, colors[c])
        round(lpha_opt1, 2)

        #print(lpha_opt1,lpha_opt2,minimo)
        poly2 = shaper2.get_shape(alpha=minimo) #TEST
        contornos2[c].append(poly2)
        ax.scatter(*zip(*points2),alpha=0.35)
        plot_alpha_shape(ax, poly2, colors[c+n_c])
        round(lpha_opt2, 2)

        intersect = contornos1[c][0].intersection(contornos2[c][0]).area
        intersect = intersect * coef_exp
        intersect_c[c].append(intersect)
        union = contornos1[c][0].union(contornos2[c][0]).area
        union = union * percent
        union_c[c].append(union)
        iou = intersect / union
        valores_IOU.append(iou)

        name_iou += 'Intersect class ' + str(c) + ' : ' + str(round(intersect, 2)) + ' , '
        name_iou += 'Union class' + str(c) + ' : ' + str(round(union, 2)) + ' ,  '
        name_iou += 'IOU class' + str(c) + ' : ' + str(round(iou, 2)) + '    \n'

    print(name_iou)

    fig.suptitle(name_iou, fontsize=10)
    plt.tight_layout()

    path = settings_osr.path
    path = path + data_dir + '/'

    fig.savefig(path + 'img' + '_' + str(cont) + '_.jpg')
    plt.show(block=False)
    plt.close('all')
    print(valores_IOU)
    return valores_IOU


def _clase_mayoritaria(votos):
    if len(votos) == 0:
        return 0
    valores, cuentas = np.unique(votos, return_counts=True)
    return valores[np.argmax(cuentas)]


def _empaquetar(etiquetas, centros, clases, cambio):
    n = len(centros)
    asignacion = np.zeros((n, 2))
    proporcion = np.zeros((n, 2))
    for i in range(n):
        asignacion[i] = (i, clases[i])
        proporcion[i] = (i, int(np.sum(etiquetas == i)))
    return etiquetas, centros, asignacion, cambio, proporcion


def Asignacion2(flag_sup, X, Y, flag, relacion, n_c, n_clu, dim, inclu_p, centr=[], asignacion_previa=[], X_anterior=None, etiquetas_anteriores=None):
    """
    No sustituye a asignacion_clusters_clases.

    No vuelve a cortar el lote con KMeans. Los grupos anteriores se mueven
    con sus puntos. Un grupo nuevo solo nace si hay una masa de puntos fuera
    de todos los radios, y hereda la clase del grupo del que salen. Si el
    corte pedido parte una nube ya explicada, se conserva el numero anterior.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y)
    n_clu = int(n_clu)
    if len(centr) == 0:
        viejos = np.empty((0, X.shape[1]))
    else:
        viejos = np.asarray(centr, dtype=float)[:, :dim]
    if len(asignacion_previa) == 0:
        clases = np.empty((0,))
    else:
        clases = np.asarray(asignacion_previa, dtype=float)[:, 1]

    hay_etiquetas = Y.size > 0 and (flag_sup == 1 or len(clases) == 0)
    if hay_etiquetas or len(viejos) == 0:
        modelo = KMeans(
            n_clusters=n_clu, n_init=10, max_iter=MAX_ITERATIONS,
            random_state=0, tol=CONVERGENCE_TOLERANCE,
        ).fit(X)
        etiquetas = modelo.labels_
        if hay_etiquetas:
            clases_out = [_clase_mayoritaria(Y[etiquetas == i]) for i in range(n_clu)]
        else:
            clases_out = [i % max(int(n_c), 1) for i in range(n_clu)]
        return _empaquetar(etiquetas, modelo.cluster_centers_, clases_out, 0)

    n_viejos = min(len(viejos), len(clases))
    viejos = viejos[:n_viejos].copy()
    clases = clases[:n_viejos].copy()
    distancias = cdist(X, viejos)
    cercano = distancias.argmin(axis=1)
    dist_cercana = distancias[np.arange(len(X)), cercano]
    radios = np.ones(n_viejos)
    for i in range(n_viejos):
        propias = dist_cercana[cercano == i]
        if len(propias) >= 5:
            radios[i] = max(float(np.quantile(propias, 0.85)), 1e-6)
    fuera = dist_cercana > 3.2 * radios[cercano]
    etiquetas = cercano.copy()
    centros = viejos.copy()
    for i in range(n_viejos):
        dentro = (cercano == i) & ~fuera
        if int(dentro.sum()) >= 5:
            centros[i] = X[dentro].mean(axis=0)
    masa_min = max(8, int(0.02 * len(X)))
    cambio = 0
    if int(fuera.sum()) >= masa_min and n_clu > n_viejos:
        centro_nuevo = X[fuera].mean(axis=0)
        distancia_centro = np.linalg.norm(centros - centro_nuevo, axis=1)
        if np.all(distancia_centro > 3.2 * radios):
            clases = np.append(clases, _clase_mayoritaria(clases[cercano[fuera]]))
            centros = np.vstack([centros, centro_nuevo])
            etiquetas[fuera] = n_viejos
            cambio = 1
    return _empaquetar(etiquetas, centros, clases, cambio)








