
# Variante sin supervision de la cardinalidad.
# RastreadorClases devuelve +1 si abre una clase y -1 si un centroide
# seguido se vacia y los demas siguen cubriendo el lote. Esas dos
# salidas se aplican aqui a numero_clusters. Las etiquetas del lote
# solo se imprimen si esa decision no coincide con el numero real.
# El fichero AIGAS_OSR_2026.py no se altera.
#
# The part of the code corresponding to DensityPeakCluster is takeb from: https://github.com/lanbing510/DensityPeakCluster/tree/master?tab=MIT-1-ov-file
# Licencia MIT
# Copyright (c) 2014 Jason

import warnings
warnings.filterwarnings('ignore')
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sklearn.metrics import silhouette_score
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from transformation import get_rotation_translation,get_rotation_translation_MULTI_D
import GNG_UTILS_OSR_2025 as GNG_Funciones
import osr_utils as osr
from osr_utils import OSRMetrics
from estimar_cardinalidad_sin_etiquetas import ClassTracker

import settings_open

settings_open.init()
path = settings_open.path

try:
    import pydevd
except ImportError:
    pass
else:
    dbg = pydevd.get_global_debugger()
    if dbg is not None:
        dbg.init_matplotlib_support()

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
tamaño_dataset=44000#220000

NUM_THREADS = 8

from configure_release import stream_config


def _label_column(df):
    for name in ("CLASS", "RESULTADO"):
        if name in df.columns:
            return name
    return df.columns[-1]


fileNames, dim_TOTAL, num_features = stream_config()
n_streams = len(fileNames)
metrica_param = ["1_NN"] * n_streams
numero_reps = [1] * n_streams
dist_save_mod = [1.5] * n_streams
including_percentage = [0.7] * n_streams
contour_percentage = [0.6] * n_streams
tolerance_iou = [0.2] * n_streams
ref_num = [3] * n_streams

if __name__ == '__main__':
    for i_name, file in enumerate(fileNames):
        print('\n',file[:-4])
        df = GNG_Funciones.leer(file)
        label_col = _label_column(df)
        x = df.iloc[:, 0:num_features[i_name]].values
        y = df[label_col].values
        number_instances = len(x)
        DIMENSION = num_features[i_name]

        if number_instances > tamaño_dataset:
            NUM_inicio_etiquetados = int((number_instances * 5) / dim_TOTAL[i_name])
        else:
            NUM_inicio_etiquetados = 1000

        numero_clases_ORIGINAL = len(np.unique(y[:NUM_inicio_etiquetados]))
        flag_EZ=-1

        numero_clases=numero_clases_ORIGINAL
        print('\nclases', numero_clases_ORIGINAL)
        #Cambiamos de nombre las etiquetas simplemente para que empiece por cero
        df[label_col].replace(numero_clases_ORIGINAL, 0, inplace=True)

        num_clases_total=len(np.unique(y))
        if num_clases_total!=numero_clases_ORIGINAL:
            df[label_col].replace(num_clases_total, numero_clases_ORIGINAL, inplace=True)

        path=settings_open.path
        path=path+file[:-4]
        dir_datset=file[:-4]

        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
            print("The new directory is created!")
        else:
            print("The directory WAS previously created!")

        numero_repeticiones = numero_reps[i_name]
        array_resultados_boxplot = []

        for _ in range(numero_repeticiones):
            # Crea una instancia de la clase de métricas
            metrics = OSRMetrics()
            nuevo_nombre = file[:-4] + str(_)

            dim_max_comparacion = dim_TOTAL[i_name]

            vol_batch = int(NUM_inicio_etiquetados / 5)
            dim_max = int((number_instances - NUM_inicio_etiquetados) / vol_batch)

            if NUM_inicio_etiquetados < vol_batch:
                NUM_inicio_etiquetados = vol_batch
            elif NUM_inicio_etiquetados % vol_batch != 0:
                tmp_num = int(NUM_inicio_etiquetados / vol_batch)
                NUM_inicio_etiquetados = tmp_num * vol_batch

            NUM_instancias_poriteracion = NUM_inicio_etiquetados
            DELTA_Conocimiento_tiempo = 0
            if vol_batch > NUM_instancias_poriteracion:
                NUM_instancias_poriteracion = vol_batch
            elif NUM_instancias_poriteracion > NUM_inicio_etiquetados:
                NUM_instancias_poriteracion = NUM_inicio_etiquetados
            elif NUM_instancias_poriteracion % vol_batch == 0:
                tmp_num = int(NUM_instancias_poriteracion / vol_batch)
                NUM_instancias_poriteracion = tmp_num * vol_batch
                DELTA_Conocimiento_tiempo = tmp_num
            elif NUM_instancias_poriteracion % vol_batch != 0:
                tmp_num = int(NUM_instancias_poriteracion / vol_batch)
                NUM_instancias_poriteracion = tmp_num * vol_batch

            if NUM_inicio_etiquetados > NUM_instancias_poriteracion:
                DELTA_Conocimiento_tiempo = int((NUM_inicio_etiquetados - NUM_instancias_poriteracion) / vol_batch)
            elif DELTA_Conocimiento_tiempo == 0:  # else:#
                DELTA_Conocimiento_tiempo = 1

            step = 1
            if number_instances>tamaño_dataset:
                numero_nodos_GNG_clase = 100
            else:
                numero_nodos_GNG_clase = int(vol_batch / numero_clases_ORIGINAL)
            print('NUMERO NODOS GNG por clase', numero_nodos_GNG_clase)
            numero_nodos_GNG = numero_nodos_GNG_clase * numero_clases_ORIGINAL

            flag_balanced = 0
            relacion_inicial = np.full(numero_clases_ORIGINAL, -1.1)

            for i in range(numero_clases_ORIGINAL):
                relacion_inicial[i] = x[:1000][y[:1000] == i].shape[0]

            if round(relacion_inicial.max() / relacion_inicial.min()) > 1:
                flag_balanced = 1
                print('NO BALANCED DATASET.')
            else:
                print('BALANCED DATASET')

            DELTA_BATCH = int(NUM_instancias_poriteracion / vol_batch)

            X_DEF = x
            YDEF = y

            print('INICIO ETIQUETADOS', NUM_inicio_etiquetados)
            print('Delta conocimiento ', DELTA_Conocimiento_tiempo)
            print('VOLUMEN BATCH ', vol_batch)
            print('Instancias por iteracion ', NUM_instancias_poriteracion)
            print('DIM MAX TOTAL', dim_max)

            if vol_batch>300:
                comp=round(vol_batch/200)
            else:
                comp = 1
            factor_proporcionalidad_batch_clusters=3*comp
            cantidad_proporcionalidad_batch_clusters=60*comp

            flag_cambio=0
            contador_cambio=0
            numero_labeled = 0
            numero_unlabeled = 0
            numero_vecinos = 0

            lista_etiquetas_test = []
            lista_proporciones= []
            lista_asignaciones = []
            lista_numero_clusters = []
            rastreador = None

            ZONA_LABELIZADA = []
            lista_centroides = []
            lista_distancias_maximas = []
            diferencias=[]
            MITAD_DISTANCIA = 0

            POSICIONESGNG = np.full((dim_max,numero_nodos_GNG,DIMENSION), -1.1)
            labelsGNG = np.full((dim_max,numero_nodos_GNG), -1.1)

            F1 = np.full(dim_max, -1.1)
            micro_F1_osr = np.full(dim_max, -1.1)
            Youden = np.full(dim_max, -1.1)
            AUROC = np.full(dim_max, -1.1)
            array_resultados = np.full(dim_max, -1.1)
            array_resultados_acumulados = np.full(dim_max, -1.1)
            array_error = np.full(dim_max, -1.1)
            array_error_acumulado = np.full(dim_max, -1.1)
            array_flag_densidad = np.full(dim_max, -1.1)
            array_distancia_centroides = np.full(dim_max, -1.1)

            array_pvalor = []

            if dim_max != dim_max_comparacion:
                F1_COMP = np.full(dim_max_comparacion, 0.0)
                micro_F1_osr_COMP = np.full(dim_max_comparacion, -1.1)
                Youden_COMP = np.full(dim_max_comparacion, -1.1)
                AUROC_COMP = np.full(dim_max_comparacion, -1.1)
                array_resultados_COMP = np.full(dim_max_comparacion, 0.0)
                array_resultados_acumulados_COMP = np.full(dim_max_comparacion, 0.0)
                array_error_COMP = np.full(dim_max_comparacion, 0.0)
                array_error_acumulado_COMP = np.full(dim_max_comparacion, 0.0)

            prueba=x[:NUM_instancias_poriteracion]

            limit = int((prueba.shape[0] // 4) ** 0.5)
            scores = []
            scores.append(0.0)
            scores.append(0.1)
            for k in range(2, limit + 1):
                model = KMeans(n_clusters=k)
                model.fit(prueba)
                pred = model.predict(prueba)
                score = silhouette_score(prueba, pred)
                scores.append(score)
            numero_clusters = np.argmax(scores)
            second_element=sorted(scores)[-2]
            index_second=np.where (np.array(scores)==second_element)[0][0]
            diferencia_clusteres=scores[numero_clusters]-second_element

            numero_clusters_INICIALES=numero_clusters

            lista_gng_guardados = []
            lista_guardados = []
            lista_seleccion_guardados = []
            lista_etiquetas_gng_guardados = []
            lista_etiquetas_seleccion_guardados = []
            lista_centroides_modelos_guardados = []
            lista_numero_clusters_modelos_guardados = []
            lista_proporciones_modelos_guardados = []
            lista_asignaciones_modelos_guardados = []
            lista_proporciones_clusters_guardados =[]

            distacia_guardar_modelos = dist_save_mod[i_name]
            TOLERANCE_IOU = tolerance_iou[i_name]
            list_iou = []

            for contador in range(dim_max):
                print(contador)
                flag_situacion=0
                SALIDA = 0
                if ((NUM_instancias_poriteracion + vol_batch * (contador + 1)) <= number_instances):
                    if (contador<= DELTA_Conocimiento_tiempo):
                        flag_situacion=1
                        if ((NUM_instancias_poriteracion+(vol_batch * contador)) <= NUM_inicio_etiquetados):
                            numero_unlabeled = 0
                        elif  numero_unlabeled<NUM_instancias_poriteracion:
                            numero_unlabeled = (NUM_instancias_poriteracion + (vol_batch * contador))-NUM_inicio_etiquetados
                        else:
                            numero_unlabeled = NUM_instancias_poriteracion
                    else:
                        flag_situacion = 2
                        if flag_cambio!=1:
                            if ((NUM_instancias_poriteracion + (vol_batch * contador)) <= NUM_inicio_etiquetados):
                                numero_unlabeled = 0
                                numero_labeled=NUM_inicio_etiquetados
                            elif numero_unlabeled < NUM_instancias_poriteracion:
                                numero_unlabeled =  ((NUM_instancias_poriteracion + (vol_batch * contador)-NUM_inicio_etiquetados))
                                numero_labeled=NUM_instancias_poriteracion-numero_unlabeled
                            else:
                                numero_unlabeled = NUM_instancias_poriteracion
                                numero_labeled=0
                        else:
                            if( (contador<=(contador_cambio+DELTA_BATCH))&(vol_batch!=NUM_instancias_poriteracion)):
                                numero_labeled = 0
                                diferencia=contador-contador_cambio
                                numero_unlabeled=diferencia*vol_batch
                            else:
                                numero_unlabeled = NUM_instancias_poriteracion
                                numero_labeled = 0
                                flag_cambio=0
                else:
                    SALIDA = 1

                if (SALIDA != 1):
                    if flag_situacion==1:
                        Xsupervised, ysupervised, Xtest, ytest = GNG_Funciones.definir_sets_training_test(X_DEF, YDEF, vol_batch, contador,NUM_instancias_poriteracion)
                        kkc_fit = np.unique(ysupervised)  # known_labels
                        # clases desconocidas
                        all_labels = np.unique(ytest)
                        unknown_labels = [c for c in all_labels if c not in kkc_fit]
                        # |C_TR|
                        c_tr = len(kkc_fit)
                        # |C_TE|
                        c_te =  len(all_labels)
                        O = 100 * osr.openness(c_tr, c_te)
                        if contador==0:
                            gng_slayer = GNG_Funciones.create_gng(n_i=DIMENSION, step_gng=0.2, n_start=2,max=numero_nodos_GNG, max_age=10, min_distance=tolerancia_coincidencia_punto)

                        if numero_clusters!=numero_clases_ORIGINAL:#numero_clusters>numero_clases_ORIGINAL:
                            flag_EZ = 1
                            numero_clases=numero_clusters
                            Xtrans = Xsupervised
                            if contador==0:
                                Ytrans,cent0,asignacion,cambio_clusters,prop_clusters=GNG_Funciones.asignacion_clusters_clases(1,X=Xsupervised,Y=ysupervised,flag=flag_balanced,relacion=relacion_inicial,n_c=numero_clases_ORIGINAL,n_clu=numero_clusters,dim=DIMENSION,inclu_p=including_percentage[i_name])
                                lista_asignaciones.append(asignacion)
                            else:
                                Ytrans, cent0, asignacion,cambio_clusters,prop_clusters = GNG_Funciones.asignacion_clusters_clases(1 ,X=Xsupervised,  Y=ysupervised, flag=flag_balanced,relacion=relacion_inicial,n_c=numero_clases_ORIGINAL,  n_clu=numero_clusters, dim=DIMENSION,inclu_p=including_percentage[i_name],centr=lista_centroides[-1],asignacion_previa=asignacion)
                                if numero_clases_ORIGINAL==numero_clusters:
                                    cantidad=np.min(lista_proporciones[-1],axis=0)[1]
                                    sum_ant=np.sum(lista_proporciones[-1], axis=0)[1]
                                    sum_act=(np.sum(prop_clusters, axis=0))[1]
                                else:
                                    cantidad = np.min(lista_proporciones[-1], axis=0)
                                    sum_ant = np.sum(lista_proporciones[-1], axis=0)
                                    sum_act = np.sum(prop_clusters, axis=0)[1]
                                print('ACTUALES ', sum_act, prop_clusters)
                                print('ANTERIORES ', sum_ant, lista_proporciones[-1])

                        else:
                            flag_EZ = 0
                            Xtrans = Xsupervised
                            Ytrans = ysupervised
                        Xsup, ysup = GNG_Funciones.SeleccionByClass(Xtrans, Ytrans, numero_clases, including_percentage[i_name])

                        mytrain = Xsup
                        my_y_train = ysup

                        if contador > 0:
                            proporciones_ACTUALES = lista_proporciones[contador - 1]
                            posiciones_anterioores_list = []
                            ini = 0
                            for d in range(numero_clases):
                                posiciones_anterioores_list.append(POSICIONESGNG[contador - step][ini:ini + proporciones_ACTUALES[d]].tolist())
                                ini += proporciones_ACTUALES[d]

                            GNG_ANT, etiquetas_ant = GNG_Funciones.from_list_t_array(posiciones_anterioores_list, DIMENSION, numero_clases)
                            tmp_node, gng_slayer = GNG_Funciones.retrain_gng(1, mytrain,gng_slayer, GNG_ANT, DIMENSION, 0.2, 20,numero_nodos_GNG, dim_max, tolerancia_coincidencia_punto)
                        else:
                            tmp_node, gng_slayer = GNG_Funciones.train_gng(mytrain, gng_slayer)

                        print('Contador: ', contador)

                        limit = int((mytrain.shape[0] // 4) ** 0.5)
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
                        index_second = np.where(np.array(scores) == second_element)[0][0]
                        diferencia_clusteresNEW = scores[np.argmax(scores)] - second_element
                        
                        if diferencia_clusteresNEW>diferencia_clusteres:
                            numero_clusters = np.argmax(scores)
                        else:
                            if contador>0:
                                numero_clusters=lista_numero_clusters[contador-1]
                            else:
                                numero_clusters = np.argmax(scores)

                        lista_numero_clusters.append(numero_clusters)

                        if contador == 0:
                            if numero_clusters != numero_clases_ORIGINAL:# numero_clusters > numero_clases_ORIGINAL:
                                kmeans = KMeans(n_clusters=numero_clases, max_iter=MAX_ITERATIONS, random_state=0,init=cent0, tol=CONVERGENCE_TOLERANCE)
                            else:
                                kmeans = KMeans(n_clusters=numero_clases, max_iter=MAX_ITERATIONS, random_state=0,init=INITIALIZE_CLUSTERS, tol=CONVERGENCE_TOLERANCE)
                            kmeans.fit(mytrain, my_y_train)

                            if numero_clases == 2:
                                DISTANCIA_CENTROIDES = np.linalg.norm(kmeans.cluster_centers_)
                                MITAD_DISTANCIA_11 = DISTANCIA_CENTROIDES / 11

                                TOLERANCE_GNG=[MITAD_DISTANCIA_11 for j in range(numero_clases)]
                            else:
                                d=[]
                                for j in range (numero_clases):
                                    tmp=np.linalg.norm(kmeans.cluster_centers_[j] - kmeans.cluster_centers_, axis=1)
                                    d.append(tmp)
                                d=np.array(d).reshape((numero_clases,numero_clases))
                                d=np.where(np.array(d) != 0., d, 100)
                                TOLERANCE_GNG=d.min(axis=1)
                                TOLERANCE_GNG=TOLERANCE_GNG/ 11

                            imp, feature_imp_num, feature_NO_imp_num, identificador_zona = GNG_Funciones.feature_importance(mytrain, my_y_train)
                            if identificador_zona == 9:
                                ZONA_LABELIZADA = np.zeros((numero_clases, DIMENSION))
                            else:
                                ZONA_LABELIZADA = np.zeros((DIMENSION))

                            t_c = []
                            t_max=[]
                            cent=[]
                            for i in range(numero_clases):
                                res=np.array(GNG_Funciones.centeroidpython(mytrain[my_y_train == i]))
                                cent.append(res)
                                if DIMENSION<3:
                                   t_c.append(np.array([GNG_Funciones.centeroidpython(mytrain[my_y_train==i])[0],GNG_Funciones.centeroidpython(mytrain[my_y_train==i])[1],0]))
                                else:
                                   t_c.append(res)
                            for i in range(numero_clases):
                                tmp_max=GNG_Funciones.distancias_al_centroide_INTRA_CLUSTER(mytrain[my_y_train == i], t_c[i])
                                t_max.append(tmp_max)
                            lista_centroides.append(t_c)
                            lista_distancias_maximas.append(t_max)

                            Tolerance_DENSITY,TOLERANCE_radio_clase,TOLERANCE_poblacion=GNG_Funciones.funcion_definicion_DENSIDAD(mytrain, my_y_train,vol_batch,numero_vecinos_ENTROPIA,numero_clases,DIMENSION,cent)
                            diametro=np.min(TOLERANCE_radio_clase)
                            ref_radios = np.max(TOLERANCE_radio_clase)

                            labelsknn_tmp = GNG_Funciones.funcion_comparativa(mytrain, my_y_train,np.array(tmp_node).reshape(np.array(tmp_node).shape[0],DIMENSION), metrica='KNN5_GNG')
                            nodos_temporales = np.array(tmp_node).reshape(np.array(tmp_node).shape[0], DIMENSION)

                            lista_ordenadostmp, indices_ordenados_tmp,prop =  GNG_Funciones.filtrarGNG(nodos_temporales, labelsknn_tmp, numero_clases)
                            array_posicionestmp,Array_etiquetastmp=GNG_Funciones.from_list_t_array(lista_ordenadostmp, DIMENSION, numero_clases)

                            proporciones = np.array(prop)
                            lista_proporciones.append(proporciones)
                            proporciones_ORIGINALES = prop
                            proporciones_ORIGINALES_CLASES=[0 for j in range(numero_clases_ORIGINAL)]

                            if numero_clusters != numero_clases_ORIGINAL:# numero_clusters > numero_clases_ORIGINAL:
                                for i in range(numero_clusters):
                                    proporciones_ORIGINALES_CLASES[(int)(asignacion[i][1])]+=proporciones_ORIGINALES[i]
                            else:
                                proporciones_ORIGINALES_CLASES=proporciones_ORIGINALES
                                cambio_clusters=0

                            ratio_distribucion_INICIAL_etiquetas = np.max(proporciones_ORIGINALES) / np.min(proporciones_ORIGINALES)
                            POSICIONESGNG[contador] = GNG_Funciones.flatten(lista_ordenadostmp)
                            labelsGNG[contador]=Array_etiquetastmp#.tolist()

                            for j in range(numero_clases-1):
                                dim_radio=0
                                if (np.abs(t_c[j]-t_c[j+1])<np.sum(TOLERANCE_radio_clase)).any():
                                    dim_radio=np.linalg.norm(np.array(t_c[j])- np.array(t_c[j+1]), axis=0)
                                    flag_dens=1
                                    break
                                else:
                                    dim_radio=np.linalg.norm(np.array(t_c[j])- np.array(t_c[j+1]), axis=0)

                            array_flag_densidad[contador] = flag_dens
                            array_distancia_centroides[contador]=dim_radio

                            labels_DEFINITIVA = GNG_Funciones.funcion_predecir( Xfeatures=array_posicionestmp, Xlabels=Array_etiquetastmp,XTRAIN=mytrain, YTRAIN=my_y_train, test=Xtest, flag=flag_dens,
                                Tol_DENSITY=Tolerance_DENSITY, metrica=metrica_param[i_name], n_c=numero_clases, dim=DIMENSION, centroides=t_c, TOL_radio_clase=TOLERANCE_radio_clase, ZONA=[], ident_Z=identificador_zona, feat_imp=feature_imp_num)

                            cent_tr, cent_tst = GNG_Funciones.obtencion_ceniodes(numero_clases, DIMENSION, mytrain,my_y_train, Xtest,np.array(labels_DEFINITIVA))
                            distancia_train_test_INICIAL = GNG_Funciones.calculo_distancia_centroides(cent_tr, cent_tst,numero_clases)
                            distancia_train_test_OROGINARIA=distancia_train_test_INICIAL
                            distancia_limite = (np.max(distancia_train_test_OROGINARIA)) * ref_num[i_name]#(np.max(distancia_train_test_OROGINARIA)) * 3
                            print(contador, '  distancia_train_test_OROGINARIA  ', distancia_train_test_OROGINARIA)
                            print(contador, '  distancia LIMITE  ', distancia_limite)
                            if numero_clusters>2:
                                LIMITE_S=LIMITE_S/numero_clases

                            if flag_EZ!=0:#numero_clusters != numero_clases_ORIGINAL:# numero_clusters > numero_clases_ORIGINAL:
                                for j in range(Xtest.shape[0]):
                                    labels_DEFINITIVA[j] = int(asignacion[labels_DEFINITIVA[j]][1])

                            flag_wilcoxon_list = []
                            tmp=np.full((numero_clases_ORIGINAL, DIMENSION), -1.1)
                            for p in range(numero_clases_ORIGINAL):
                                for f in range(DIMENSION):
                                    flag_wilcoxon, pvalorc = GNG_Funciones.Wilcoxon_flag_NEW(mytrain[np.array(my_y_train) == p][:, f],Xtest[np.array(labels_DEFINITIVA) == p][:, f])
                                    tmp[p][f]=pvalorc
                                    flag_wilcoxon_list.append(flag_wilcoxon)
                            array_pvalor.append(tmp)

                            F1[contador] = (f1_score(ytest, labels_DEFINITIVA, average='macro'))
                            lista_etiquetas_test.append(ytest)  # SUPERVISADO
                            comparativa = ytest

                            cont_conflicto = 0
                            indices_conflicto = []
                            copia_label_test = labels_DEFINITIVA.copy()
                            puntos_zona_conflictiva = []
                            label_zona_conflictiva = []
                            for cont in range(len(labels_DEFINITIVA)):
                                if labels_DEFINITIVA[cont] != comparativa[cont]:
                                    indices_conflicto.append(cont)
                                    cont_conflicto += 1
                                    puntos_zona_conflictiva.append(Xtest[cont])
                                    label_zona_conflictiva.append(ytest[cont])
                                    labels_DEFINITIVA[cont] = comparativa[cont]

                            if identificador_zona != 9:
                                if len(puntos_zona_conflictiva) > 0:
                                    tmp = np.array(puntos_zona_conflictiva)[:, feature_imp_num]
                                    ZONA_LABELIZADA = [tmp.min() - TOLERANCE_ZONA, tmp.max() + TOLERANCE_ZONA]
                                else:
                                    tmp_init = 0
                                    tmp_media = 0
                                    tmp_fin = 0
                                    for j in range(numero_clases):
                                        tmp_init += lista_centroides[0][j][feature_imp_num]
                                        tmp_fin += lista_centroides[0][j][feature_imp_num]
                                        tmp_media += np.array(lista_centroides[0][j][feature_imp_num] )

                                    referencia_init = tmp_init / 2
                                    referencia_actual = tmp_fin / 2
                                    referencia_media = tmp_media / 2

                                    ZONA_LABELIZADA = [referencia_media - TOLERANCE_ZONA,referencia_media + TOLERANCE_ZONA]
                            else:
                                if len(puntos_zona_conflictiva) > 0:
                                    ARRAY_CONFLICTO = np.array(puntos_zona_conflictiva)
                                    ZONA_CONFLICTO_punto_medio = ARRAY_CONFLICTO.mean(axis=0)

                                    minimos_conflicto = np.array(ARRAY_CONFLICTO).min(axis=0)
                                    maximos_conflicto = np.array(ARRAY_CONFLICTO).max(axis=0)

                                    for fea in range(DIMENSION):
                                        ZONA_LABELIZADA[fea] = [minimos_conflicto[fea] - TOLERANCE_ZONA, maximos_conflicto[fea] + TOLERANCE_ZONA]

                            # ANALISIS MOVIMIENTO
                            maximos = []
                            for i in range(numero_clases):
                                cercano = lista_centroides[0][i]
                                group_tmp = []
                                group_tmp.append(cercano)
                                for j in range(1, len(lista_centroides)):
                                    indice = GNG_Funciones.closest_node(cercano,lista_centroides[j])
                                    group_tmp.append(lista_centroides[j][indice])
                                conj = np.array(group_tmp)[:, feature_NO_imp_num]
                                ext = np.zeros(1)
                                np.concatenate((conj[1:], ext))
                                maxi = np.max(conj - np.concatenate((conj[1:], ext)))
                                maximos.append(maxi)

                            puntos = np.array(puntos_zona_conflictiva)
                            arr_lb = np.array(label_zona_conflictiva)

                            resVERDADERA = np.abs(comparativa - copia_label_test)
                            errorVERDADERA = len(np.where(resVERDADERA != 0)[0]) / len(resVERDADERA)
                            totalVERDADERA = 100 * (1 - errorVERDADERA)
                            array_resultados[contador] = (1 - errorVERDADERA)
                            array_error[contador] = errorVERDADERA
                            acumulado = GNG_Funciones.calcularAcumuladonew(contador, array_resultados, array_resultados[0])
                            array_resultados_acumulados[contador] = acumulado
                            array_error_acumulado[contador] = GNG_Funciones.calcular_ERROR_Acumuladonew(contador, array_error,array_error[0])

                            print('F1 ', F1[contador])
                            print(contador, '  RESULTADOS:  ', totalVERDADERA)

                            # F1 y Youden usan las ETIQUETAS (preds_test)
                            F = metrics.micro_f1(comparativa, labels_DEFINITIVA, kkc_fit)
                            youd = metrics.youden_index_evl(comparativa, labels_DEFINITIVA, kkc_fit)
                            Youden[contador] = youd
                            micro_F1_osr[contador] = F
                            print(f"microF1_OSR: {F}, youden: {youd}")

                            inicio=0
                            seleccion_TMP = []
                            todos_TMP=[]
                            for g in range(numero_clases):
                                lista_tmp=GNG_Funciones.flatten(POSICIONESGNG[contador][inicio:inicio+proporciones_ORIGINALES[g]])
                                tam_tmp=np.array(POSICIONESGNG[contador][inicio:inicio + proporciones_ORIGINALES[g]]).shape[0]
                                array_tmp =np.array(lista_tmp).reshape(tam_tmp, DIMENSION)
                                out=  GNG_Funciones.SeleccionBy1Class(array_tmp,np.array(t_c)[g,:DIMENSION], including_percentage[i_name])
                                seleccion_TMP.append(out)
                                todos_TMP.append(array_tmp)
                                inicio += proporciones_ORIGINALES[g]
                                if g==0:
                                    lista_gng_guardados.append(POSICIONESGNG[contador].tolist())
                                    lista_etiquetas_gng_guardados.append(labelsGNG[contador].tolist())
                                    lista_centroides_modelos_guardados.append(np.array(t_c))
                                    lista_numero_clusters_modelos_guardados.append(numero_clases)
                                    lista_proporciones_modelos_guardados.append(proporciones_ORIGINALES)
                                    if numero_clases!=numero_clases_ORIGINAL:
                                        lista_asignaciones_modelos_guardados.append(asignacion)
                                        lista_proporciones_clusters_guardados.append(lista_proporciones[-1])
                                elif g == numero_clases - 1:
                                    lista_guardados.append(todos_TMP)
                                    lista_seleccion_guardados.append(seleccion_TMP)
                        else:
                            nodos_temporales = np.array(tmp_node).reshape(np.array(tmp_node).shape[0], DIMENSION)
                            labels_tmp = GNG_Funciones.funcion_comparativa(mytrain, my_y_train, nodos_temporales, metrica='KNN5_GNG')

                            lista_ordenadostmp, indices_ordenados_tmp, prop = GNG_Funciones.filtrarGNG(nodos_temporales,labels_tmp , numero_clases)
                            array_posicionestmp, Array_etiquetastmp = GNG_Funciones.from_list_t_array(lista_ordenadostmp, DIMENSION,numero_clases)

                            if cambio_clusters!=0:
                                proporciones=proporciones_ACTUALES
                                lista_proporciones.append(proporciones_ACTUALES)
                            else:
                                lista_proporciones.append(proporciones)

                            array_posiciones_anteriores, Array_etiquetas_anteriores = GNG_Funciones.from_list_t_array( posiciones_anterioores_list, DIMENSION, numero_clases)
                            nodos_t_anterior_para_comparar,nodos_t_actual_ordenados,nodos_ant_no_comp = GNG_Funciones.merge_hungarian(numero_clases,posiciones_anterioores_list, lista_ordenadostmp)

                            tmp_ang_base = []
                            tmp_ang_clase = []
                            tmp_clas_rot = []
                            tmp_clas_tras = []
                            centroides_sin_transformar = []
                            centroides_transformados = []

                            if cambio_clusters==0:
                                for i in range(numero_clases):
                                    A = GNG_Funciones.coord_to_list(np.array(nodos_t_anterior_para_comparar[i]), DIMENSION, len(nodos_t_anterior_para_comparar[i]))
                                    B = GNG_Funciones.coord_to_list(np.array(nodos_t_actual_ordenados[i]), DIMENSION,len(nodos_t_actual_ordenados[i]))

                                    # Recover R and t
                                    if DIMENSION<=3:
                                        ret_R, ret_t, centroide_sin_t, centroide_t = get_rotation_translation(np.array(A), np.array(B))
                                    else:
                                        ret_R, ret_t, centroide_sin_t, centroide_t = get_rotation_translation_MULTI_D(DIMENSION,np.array(A), np.array(B))

                                    compro=GNG_Funciones.comprobacion_movimiento_punto(centroide_sin_t, ret_R, ret_t, DIMENSION)
                                    angulo_base,angulo_clase,matriz_reconstruida=GNG_Funciones.comprobacion_angulos(i,ret_R,ret_t,DIMENSION)

                                    tmp_ang_base.append(angulo_base)
                                    tmp_ang_clase.append(angulo_clase)
                                    tmp_clas_rot.append(matriz_reconstruida)
                                    tmp_clas_tras.append(ret_t.ravel())
                                    centroides_sin_transformar.append(centroide_sin_t.ravel())
                                    centroides_transformados.append(centroide_t.ravel())
                                t_max=[]
                                for i in range(numero_clases):
                                    tmp_max = GNG_Funciones.distancias_al_centroide_INTRA_CLUSTER(
                                        nodos_t_actual_ordenados[i], centroides_transformados[i])
                                    t_max.append(tmp_max)
                                lista_distancias_maximas.append(t_max)
                                centroides_provisioneles = GNG_Funciones.comprobacion_centroides(lista_centroides[contador - step],  centroides_transformados, centroides_sin_transformar, tmp_clas_rot, tmp_clas_tras, tmp_ang_base, tmp_ang_clase,  DIMENSION, numero_clases)
                                lista_centroides.append(centroides_transformados)

                                distancia = GNG_Funciones.calculo_distancia_centroides(lista_centroides[contador - step],centroides_transformados, numero_clases)
                                TOLERANCE_distancia_entre_centroides, flag_densidad = GNG_Funciones.calculo_TOLERANCE_centroides( centroides_transformados, numero_clases, TOLERANCE_radio_clase)
                                array_flag_densidad[contador] = flag_densidad
                                if numero_clases > 2:
                                    array_distancia_centroides[contador] = np.min(TOLERANCE_distancia_entre_centroides)[  0]
                                else:
                                    array_distancia_centroides[contador] = np.min(TOLERANCE_distancia_entre_centroides)

                                for i in range(numero_clases):
                                    if (np.abs(TOLERANCE_GNG[i]) < np.abs(tmp_clas_tras[i])).any():
                                        TOLERANCE_GNG[i] = np.max(np.abs(tmp_clas_tras[i]))
                                    elif (np.abs(TOLERANCE_GNG[i]) < np.abs(distancia[i])).any():
                                        TOLERANCE_GNG[i] = np.max(np.abs(distancia[i]))

                                Lista_nodos_anterior_GNG_transformados = GNG_Funciones.aplicar_movimiento( posiciones_anterioores_list, tmp_clas_rot, tmp_clas_tras, DIMENSION, numero_clases)
                                array_nodos_anteriores_transformados, etiquetas_anteriores_trans = GNG_Funciones.from_list_t_array( Lista_nodos_anterior_GNG_transformados, DIMENSION, numero_clases)

                                #movimiento
                                nodos_anteriores_ordde_hung, nodos_t_act_ord, nodos_ant_no_comp,indicesno,nodos_act_no_ut = GNG_Funciones.merge_hungarianGNG(numero_clases, Lista_nodos_anterior_GNG_transformados, lista_ordenadostmp)
                                nodos_para_AÑADIR_listahung = GNG_Funciones.aplicar_movimiento( nodos_ant_no_comp, tmp_clas_rot, tmp_clas_tras, DIMENSION,numero_clases)

                                GNG_ACTUAL_lista = GNG_Funciones.concateno_listas(nodos_t_act_ord,nodos_para_AÑADIR_listahung, indicesno,proporciones_ORIGINALES,numero_clases)
                                array_nodos_actuales_ordenados, etiquetas_actuales_orden = GNG_Funciones.from_list_t_array(  GNG_ACTUAL_lista, DIMENSION, numero_clases)

                                lista_nodos_ACTUALES_transformados_para_test = GNG_Funciones.aplicar_movimiento( GNG_ACTUAL_lista, tmp_clas_rot, tmp_clas_tras, DIMENSION, numero_clases)
                                array_nodos_actuales_transformados, etiquetas_actuales_trans = GNG_Funciones.from_list_t_array(lista_nodos_ACTUALES_transformados_para_test, DIMENSION, numero_clases)

                                GNG_listaDEF=GNG_Funciones.check_list(GNG_ACTUAL_lista, numero_clusters, centroides_transformados, TOLERANCE_radio_clase,DIMENSION,dim_radio)

                                POSICIONESGNG[contador] = GNG_Funciones.flatten(GNG_listaDEF)
                                labelsGNG[contador] = etiquetas_actuales_orden#.tolist()
                            else:
                                t_c = []
                                for i in range(numero_clases):
                                    if DIMENSION < 3:
                                        t_c.append(np.array([cent0[i][0], cent0[i][1], 0]))
                                    else:
                                        t_c.append(np.array(cent0[i]))

                                centroides_transformados = t_c
                                lista_centroides.append(centroides_transformados)

                                GNG_ACTUAL_lista = array_posicionestmp
                                array_nodos_actuales_transformados = array_posicionestmp
                                etiquetas_actuales_trans = Array_etiquetastmp

                                GNG_listaDEF=GNG_Funciones.check_list(GNG_ACTUAL_lista, numero_clusters, centroides_transformados, TOLERANCE_radio_clase,DIMENSION,dim_radio)

                                t_max=[]
                                for i in range(numero_clases):
                                    tmp_max = GNG_Funciones.distancias_al_centroide_INTRA_CLUSTER(
                                        GNG_listaDEF[i], centroides_transformados[i])
                                    t_max.append(tmp_max)
                                lista_distancias_maximas.append(t_max)

                                POSICIONESGNG[contador] = GNG_listaDEF
                                labelsGNG[contador] = etiquetas_actuales_trans

                            XSELECTED, ySELECTED = GNG_Funciones.SeleccionByClass(array_nodos_actuales_transformados, etiquetas_actuales_trans, numero_clases,including_percentage[i_name])  # param

                            labels_DEFINITIVA = GNG_Funciones.funcion_predecir(Xfeatures=XSELECTED, Xlabels=ySELECTED,XTRAIN=mytrain, YTRAIN=my_y_train, test=Xtest, flag=flag_densidad,Tol_DENSITY=Tolerance_DENSITY, metrica=metrica_param[i_name],
                                                                                 n_c=numero_clases, dim=DIMENSION, centroides=centroides_transformados,TOL_radio_clase=TOLERANCE_radio_clase, ZONA=ZONA_LABELIZADA,ident_Z=identificador_zona,feat_imp=feature_imp_num)

                            cent_tr, cent_tst = GNG_Funciones.obtencion_ceniodes(numero_clases, DIMENSION, mytrain, my_y_train, Xtest,np.array(labels_DEFINITIVA))
                            distancia_train_test = GNG_Funciones.calculo_distancia_centroides(cent_tr, cent_tst,numero_clases)
                            flag_movimiento=GNG_Funciones.aumenta_distancia(cent_tr,cent_tst,diametro)

                            if flag_EZ!=0:#numero_clusters != numero_clases_ORIGINAL:# numero_clusters > numero_clases_ORIGINAL:
                                etiquetas_distribucion = np.full([len(labels_DEFINITIVA)], -1)
                                for j in range(Xtest.shape[0]):
                                    etiquetas_distribucion[j] = int(asignacion[labels_DEFINITIVA[j]][1])
                                    distribucion_etiquetas = [len(Xtest[np.array(etiquetas_distribucion) == c]) for c in range(numero_clases_ORIGINAL)]
                                    ratio_distribucion_etiquetas = np.max(distribucion_etiquetas) / np.min(distribucion_etiquetas)
                            else:
                                distribucion_etiquetas = [len(Xtest[np.array(labels_DEFINITIVA) == c]) for c in range(numero_clases)]
                                ratio_distribucion_etiquetas = np.max(distribucion_etiquetas) / np.min(distribucion_etiquetas)

                            if( (ratio_distribucion_etiquetas>ratio_distribucion_INICIAL_etiquetas*3)):
                                kmeanstest = KMeans(n_clusters=numero_clases_ORIGINAL, random_state=0)
                                kmeanstest.fit(Xtest)
                                centroidestest = kmeanstest.cluster_centers_

                                _asociacion_clusters = kmeanstest.fit(Xtest).labels_
                                new_lab = kmeanstest.predict(centroidestest)
                                knn = KNeighborsClassifier(n_neighbors=1).fit(centroidestest, new_lab)
                                prediccion_knn = knn.predict(Xtest).astype(int)

                                for l in range(numero_clases):
                                    if l>0:
                                        centroidestest = np.flip(centroidestest)
                                        kmeanstest = KMeans(n_clusters=numero_clases_ORIGINAL, init=centroidestest,random_state=0)
                                        kmeanstest.fit(Xtest)

                                        _asociacion_clusters = kmeanstest.fit(Xtest).labels_
                                        new_lab = kmeanstest.predict(centroidestest)
                                        knn = KNeighborsClassifier(n_neighbors=1).fit(centroidestest, new_lab)
                                        prediccion_knn = knn.predict(Xtest).astype(int)

                                    size_save = len(lista_seleccion_guardados)
                                    for tam in range(size_save):
                                        distancia_comparativa = GNG_Funciones.calculo_distancia_centroides(np.array(lista_centroides_modelos_guardados)[tam][:,:DIMENSION].tolist(),centroidestest.tolist(), numero_clases)
                                        #print(tam,distancia_comparativa)
                                        if (np.array(distancia_comparativa) < (diametro+TOLERANCE_IOU)).all():
                                            if np.array(lista_numero_clusters_modelos_guardados)[tam] == numero_clases:
                                                valores2 = GNG_Funciones.obtain_alpha_IOU_2025(contador, np.array(lista_numero_clusters_modelos_guardados)[tam],lista_guardados[tam], XtestP, prediccion_knn,including_percentage[i_name],dir_datset)
                                                list_iou.append([contador, np.array(valores2)])
                                                if (np.array(valores2) >=contour_percentage[i_name] ).all():
                                                    print('CAMBIO')
                                                    flag_cambio = 1
                                                    contador_cambio = contador
                                                    array_nodos_actuales_transformados = GNG_Funciones.flatten(lista_gng_guardados[tam])
                                                    array_nodos_actuales_transformados = np.array(array_nodos_actuales_transformados).reshape(numero_nodos_GNG,DIMENSION)
                                                    etiquetas_actuales_trans = lista_etiquetas_gng_guardados[tam]
                                                    POSICIONESGNG[contador] = array_nodos_actuales_transformados.tolist()
                                                    labelsGNG[contador] = np.array(etiquetas_actuales_trans)
                                                    labels_DEFINITIVA_P = prediccion_knn
                                                    lista_centroides.pop()
                                                    if DIMENSION == 2:
                                                        tmp = np.zeros((numero_clases, 3))
                                                        for q in range(numero_clases):
                                                            tmp[q] = np.array(
                                                                [centroidestest[q][0], centroidestest[q][1], 0])
                                                        t_max=[]
                                                        for i in range(numero_clases):
                                                            tmp_max = GNG_Funciones.distancias_al_centroide_INTRA_CLUSTER(
                                                                lista_gng_guardados[tam][i], tmp[i])
                                                            t_max.append(tmp_max)
                                                        lista_distancias_maximas.append(t_max)
                                                        lista_centroides.append(tmp)
                                                    else:
                                                        t_max=[]
                                                        for i in range(numero_clases):
                                                            tmp_max = GNG_Funciones.distancias_al_centroide_INTRA_CLUSTER(
                                                                lista_gng_guardados[tam][i], centroidestest[i])
                                                            t_max.append(tmp_max)
                                                        lista_distancias_maximas.append(t_max)
                                                        lista_centroides.append(centroidestest)
                                            break

                            if flag_EZ!=0:#numero_clusters != numero_clases_ORIGINAL:# numero_clusters > numero_clases_ORIGINAL:
                                for j in range(Xtest.shape[0]):
                                    labels_DEFINITIVA[j] = int(asignacion[labels_DEFINITIVA[j]][1])

                            flag_wilcoxon_list = []
                            tmp = np.full((numero_clases_ORIGINAL, DIMENSION), -1.1)
                            if len(set(labels_DEFINITIVA)) > 1:
                                for p in range(numero_clases_ORIGINAL):
                                    for f in range(DIMENSION):
                                        flag_wilcoxon, pvalorc = GNG_Funciones.Wilcoxon_flag_NEW(
                                            mytrain[np.array(my_y_train) == p][:, f],
                                            Xtest[np.array(labels_DEFINITIVA) == p][:, f])
                                        tmp[p][f] = pvalorc
                                        flag_wilcoxon_list.append(flag_wilcoxon)
                                array_pvalor.append(tmp)
                            else:
                                for p in range(numero_clases_ORIGINAL):
                                    for f in range(DIMENSION):
                                        tmp[p][f] = 1
                                        flag_wilcoxon = 1
                                        flag_wilcoxon_list.append(flag_wilcoxon)
                                array_pvalor.append(tmp)

                            F1[contador]=(f1_score(ytest, labels_DEFINITIVA, average='macro'))

                            comparativa = ytest

                            cont_conflicto=0
                            indices_conflicto=[]
                            copia_label_DEF_test=labels_DEFINITIVA.copy()
                            puntos_zona_conflictiva = []
                            for cont in range(len(labels_DEFINITIVA)):
                                if labels_DEFINITIVA[cont] != comparativa[cont]:
                                    indices_conflicto.append(cont)
                                    cont_conflicto+=1
                                    puntos_zona_conflictiva.append(Xtest[cont])
                                    label_zona_conflictiva.append(labels_DEFINITIVA[cont])
                                    labels_DEFINITIVA[cont] = comparativa[cont]

                            if identificador_zona != 9:
                                if len(puntos_zona_conflictiva) > 0:
                                    tmp = np.array(puntos_zona_conflictiva)[:, feature_imp_num]
                                    ZONA_LABELIZADA = [tmp.min() - TOLERANCE_ZONA, tmp.max() + TOLERANCE_ZONA]
                                else:
                                    referencia_init = np.array(lista_centroides)[0,:,feature_imp_num].mean()
                                    referencia_actual = np.array(lista_centroides)[contador,:,feature_imp_num].mean()
                                    referencia_media = np.array(lista_centroides)[:,:,feature_imp_num].mean()

                                    ZONA_LABELIZADA = [referencia_media - TOLERANCE_ZONA,
                                                       referencia_media + TOLERANCE_ZONA]
                            else:
                                if len(puntos_zona_conflictiva) > 0:
                                    ARRAY_CONFLICTO = np.array(puntos_zona_conflictiva)
                                    ZONA_CONFLICTO_punto_medio = ARRAY_CONFLICTO.mean(axis=0)

                                    minimos_conflicto = np.array(ARRAY_CONFLICTO).min(axis=0)
                                    maximos_conflicto = np.array(ARRAY_CONFLICTO).max(axis=0)

                                    ZONA_LABELIZADA = np.zeros((numero_clases, DIMENSION))
                                    for fea in range(DIMENSION):
                                        ZONA_LABELIZADA[fea] = [minimos_conflicto[fea] - TOLERANCE_ZONA,
                                                                maximos_conflicto[fea] + TOLERANCE_ZONA]
                                else:
                                    print('ACURACY TOTAL')

                            maximos = []
                            for i in range(numero_clases):
                                cercano = lista_centroides[0][i]
                                group_tmp = []
                                group_tmp.append(cercano)
                                for j in range(1, len(lista_centroides)):
                                    indice = GNG_Funciones.closest_node(cercano, lista_centroides[j])
                                    group_tmp.append(lista_centroides[j][indice])
                                conj = np.array(group_tmp)[:, feature_NO_imp_num]
                                ext = np.zeros(1)
                                np.concatenate((conj[1:], ext))
                                maxi = np.max(conj - np.concatenate((conj[1:], ext)))
                                maximos.append(maxi)

                            puntos=np.array(puntos_zona_conflictiva)
                            arr_lb = np.array(label_zona_conflictiva)
                            lista_etiquetas_test.append(ytest)  # SUPERVISADO

                            resDEF = np.abs(comparativa - copia_label_DEF_test)
                            errorDEF =len(np.where(resDEF != 0)[0]) / len(resDEF)
                            totalDEF = 100 * (1 - errorDEF)

                            array_resultados[contador] = (1 - errorDEF)
                            array_error[contador] = errorDEF

                            acumulado = GNG_Funciones.calcularAcumuladonew(contador, array_resultados, array_resultados[0])#25 ,vol_batch)
                            array_resultados_acumulados[contador] = acumulado
                            array_error_acumulado[contador] = GNG_Funciones.calcular_ERROR_Acumuladonew(contador, array_error, array_error[0])#25 ,vol_batch)

                            print('F1 ',F1[contador])
                            print(contador, '  RESULTADOS:  ', totalDEF)

                            # F1 y Youden usan las ETIQUETAS (preds_test)
                            F = metrics.micro_f1(comparativa, copia_label_DEF_test, kkc_fit)
                            youd = metrics.youden_index_evl(comparativa, copia_label_DEF_test, kkc_fit)
                            Youden[contador] = youd
                            micro_F1_osr[contador] = F
                            print(f"microF1_OSR: {F}, youden: {youd}")

                            distancia_comparativa = GNG_Funciones.calculo_distancia_centroides(np.array(lista_centroides_modelos_guardados)[-1].tolist(), centroides_transformados, numero_clases)
                            if (np.array(distancia_comparativa) >= distacia_guardar_modelos).any():
                                inicio=0
                                seleccion_TMP = []
                                todos_TMP = []
                                for g in range(numero_clases):
                                    lista_tmp = GNG_Funciones.flatten(POSICIONESGNG[contador][inicio:inicio + proporciones_ACTUALES[g]])
                                    tam_tmp = np.array(POSICIONESGNG[contador][inicio:inicio + proporciones_ACTUALES[g]]).shape[0]
                                    array_tmp = np.array(lista_tmp).reshape(tam_tmp, DIMENSION)
                                    out = GNG_Funciones.SeleccionBy1Class(array_tmp,np.array(centroides_transformados)[g,:DIMENSION], including_percentage[i_name])
                                    seleccion_TMP.append(out)
                                    todos_TMP.append(array_tmp)
                                    inicio += proporciones_ACTUALES[g]
                                    if g==0:
                                        lista_gng_guardados.append(POSICIONESGNG[contador].tolist())
                                        lista_etiquetas_gng_guardados.append(labelsGNG[contador])#.append(labelsGNG[contador].tolist())
                                        lista_centroides_modelos_guardados.append(np.array(centroides_transformados))
                                        lista_numero_clusters_modelos_guardados.append(numero_clases)
                                        lista_proporciones_modelos_guardados.append(proporciones_ACTUALES)
                                        if numero_clases != numero_clases_ORIGINAL:
                                            lista_asignaciones_modelos_guardados.append(asignacion)
                                            lista_proporciones_clusters_guardados.append(lista_proporciones[-1])
                                    elif g == numero_clases - 1:
                                        lista_guardados.append(todos_TMP)
                                        lista_seleccion_guardados.append(seleccion_TMP)
                    else:

                        ypred = []
                        if numero_unlabeled>=NUM_instancias_poriteracion:
                            numero_paquetes = (int)(NUM_instancias_poriteracion / vol_batch)
                        else:
                            numero_paquetes = (int)(numero_unlabeled / vol_batch)

                        for g in range(numero_paquetes, 0, -1):
                            ypred.append(lista_etiquetas_test[contador - g])

                        ypred = np.array(ypred).flatten()
                        Xparcial_T, yparcial_T, XtestP, ytestP = GNG_Funciones.definir_sets_training_test_parciales(X_DEF, YDEF, ypred,flag_cambio, contador, vol_batch,numero_labeled, numero_unlabeled,NUM_instancias_poriteracion,NUM_inicio_etiquetados)
                        kkc_fitP = np.unique(yparcial_T)  # known_labels
                        # clases desconocidas
                        all_labelsP = np.unique(ytestP)
                        unknown_labelsP = [c for c in all_labelsP if c not in kkc_fitP]
                        Minusunknown_labelsP = [c for c in kkc_fitP if c not in all_labelsP]
                        # |C_TR|
                        c_trP = len(kkc_fitP)
                        # |C_TE|
                        c_teP =  len(all_labelsP)
                        OP = 100 * osr.openness(c_trP, c_teP)
                        limittrain = int((Xparcial_T.shape[0] // 4) ** 0.5)
                        scores = []
                        disttrainmin=[]
                        disttrainmax = []
                        scores.append(0.0)
                        scores.append(0.1)
                        for k in range(2, limittrain + 1):
                            model = KMeans(n_clusters=k)
                            model.fit(Xparcial_T)
                            cent=model.cluster_centers_
                            disttraintmpMin,disttraintmpMax = GNG_Funciones.distancias_extremas(cent)
                            disttrainmin.append(disttraintmpMin)
                            disttrainmax.append(disttraintmpMax)
                            pred = model.predict(Xparcial_T)
                            score = silhouette_score(Xparcial_T, pred)
                            scores.append(score)

                        index_firsttrain, index_secondtrain = GNG_Funciones.indices_mayores(np.array(scores))
                        first_elementtrain = scores[index_firsttrain]
                        second_elementtrain = scores[index_secondtrain]
                        indexTRAIN_tmp = index_firsttrain
                        diferencia_clusterestrain = first_elementtrain - second_elementtrain

                        limitTEST = int((XtestP.shape[0] // 4) ** 0.5)
                        scoresTEST = []
                        scoresTEST.append(0.0)
                        scoresTEST.append(0.1)
                        disttestmin=[]
                        disttestmax = []
                        for k in range(2, limitTEST + 1):
                            modelt = KMeans(n_clusters=k)
                            modelt.fit(XtestP)
                            cent_test = modelt.cluster_centers_
                            disttesttmpmin,disttesttmpmax=GNG_Funciones.distancias_extremas(cent_test)
                            disttestmin.append(disttesttmpmin)
                            disttestmax.append(disttesttmpmax)
                            predT = modelt.predict(XtestP)
                            scoret = silhouette_score(XtestP, predT)
                            scoresTEST.append(scoret)
                        num_max=np.min((limitTEST,limittrain))
                        diferenciasmin=[]
                        diferenciasmax = []
                        for a in range (num_max-2):
                            diferenciasmin.append(np.abs(disttestmin[a]-disttrainmin[a]))
                            diferenciasmax.append(np.abs(disttestmax[a] - disttrainmax[a]))

                        index_firstTEST, index_secondTEST = GNG_Funciones.indices_mayores(np.array(scoresTEST))
                        first_elementTEST = scoresTEST[index_firstTEST]
                        second_elementTEST = scoresTEST[index_secondTEST]
                        indexTEST_tmp = index_firstTEST

                        ordenado = np.sort(scoresTEST)[::-1]
                        diffsTEST = np.abs(np.diff(ordenado))

                        ordenadoT = np.sort(scores)[::-1]
                        diffsTRAIN = np.diff(ordenadoT)

                        limite_clusters=np.abs(first_elementtrain - first_elementTEST)
                        print(limite_clusters, LIMITE_S)
                        print(contador,index_firstTEST,index_firsttrain)

                        if index_firstTEST==np.array(lista_centroides[-1])[:,:DIMENSION].shape[0]:
                            modeltmp = KMeans(n_clusters=index_firstTEST)
                            modeltmp.fit(XtestP)
                            cent_testmp2 = modeltmp.cluster_centers_
                            flag_movimientot = GNG_Funciones.aumenta_distancia(np.array(lista_centroides[-1])[:,:DIMENSION], cent_testmp2, ref_radios)
                            print('Aumenta distancia ' ,flag_movimientot)#ref_radios   diametro
                        elif index_secondTEST==np.array(lista_centroides[-1])[:,:DIMENSION].shape[0]:
                            modeltmp = KMeans(n_clusters=index_secondTEST)
                            modeltmp.fit(XtestP)
                            cent_testmp2 = modeltmp.cluster_centers_
                            flag_movimientot = GNG_Funciones.aumenta_distancia(
                                np.array(lista_centroides[-1])[:, :DIMENSION], cent_testmp2, ref_radios)
                            print('Aumenta distancia ', flag_movimientot)  # ref_radios   diametro
                        else:
                            flag_movimientot=True

                        if len(lista_asignaciones)>0:
                            ultima_asignacion=lista_asignaciones[-1]
                        else:
                            ultima_asignacion=[]


                        # La cardinalidad no lee unknown_labelsP. Esas etiquetas solo
                        # sirven para imprimir si la regla geometrica no coincide con el numero real.
                        if rastreador is None:
                            rastreador = ClassTracker(
                                X_DEF[:NUM_inicio_etiquetados],
                                YDEF[:NUM_inicio_etiquetados],
                            )
                        n_new, _ = rastreador.paso(XtestP)
                        if len(unknown_labelsP) > 0:
                            delta_real = 1
                        elif OP != 0:
                            delta_real = -1
                        else:
                            delta_real = 0
                        if n_new != delta_real:
                            print(
                                'DISCREPANCIA', file, 'lote', contador,
                                'K_geom', numero_clusters + n_new,
                                'delta_geom', n_new, 'delta_real', delta_real
                            )
                        if n_new:
                            cambio_clusters = 2
                            numero_clusters = max(numero_clases_ORIGINAL, numero_clusters + n_new)
                            SILHOUTE_TOLERANCE = 0.12
                            if n_new > 0:
                                print('CLASE NUEVA SIN ETIQUETAS', contador, numero_clusters)
                            else:
                                print('CLASE CERRADA SIN ETIQUETAS', contador, numero_clusters)
                        elif( numero_clases_ORIGINAL!=numero_clusters_INICIALES):# ( (np.abs(index_firstTEST-index_firsttrain)==1)&(index_firstTEST!=numero_clusters))numero_clases_ORIGINAL!=numero_clusters_INICIALES
                            print('ENTRAMOS')
                            index_firstTEST,index_secondTEST,index_firsttrain,index_secondtrain,cambio_clusters,flag_pasot,flag_paso,n_c=GNG_Funciones.analisis_clusters_TEST(flag_EZ,numero_clusters,
                                            index_firstTEST, index_secondTEST, XtestP, index_firsttrain, index_secondtrain, Xparcial_T,disttestmax, disttrainmax, ref_radios, number_instances,
                                            tamaño_dataset,flag_movimientot, lista_centroides[-1], lista_proporciones[-1], array_distancia_centroides[contador - 1],numero_nodos_GNG_clase,
                                            diametro, DIMENSION, flag_balanced,flag_movimiento, diffsTEST, diffsTRAIN, including_percentage[i_name], ultima_asignacion,
                                            diferencias, distancia_limite, limite_clusters, LIMITE_S,numero_clases_ORIGINAL, contador, contador_cambio)

                            if ((cambio_clusters==-1)&(index_firstTEST == numero_clusters)):
                                numero_clusters = lista_numero_clusters[-1]
                                cambio_clusters = 0
                            elif (cambio_clusters == 3) :
                                if  index_firstTEST != numero_clusters:
                                    cambio_clusters = 2
                                    numero_clusters = index_firstTEST
                                    SILHOUTE_TOLERANCE = 0.12
                                    print('M1',contador,numero_clusters)
                                elif (numero_clusters < index_secondTEST):
                                    numero_clusters = index_secondTEST
                                    SILHOUTE_TOLERANCE = 0.12
                                    cambio_clusters = 2
                                    print('M2',contador, numero_clusters)
                                elif (numero_clusters > index_secondTEST):
                                    numero_clusters = index_secondTEST
                                    SILHOUTE_TOLERANCE = 0.12
                                    cambio_clusters = 2
                                    print('M3',contador, numero_clusters)
                                else:
                                    numero_clusters = lista_numero_clusters[-1]
                                    cambio_clusters = 0
                                    print('M4', contador, numero_clusters)
                            elif ((index_firsttrain != index_firstTEST)  & (index_firstTEST != numero_clusters)):#duda OR
                                numero_clusters = index_firstTEST
                                SILHOUTE_TOLERANCE = 0.12
                                cambio_clusters = 2
                            elif ((cambio_clusters==-1)&(index_firstTEST > numero_clusters)):
                                numero_clusters = index_firstTEST
                                SILHOUTE_TOLERANCE = 0.12
                                cambio_clusters = 2
                            elif(contador < contador_cambio + 5):
                                numero_clusters = lista_numero_clusters[-1]
                                cambio_clusters = 0
                            elif ( (contador >= contador_cambio + 5) & (index_firsttrain == numero_clusters)&(index_firstTEST == numero_clusters)):
                                numero_clusters = lista_numero_clusters[-1]
                                cambio_clusters = 0
                                print('Continuamos ', numero_clusters)
                            elif ((index_firsttrain==index_firstTEST)&(flag_pasot==1) and (flag_paso==1)):
                                print('REVISO CAMBIO  ', numero_clusters)
                                numero_clusters = lista_numero_clusters[-1]
                                cambio_clusters = 0
                            else:
                                cambio_clusters= 0
                                index_firstTEST = index_secondTEST
                                numero_clusters = index_firstTEST
                                print(' Sin CAMBIO ',numero_clusters)

                            if flag_EZ==1:
                                num_cl_comp=numero_clases_ORIGINAL
                            else:
                                num_cl_comp = numero_clases

                            if len(lista_asignaciones) == 0:
                                YtransTMP2, cent0TMP2, asignacionTMP2, cambio_clustersTMP2, prop_clustersTMP2 = GNG_Funciones.asignacion_clusters_clases(
                                0,
                                X=XtestP, Y=[], flag=flag_balanced, relacion=[], n_c=num_cl_comp,
                                n_clu=numero_clusters, dim=DIMENSION, inclu_p=including_percentage[i_name],
                                centr=lista_centroides[-1])
                            else:
                                YtransTMP2, cent0TMP2, asignacionTMP2, cambio_clustersTMP2, prop_clustersTMP2 = GNG_Funciones.asignacion_clusters_clases(
                                    0,
                                    X=XtestP, Y=[], flag=flag_balanced, relacion=[], n_c=num_cl_comp,
                                    n_clu=numero_clusters, dim=DIMENSION, inclu_p=including_percentage[i_name],
                                    centr=lista_centroides[-1], asignacion_previa=lista_asignaciones[-1])

                            cantidad_verd = np.min(prop_clustersTMP2[:, -1])
                            print('cantidad minima verdad ', cantidad_verd)
                            if ((cantidad_verd == 0)):
                                if index_firstTEST < numero_clusters:
                                    index_firstTEST = index_firstTEST + 1
                                else:
                                    index_firstTEST = index_firstTEST - 1

                            if((flag_EZ == 0) & (cambio_clusters != cambio_clustersTMP2)):
                                indtest=indtr=0
                                if( np.abs(indexTEST_tmp-index_secondTEST)==1):
                                    indtest = GNG_Funciones.analisis_morfologico(0, flag_movimientot, indexTEST_tmp,
                                                                              index_secondTEST, XtestP,
                                                                              lista_centroides[-1],
                                                                              lista_proporciones[-1],
                                                                              array_distancia_centroides[contador - 1],
                                                                              numero_nodos_GNG_clase, ref_radios,
                                                                              diametro, DIMENSION, flag_balanced)  # ref_radios

                                if( np.abs(indexTRAIN_tmp-index_secondtrain)==1):
                                    indtr = GNG_Funciones.analisis_morfologico(1, flag_movimientot,
                                                                           indexTRAIN_tmp,
                                                                           index_secondtrain, Xparcial_T,
                                                                           lista_centroides[-1], lista_proporciones[-1],
                                                                           array_distancia_centroides[
                                                                               contador - 1],
                                                                           numero_nodos_GNG_clase, ref_radios,
                                                                           diametro, DIMENSION, flag_balanced)


                                if ((indtest==indtr)&(indtest==lista_numero_clusters[-1])&(cambio_clusters==0)):
                                    numero_clusters=lista_numero_clusters[-1]
                                    cambio_clusters=0
                            elif cambio_clustersTMP2==1:
                                indtest = indtr = 0
                                if (np.abs(indexTEST_tmp - index_secondTEST) == 1):
                                    indtest = GNG_Funciones.analisis_morfologico(0, flag_movimientot, indexTEST_tmp,
                                                                                 index_secondTEST, XtestP,
                                                                                 lista_centroides[-1],
                                                                                 lista_proporciones[-1],
                                                                                 array_distancia_centroides[contador - 1],
                                                                                 numero_nodos_GNG_clase, ref_radios,
                                                                                 diametro, DIMENSION,
                                                                                 flag_balanced)  # ref_radios
                                if (np.abs(indexTRAIN_tmp - index_secondtrain) == 1):
                                    indtr = GNG_Funciones.analisis_morfologico(1, flag_movimientot,
                                                                               indexTRAIN_tmp,
                                                                               index_secondtrain, Xparcial_T,
                                                                               lista_centroides[-1], lista_proporciones[-1],
                                                                               array_distancia_centroides[
                                                                                   contador - 1],
                                                                               numero_nodos_GNG_clase, ref_radios,
                                                                               diametro, DIMENSION, flag_balanced)
                                if ((indtest == indtr) & (indtest == lista_numero_clusters[-1]) ):
                                    numero_clusters = lista_numero_clusters[-1]
                                    cambio_clusters = 0
                        else:
                            numero_clusters=lista_numero_clusters[-1]
                            cambio_clusters=0

                        lista_numero_clusters.append(numero_clusters)

                        if ((numero_clusters > numero_clases_ORIGINAL)&(cambio_clusters==2)):
                            numero_clases= numero_clusters
                            Xtrans = XtestP

                            if flag_EZ == 1:
                                num_cl_comp = numero_clases_ORIGINAL
                            else:
                                num_cl_comp = numero_clases

                            if len(lista_asignaciones) == 0:
                                Ytrans, cent0, asignacion,cambio_clusters2,prop_clusters = GNG_Funciones.asignacion_clusters_clases(0,X=Xtrans, Y=[],flag=flag_balanced,relacion=[],  n_c=num_cl_comp,n_clu=numero_clusters, dim=DIMENSION,inclu_p=including_percentage[i_name],centr=lista_centroides[-1])
                            else:
                                Ytrans, cent0, asignacion, cambio_clusters2, prop_clusters = GNG_Funciones.asignacion_clusters_clases(0,
                                    X=Xtrans, Y=[], flag=flag_balanced, relacion=[], n_c=num_cl_comp,
                                    n_clu=numero_clusters, dim=DIMENSION, inclu_p=including_percentage[i_name],
                                    centr=lista_centroides[-1],asignacion_previa=lista_asignaciones[-1])

                            if ((flag_EZ == 0)&(cambio_clusters!=cambio_clusters2)):
                                indtest = indtr = 0
                                if (np.abs(indexTEST_tmp - index_secondTEST) == 1):
                                    indtest = GNG_Funciones.analisis_morfologico(0, flag_movimientot, indexTEST_tmp,
                                                                                 index_secondTEST, XtestP,
                                                                                 lista_centroides[-1],
                                                                                 lista_proporciones[-1],
                                                                                 array_distancia_centroides[contador - 1],
                                                                                 numero_nodos_GNG_clase, ref_radios,
                                                                                 diametro, DIMENSION,
                                                                                 flag_balanced)  # ref_radios
                                if (np.abs(indexTRAIN_tmp - index_secondtrain) == 1):
                                    indtr = GNG_Funciones.analisis_morfologico(1, flag_movimientot,
                                                                               indexTRAIN_tmp,
                                                                               index_secondtrain, Xparcial_T,
                                                                               lista_centroides[-1], lista_proporciones[-1],
                                                                               array_distancia_centroides[
                                                                                   contador - 1],
                                                                               numero_nodos_GNG_clase, ref_radios,
                                                                               diametro, DIMENSION, flag_balanced)
                                if ((indtest == indtr) & (indtest == lista_numero_clusters[-1]) & (
                                        cambio_clusters == 0)):
                                    numero_clusters = lista_numero_clusters[-1]
                                    cambio_clusters = 0

                            # SI HAY CAMBIO en el numero de culsters si hay que tener en cuenta OSR es aqui donde se debe analizar
                            cantidad = np.min(lista_proporciones[-1], axis=0)
                            cantidad_act =np.min(prop_clusters[:, -1])
                            print('cantidad minima anterior ', cantidad)
                            print('cantidad minima actual ', cantidad_act)
                            print('ACTUALES  ',np.sum(prop_clusters[:, -1], axis=0),prop_clusters)
                            print('ANTERIORES  ', np.sum(lista_proporciones[-1], axis=0),lista_proporciones[-1])
                            lista_asignaciones.append(asignacion)

                            desdoble=0
                            #Se analizan las localizaciones de los centroides y las proporciones , para determinar si hay escision o no.
                            if(cambio_clusters != 0):
                                #Para definir el threshold:
                                if len(lista_centroides[-1])>2:
                                    distancia_min_ant,distancias_max_ant=GNG_Funciones.distancias_extremas(np.array(lista_centroides[-1])[:,:DIMENSION])
                                    splt_max_distance_ant = (distancias_max_ant - distancia_min_ant) / numero_clusters
                                else:
                                    splt_max_distance_ant=GNG_Funciones.distancia(lista_centroides[-1][0], lista_centroides[-1][1])

                                if len(cent0)>2:
                                    distancia_min_act, distancias_max_act = GNG_Funciones.distancias_extremas(cent0)
                                    splt_max_distance_act = (distancias_max_act - distancia_min_act) / numero_clusters
                                else:
                                    splt_max_distance_ant=GNG_Funciones.distancia(cent0[-1][0], cent0[-1][1])

                                splt_distance=np.array([splt_max_distance_ant,splt_max_distance_act]).min()

                                mapping, noasgnados = GNG_Funciones.match_centroids(np.array(lista_centroides[-1])[:,:DIMENSION], cent0, split_threshold=splt_distance)
                                print("Mapping de centroides:", mapping)

                                for nn in noasgnados:
                                    distancias = np.linalg.norm(
                                        np.array(lista_centroides[-1])[:, :DIMENSION] - cent0[nn], axis=1)
                                    sol = distancias.min()
                                    print('distancia division ', sol, distancia_min_act)

                                    if sol < distancia_min_act:
                                        desdoble = 1
                                        indice_centrode_cercano = GNG_Funciones.punto_mas_cercano(cent0[nn], np.array(
                                            lista_centroides[-1])[:, :DIMENSION])
                                        print(f" El centroide {cent0[indice_centrode_cercano]} se dividió en {len(noasgnados)} centroides: {cent0[nn]}")

                                if ((desdoble==1)):
                                    print('OSR: SI HAY ESCISION')
                                else:
                                    print('OSR: NO HAY ESCISION')
                        elif ((numero_clusters <= numero_clases_ORIGINAL)&(cambio_clusters==2)):
                            numero_clases= numero_clusters
                            Xtrans = XtestP

                            if flag_EZ == 1:
                                num_cl_comp = numero_clases_ORIGINAL
                            else:
                                num_cl_comp = numero_clases

                            Ytrans, cent0, asignacion,cambio_clusters2,prop_clusters = GNG_Funciones.asignacion_clusters_clases(0,X=Xtrans, Y=[],flag=flag_balanced,relacion=[],  n_c=num_cl_comp,n_clu=numero_clusters, dim=DIMENSION,inclu_p=including_percentage[i_name],centr=lista_centroides[-1])

                            cantidad = np.min(lista_proporciones[-1], axis=0)#LISTA_PROPORCINES_CLUSTERS
                            cantidad_act =np.min(prop_clusters[:, -1])
                            print('cantidad minima anterior ', cantidad)
                            print('cantidad minima actual ', cantidad_act)
                            print('ACTUALES  ',np.sum(prop_clusters[:, -1], axis=0),prop_clusters)
                            print('ANTERIORES  ', np.sum(lista_proporciones[-1], axis=0),lista_proporciones[-1])#LISTA_PROPORCINES_CLUSTERS
                            if cantidad_act==0:
                                lista_asignaciones.append(asignacion[:-1])
                            else:
                                lista_asignaciones.append(asignacion)
                        elif (numero_clusters > numero_clases_ORIGINAL):
                            Xtrans = Xparcial_T
                            numero_clases= numero_clusters

                            if flag_EZ == 1:
                                num_cl_comp = numero_clases_ORIGINAL
                            else:
                                num_cl_comp = numero_clases

                            relacion_parcial = np.full(num_cl_comp, -1.1)
                            tamaño = Xparcial_T.shape[0]
                            for i in range(num_cl_comp):
                                relacion_parcial[i] = Xparcial_T[yparcial_T == i].shape[0]

                            Ytrans, cent0, asignacion2, cambio_clusters2, prop_clusters = GNG_Funciones.asignacion_clusters_clases(0,
                                X=Xparcial_T, Y=yparcial_T, flag=flag_balanced, relacion=relacion_parcial,
                                n_c=num_cl_comp, n_clu=numero_clusters, dim=DIMENSION,
                                inclu_p=including_percentage[i_name], asignacion_previa=lista_asignaciones[-1],
                                centr=lista_centroides[-1])
                            if ((flag_EZ == 0)&(cambio_clusters != cambio_clusters2)):
                                print('DISCREPANCIAS')
                        else:
                            Xtrans = Xparcial_T
                            Ytrans = yparcial_T

                        Xparcial, yparcial = GNG_Funciones.SeleccionByClass(Xtrans, Ytrans, numero_clases, including_percentage[i_name])

                        mytrain = Xparcial
                        my_y_train = yparcial
                        #comprobamos

                        if cambio_clusters==0:
                            print(contador, numero_clusters, numero_clases, numero_clases_ORIGINAL)
                            proporciones_ACTUALES = lista_proporciones[contador-1]
                            posiciones_anterioores_list = []
                            ini = 0
                            for d in range(numero_clases):
                                posiciones_anterioores_list.append(POSICIONESGNG[contador - step][ini:ini+proporciones_ACTUALES[d]].tolist())
                                ini+= proporciones_ACTUALES[d]

                            GNG_ANT, etiquetas_ant = GNG_Funciones.from_list_t_array(posiciones_anterioores_list, DIMENSION, numero_clases)
                        elif cambio_clusters==2:
                            print('OSR.Comenzar de cero.')
                            if numero_clusters==lista_numero_clusters[-2]:
                                print('MAL')
                                break
                            print(contador,numero_clusters,numero_clases,numero_clases_ORIGINAL)
                            SILHOUTE_TOLERANCE = 0.12
                            flag_cambio = 1
                            contador_cambio = contador

                            mytrain = XtestP
                            my_y_train = Ytrans

                            # Debemos re-construir la GNG
                            tmp_node, gng_slayer =GNG_Funciones.reset_gng(1,mytrain, DIMENSION, 0.2, 20, numero_nodos_GNG, dim_max, tolerancia_coincidencia_punto,gng_anterior=[])
                            nodos_temporales = np.array(tmp_node).reshape(np.array(tmp_node).shape[0], DIMENSION)
                            labels_tmp = GNG_Funciones.funcion_comparativa(mytrain, my_y_train, nodos_temporales, metrica='KNN5_GNG')

                            lista_ordenadostmp, indices_ordenados_tmp, prop = GNG_Funciones.filtrarGNG(nodos_temporales,
                                                                                                       labels_tmp,
                                                                                                       numero_clases)
                            array_posicionestmp, Array_etiquetastmp = GNG_Funciones.from_list_t_array(
                                lista_ordenadostmp, DIMENSION, numero_clases)
                            if nodos_temporales.shape[0]!=numero_nodos_GNG:
                                print('TAMAÑO ')

                            POSICIONESGNG[contador] = GNG_Funciones.flatten(lista_ordenadostmp)
                            labelsGNG[contador] = Array_etiquetastmp

                            if numero_clusters != numero_clases_ORIGINAL:# numero_clusters > numero_clases_ORIGINAL:
                                kmeans = KMeans(n_clusters=numero_clases, max_iter=MAX_ITERATIONS, random_state=0,init=cent0, tol=CONVERGENCE_TOLERANCE)
                            else:
                                kmeans = KMeans(n_clusters=numero_clases, max_iter=MAX_ITERATIONS, random_state=0,init=INITIALIZE_CLUSTERS, tol=CONVERGENCE_TOLERANCE)
                            kmeans.fit(mytrain, my_y_train)

                            if numero_clases == 2:
                                DISTANCIA_CENTROIDES = np.linalg.norm(kmeans.cluster_centers_)
                                MITAD_DISTANCIA_11 = DISTANCIA_CENTROIDES / 11

                                TOLERANCE_GNG=[MITAD_DISTANCIA_11 for j in range(numero_clases)]
                            else:
                                d=[]
                                for j in range (numero_clases):
                                    tmp=np.linalg.norm(kmeans.cluster_centers_[j] - kmeans.cluster_centers_, axis=1)
                                    d.append(tmp)
                                d=np.array(d).reshape((numero_clases,numero_clases))
                                d=np.where(np.array(d) != 0., d, 100)
                                TOLERANCE_GNG=d.min(axis=1)
                                TOLERANCE_GNG=TOLERANCE_GNG/ 11

                            cent_tr, cent_tst = GNG_Funciones.obtencion_ceniodes(numero_clases, DIMENSION, nodos_temporales, labels_tmp, XtestP,np.array(Ytrans))
                            distancia_train_test_tmp= GNG_Funciones.calculo_distancia_centroides(cent_tr, cent_tst,  numero_clases)
                            if (len(lista_centroides[-1])!=len(cent_tst)):
                                flag_movimiento=False
                            else:
                                flag_movimiento = GNG_Funciones.aumenta_distancia(cent_tr, cent_tst,diametro)#ref_radios

                            if ((flag_EZ!=1)&(len(distancia_train_test_OROGINARIA) > numero_clases_ORIGINAL)):
                                distancia_train_test_INICIAL.append(distancia_train_test_tmp[-1])
                            elif((flag_EZ!=1)&(len(distancia_train_test_OROGINARIA) < numero_clases_ORIGINAL)):
                                distancia_train_test_INICIAL.pop()
                            elif ((flag_EZ==1)&(len(distancia_train_test_OROGINARIA) > numero_clases)):
                                distancia_train_test_INICIAL.append(distancia_train_test_tmp[-1])
                            elif((flag_EZ==1)&(len(distancia_train_test_OROGINARIA) < numero_clases)):
                                distancia_train_test_INICIAL.pop()

                            print(contador, '  distancia_train_test_OROGINARIA  ', distancia_train_test_OROGINARIA)

                            if numero_clusters>2:
                                LIMITE_S=LIMITE_S/numero_clases
                                print(contador,' LIMITE_S ',LIMITE_S)

                            t_max = []
                            for i in range(numero_clases):
                                tmp_max = GNG_Funciones.distancias_al_centroide_INTRA_CLUSTER(mytrain[my_y_train == i], cent0[i])
                                t_max.append(tmp_max)
                            lista_distancias_maximas.append(t_max)

                            Tolerance_DENSITY, TOLERANCE_radio_clase, TOLERANCE_poblacion = GNG_Funciones.funcion_definicion_DENSIDAD(
                                mytrain, my_y_train, vol_batch, numero_vecinos_ENTROPIA, numero_clases, DIMENSION, cent0)
                            ref_radios = np.max(TOLERANCE_radio_clase)

                            proporciones = np.array(prop)
                            lista_proporciones.append(proporciones)
                            proporciones_ORIGINALES = prop
                            proporciones_ORIGINALES_CLASES = [0 for j in range(numero_clases_ORIGINAL)]
                            ratio_distribucion_INICIAL_etiquetas = np.max(proporciones_ORIGINALES) / np.min(
                                proporciones_ORIGINALES)

                            if flag_EZ!=0:
                                for j in range(XtestP.shape[0]):
                                    labels_DEFINITIVA_P[j] = int(asignacion[Ytrans[j]][1])
                            else:
                                labels_DEFINITIVA_P = Ytrans

                            if DIMENSION == 2:
                                tmpCENTROIDE = np.zeros((numero_clases, 3))
                                for q in range(numero_clases):
                                    tmpCENTROIDE[q] = np.array([cent0[q][0], cent0[q][1], 0])
                                lista_centroides.append(tmpCENTROIDE)
                            else:
                                tmpCENTROIDE=cent0
                                lista_centroides.append(cent0)

                            TOLERANCE_distancia_entre_centroides, flag_densidad = GNG_Funciones.calculo_TOLERANCE_centroides(
                                tmpCENTROIDE, numero_clases, TOLERANCE_radio_clase)
                            array_flag_densidad[contador] = flag_densidad
                            if numero_clases > 2:
                                array_distancia_centroides[contador] = np.min(TOLERANCE_distancia_entre_centroides)[0]
                            else:
                                array_distancia_centroides[contador] = np.min(TOLERANCE_distancia_entre_centroides)

                            proporciones_ACTUALES = lista_proporciones[-1]

                            inicio = 0
                            seleccion_TMP = []
                            todos_TMP = []
                            for g in range(numero_clases):
                                array_tmp=POSICIONESGNG[contador][int(inicio):int(inicio) + (int)(proporciones_ACTUALES[g])]
                                out = GNG_Funciones.SeleccionBy1Class(array_tmp,
                                                                      np.array(cent0)[g, :DIMENSION],
                                                                      including_percentage[i_name])
                                seleccion_TMP.append(out)
                                todos_TMP.append(array_tmp)
                                inicio += proporciones_ACTUALES[g]
                                if g == 0:
                                    lista_gng_guardados.append(POSICIONESGNG[contador].tolist())
                                    lista_etiquetas_gng_guardados.append(
                                        labelsGNG[contador])  # .append(labelsGNG[contador].tolist())
                                    lista_centroides_modelos_guardados.append(np.array(tmpCENTROIDE))
                                    lista_numero_clusters_modelos_guardados.append(numero_clases)
                                    lista_proporciones_modelos_guardados.append(proporciones_ACTUALES)
                                    if numero_clases != numero_clases_ORIGINAL:
                                        lista_asignaciones_modelos_guardados.append(asignacion)
                                        lista_proporciones_clusters_guardados.append(
                                            lista_proporciones[-1])
                                elif g == numero_clases - 1:
                                    lista_guardados.append(todos_TMP)
                                    lista_seleccion_guardados.append(seleccion_TMP)
                            lista_etiquetas_test.append(labels_DEFINITIVA_P)
                            comparativaP = ytestP

                            resDEF = np.abs(comparativaP - labels_DEFINITIVA_P)
                            errorDEFP = len(np.where(resDEF != 0)[0]) / len(resDEF)
                            totalDEFP = 100 * (1 - errorDEFP)

                            array_resultados[contador] = (1 - errorDEFP)
                            array_error[contador] = errorDEFP

                            acumulado = GNG_Funciones.calcularAcumuladonew(contador, array_resultados,
                                                                           array_resultados[0])
                            array_resultados_acumulados[contador] = acumulado
                            array_error_acumulado[contador] = GNG_Funciones.calcular_ERROR_Acumuladonew(contador,
                                                                                                        array_error,
                                                                                                        array_error[0])
                            F1[contador] = (f1_score(ytestP, labels_DEFINITIVA_P, average='macro'))

                            minimos_abosulos = np.zeros((DIMENSION))
                            maximos_absolutos = np.zeros((DIMENSION))
                            imp, feature_imp_num, feature_NO_imp_num, identificador_zona = GNG_Funciones.feature_importance(
                                XtestP, labels_DEFINITIVA_P)
                            if identificador_zona == 9:
                                ZONA_LABELIZADA = np.zeros((numero_clases, DIMENSION))
                            else:
                                ZONA_LABELIZADA = np.zeros((DIMENSION))


                            if identificador_zona != 9:
                                minimos_abosulos = np.min(np.array(lista_centroides[contador]), axis=0)
                                maximos_absolutos = np.max(np.array(lista_centroides[contador]), axis=0)

                                ZONA_LABELIZADA = [minimos_abosulos[feature_imp_num],
                                                   maximos_absolutos[feature_imp_num]]
                            else:
                                if DIMENSION == 2:
                                    for fea in range(DIMENSION):
                                        minimos_abosulos[fea] = min(
                                            min(np.array(lista_centroides[contador])[fea][0:DIMENSION]),
                                            min(np.array(lista_centroides[contador])[fea][0:DIMENSION]))
                                        maximos_absolutos[fea] = max(
                                            max(np.array(lista_centroides[contador])[fea][0:DIMENSION]),
                                            max(np.array(lista_centroides[contador])[fea][0:DIMENSION]))

                                        ZONA_LABELIZADA[fea] = [minimos_abosulos[fea], maximos_absolutos[fea]]
                                else:
                                    for fea in range(DIMENSION):
                                        minimos_abosulos[fea] = min(min(np.array(lista_centroides[contador])[fea]),
                                                                    min(np.array(lista_centroides[contador])[fea]))
                                        maximos_absolutos[fea] = max(max(np.array(lista_centroides[contador])[fea]),
                                                                     max(np.array(lista_centroides[contador])[fea]))

                                        ZONA_LABELIZADA[fea] = [minimos_abosulos[fea], maximos_absolutos[fea]]
                            print('F1 ', F1[contador])
                            print(contador, '  RESULTADOS:  ', totalDEFP)

                            # F1 y Youden usan las ETIQUETAS (preds_test)
                            F = metrics.micro_f1(comparativaP, labels_DEFINITIVA_P, kkc_fitP)
                            youd = metrics.youden_index_evl(comparativaP, labels_DEFINITIVA_P, kkc_fitP)
                            Youden[contador] = youd
                            micro_F1_osr[contador] = F
                            print(f"microF1_OSR: {F},  youden: {youd}")

                            print('OSR: Cambia el numero clases : ',numero_clusters)
                        else:
                            print(contador, numero_clusters, numero_clases, numero_clases_ORIGINAL)
                            TOLERANCE_ACTUAL_GNG = []
                            TOLERANCE_radio_clase_ACTUAL = []
                            proporciones_ACTUALES = []
                            proporciones_anteriores = lista_proporciones[contador - 1]
                            print(lista_proporciones[contador - 1], lista_proporciones[-1])
                            asignaciones_anteriores = lista_asignaciones[-1]

                            cont_c = 0
                            cont_tol = 0
                            for d in range(numero_clases - 1):
                                if (len(asignaciones_anteriores) < len(asignacion)):
                                    if cont_c < numero_clases - 1:
                                        if len(asignaciones_anteriores) > d + 1:
                                            if ((asignacion[d][1] == asignaciones_anteriores[d][1]) and (
                                            (asignacion[d + 1][1] == asignaciones_anteriores[d + 1][1]))):
                                                proporciones_ACTUALES.append(proporciones_anteriores[cont_c])
                                                TOLERANCE_ACTUAL_GNG.append(TOLERANCE_GNG[cont_tol])
                                                TOLERANCE_radio_clase_ACTUAL.append(TOLERANCE_radio_clase[cont_tol])
                                                cont_tol += 1
                                                cont_c += 1
                                            else:
                                                if (asignacion[d][1] == asignacion[d + 1][1]):
                                                    tmp = int(proporciones_anteriores[cont_c] / 2)
                                                    proporciones_ACTUALES.append(tmp)
                                                    proporciones_ACTUALES.append(proporciones_anteriores[cont_c] - tmp)
                                                    TOLERANCE_ACTUAL_GNG.append(TOLERANCE_GNG[cont_tol])
                                                    TOLERANCE_ACTUAL_GNG.append(TOLERANCE_GNG[cont_tol])
                                                    TOLERANCE_radio_clase_ACTUAL.append(TOLERANCE_radio_clase[cont_tol])
                                                    TOLERANCE_radio_clase_ACTUAL.append(TOLERANCE_radio_clase[cont_tol])
                                                    cont_c += 2
                                                    cont_tol += 1
                                                else:
                                                    proporciones_ACTUALES.append(
                                                        proporciones_anteriores[d] + proporciones_anteriores[d + 1])
                                                    TOLERANCE_ACTUAL_GNG.append(TOLERANCE_GNG[cont_tol])
                                                    TOLERANCE_radio_clase_ACTUAL.append(TOLERANCE_radio_clase[cont_tol])
                                                    cont_c += 1
                                                    cont_tol += 1
                                elif((asignaciones_anteriores==asignacion).all()):
                                    print('SITUACION ANTERIOR')
                                    proporciones_ACTUALES=proporciones_anteriores
                                    TOLERANCE_ACTUAL_GNG=TOLERANCE_GNG
                                    TOLERANCE_radio_clase_ACTUAL=TOLERANCE_radio_clase
                                    break
                                else:
                                    if cont_c < numero_clases:
                                        if len(asignacion) > d + 1:
                                            if ((asignacion[d][1] == asignaciones_anteriores[d][1]) and (
                                            (asignacion[d + 1][1] == asignaciones_anteriores[d + 1][1]))):
                                                proporciones_ACTUALES.append(proporciones_anteriores[cont_c])
                                                TOLERANCE_ACTUAL_GNG.append(TOLERANCE_GNG[cont_tol])
                                                TOLERANCE_radio_clase_ACTUAL.append(TOLERANCE_radio_clase[cont_tol])
                                                cont_tol += 1
                                                cont_c += 1
                                            else:
                                                if ((asignacion[d][1] == asignaciones_anteriores[d][1]) & (
                                                        asignaciones_anteriores[d + 1][1] ==
                                                        asignaciones_anteriores[d + 1][1])):
                                                    tmp = int(proporciones_anteriores[cont_c] + proporciones_anteriores[
                                                        cont_c + 1])
                                                    proporciones_ACTUALES.append(tmp)
                                                    TOLERANCE_ACTUAL_GNG.append(TOLERANCE_GNG[cont_tol])
                                                    TOLERANCE_radio_clase_ACTUAL.append(TOLERANCE_radio_clase[cont_tol])
                                                    cont_c += 2
                                                    cont_tol += 1
                                                else:
                                                    proporciones_ACTUALES.append(proporciones_anteriores[cont_c])
                                                    TOLERANCE_ACTUAL_GNG.append(TOLERANCE_GNG[cont_tol])
                                                    TOLERANCE_radio_clase_ACTUAL.append(TOLERANCE_radio_clase[cont_tol])
                                                    cont_c += 1
                                                    cont_tol += 1
                                        else:
                                            if ((asignacion[d][1] == asignacion[d - 1][1]) and (
                                            (asignaciones_anteriores[d][1] == asignaciones_anteriores[d - 1][1]))):
                                                proporciones_ACTUALES.append(proporciones_anteriores[cont_c])
                                                TOLERANCE_ACTUAL_GNG.append(TOLERANCE_GNG[cont_tol])
                                                TOLERANCE_radio_clase_ACTUAL.append(TOLERANCE_radio_clase[cont_tol])
                                                cont_c += 1
                                                cont_tol += 1
                                    else:
                                        if len(asignacion) > d + 1:
                                            if ((asignacion[d][1] == asignacion[d + 1][1]) and (
                                            (asignaciones_anteriores[d][1] == asignaciones_anteriores[d + 1][1]))):
                                                proporciones_ACTUALES.append(proporciones_anteriores[cont_c + 1])
                                                TOLERANCE_ACTUAL_GNG.append(TOLERANCE_GNG[cont_tol])
                                                TOLERANCE_radio_clase_ACTUAL.append(TOLERANCE_radio_clase[cont_tol])
                                                cont_c += 1
                                                cont_tol += 1

                            if len(proporciones_ACTUALES) < numero_clases:
                                if (len(asignaciones_anteriores) < len(asignacion)):
                                    if ((asignacion[cont_c][1] == asignaciones_anteriores[-1][1]) and (
                                    (asignacion[cont_c][1] == asignacion[cont_c + 1][1]))):
                                        tmp = int(proporciones_anteriores[d] / 2)
                                        proporciones_ACTUALES.append(tmp)
                                        proporciones_ACTUALES.append(proporciones_anteriores[d] - tmp)
                                        TOLERANCE_ACTUAL_GNG.append(TOLERANCE_GNG[cont_tol])
                                        TOLERANCE_ACTUAL_GNG.append(TOLERANCE_GNG[cont_tol])
                                        TOLERANCE_radio_clase_ACTUAL.append(TOLERANCE_radio_clase[cont_tol])
                                        TOLERANCE_radio_clase_ACTUAL.append(TOLERANCE_radio_clase[cont_tol])
                                        cont_c += 2
                                        cont_tol += 1
                                else:
                                    if ((asignacion[-1][1] == asignaciones_anteriores[-1][1]) and (
                                    (asignacion[-2][1] == asignaciones_anteriores[-2][1]))):
                                        proporciones_ACTUALES.append(proporciones_anteriores[cont_c])
                                        if len(TOLERANCE_GNG)>cont_tol:
                                            TOLERANCE_ACTUAL_GNG.append(TOLERANCE_GNG[cont_tol])
                                            TOLERANCE_radio_clase_ACTUAL.append(TOLERANCE_radio_clase[cont_tol])
                                        else:
                                            TOLERANCE_ACTUAL_GNG.append(TOLERANCE_GNG[-1])
                                            TOLERANCE_radio_clase_ACTUAL.append(TOLERANCE_radio_clase[-1])
                                        cont_c += 1
                                        cont_tol += 1

                            TOLERANCE_ACTUAL_GNG = np.array(TOLERANCE_ACTUAL_GNG)
                            TOLERANCE_radio_clase_ACTUAL = np.array(TOLERANCE_radio_clase_ACTUAL)
                            TOLERANCE_GNG = TOLERANCE_ACTUAL_GNG
                            TOLERANCE_radio_clase = TOLERANCE_radio_clase_ACTUAL
                            ref_radios=np.max(TOLERANCE_radio_clase)

                            posiciones_anterioores_list = []

                            ini = 0
                            #estan mal las proporciones actuales
                            for s in range(numero_clases):
                                posiciones_anterioores_list.append(
                                    POSICIONESGNG[contador - step][ini:ini + proporciones_ACTUALES[s]].tolist())
                                ini += proporciones_ACTUALES[s]

                            GNG_ANT, etiquetas_ant = GNG_Funciones.from_list_t_array(posiciones_anterioores_list,
                                                                                     DIMENSION, numero_clases)

                        if cambio_clusters!=2:

                            tmp_node, gng_slayer = GNG_Funciones.retrain_gng(0, mytrain, gng_slayer,GNG_ANT, DIMENSION, 0.2, 20, numero_nodos_GNG,dim_max, tolerancia_coincidencia_punto)

                            print('Contador: ', contador)

                            nodos_temporales =np.array(tmp_node).reshape(np.array(tmp_node).shape[0],DIMENSION)
                            labels_tmp = GNG_Funciones.funcion_comparativa(mytrain, my_y_train, nodos_temporales,  metrica='KNN5_GNG')

                            lista_ordenadostmp, indices_ordenados_tmp, prop = GNG_Funciones.filtrarGNG( nodos_temporales, labels_tmp,numero_clases)
                            array_posicionestmp, Array_etiquetastmp = GNG_Funciones.from_list_t_array(lista_ordenadostmp, DIMENSION, numero_clases)
                            print(prop,numero_nodos_GNG_clase/5)
                            while (np.array(prop)<(numero_nodos_GNG_clase/5)).any():#CAMBIADO OSR
                                tmp_node, gng_slayer = GNG_Funciones.reset_gng(0,mytrain,DIMENSION,0.2,20,numero_nodos_GNG,dim_max,tolerancia_coincidencia_punto, gng_anterior=GNG_ANT)

                                nodos_temporales = np.array(tmp_node).reshape(np.array(tmp_node).shape[0], DIMENSION)
                                labels_tmp = GNG_Funciones.funcion_comparativa(mytrain, my_y_train, nodos_temporales, metrica='KNN5_GNG')

                                lista_ordenadostmp, indices_ordenados_tmp, prop = GNG_Funciones.filtrarGNG(nodos_temporales, labels_tmp,  numero_clases)
                                array_posicionestmp, Array_etiquetastmp = GNG_Funciones.from_list_t_array(lista_ordenadostmp, DIMENSION,   numero_clases)
                                print('Contador: ', contador)

                            if cambio_clusters!=0:
                                proporciones = proporciones_ACTUALES
                                if len(lista_proporciones)<=contador:
                                    lista_proporciones.append(proporciones)
                            else:
                                if len(lista_proporciones)<=contador:
                                    lista_proporciones.append(proporciones)


                            array_posiciones_anteriores, Array_etiquetas_anteriores = GNG_Funciones.from_list_t_array(posiciones_anterioores_list, DIMENSION, numero_clases)
                            nodos_t_anterior_para_comparar,nodos_t_actual_ordenados,nodos_ant_no_comp = GNG_Funciones.merge_hungarian(numero_clases,posiciones_anterioores_list, lista_ordenadostmp)

                            tmp_ang_base = []
                            tmp_ang_clase = []
                            tmp_clas_rot = []
                            tmp_clas_tras = []
                            centroides_sin_transformar = []
                            centroides_transformados = []

                            if cambio_clusters==0:
                                for i in range(numero_clases):
                                    A = GNG_Funciones.coord_to_list(np.array(nodos_t_anterior_para_comparar[i]), DIMENSION,len(nodos_t_anterior_para_comparar[i]))
                                    B = GNG_Funciones.coord_to_list(np.array(nodos_t_actual_ordenados[i]), DIMENSION, len(nodos_t_actual_ordenados[i]))

                                    # Recover R and t
                                    if DIMENSION <= 3:
                                        ret_R, ret_t, centroide_sin_t, centroide_t = get_rotation_translation(np.array(A), np.array(B))
                                    else:
                                        ret_R, ret_t, centroide_sin_t, centroide_t = get_rotation_translation_MULTI_D(DIMENSION,np.array(A), np.array(B))

                                    punto_centr = GNG_Funciones.comprobacion_movimiento_punto(centroide_sin_t, ret_R, ret_t, DIMENSION)

                                    angulo_base, angulo_clase, matriz_reconstruida = GNG_Funciones.comprobacion_angulos(i,ret_R,ret_t,DIMENSION)

                                    tmp_ang_base.append(angulo_base)
                                    tmp_ang_clase.append(angulo_clase)
                                    tmp_clas_rot.append(matriz_reconstruida)
                                    tmp_clas_tras.append(ret_t.ravel())
                                    centroides_sin_transformar.append(centroide_sin_t.ravel())
                                    centroides_transformados.append(centroide_t.ravel())

                                t_max=[]
                                for i in range(numero_clases):
                                    tmp_max = GNG_Funciones.distancias_al_centroide_INTRA_CLUSTER(
                                        nodos_t_actual_ordenados[i], centroides_transformados[i])
                                    t_max.append(tmp_max)

                                lista_distancias_maximas.append(t_max)
                                centroides_provisioneles = GNG_Funciones.comprobacion_centroides(lista_centroides[contador - step], centroides_transformados, centroides_sin_transformar, tmp_clas_rot, tmp_clas_tras, tmp_ang_base, tmp_ang_clase,  DIMENSION, numero_clases)
                                lista_centroides.append(centroides_transformados)

                                distancia = GNG_Funciones.calculo_distancia_centroides(lista_centroides[contador - step],  centroides_transformados, numero_clases)

                                for i in range(numero_clases):
                                    if (np.abs(TOLERANCE_GNG[i]) < np.abs(tmp_clas_tras[i])).any():
                                        TOLERANCE_GNG[i] = np.max(np.abs(tmp_clas_tras[i]))
                                    elif (np.abs(TOLERANCE_GNG[i]) < np.abs(distancia[i])).any():
                                        TOLERANCE_GNG[i] = np.max(np.abs(distancia[i]))

                                TOLERANCE_distancia_entre_centroides, flag_densidad = GNG_Funciones.calculo_TOLERANCE_centroides( centroides_transformados,numero_clases, TOLERANCE_radio_clase)
                                array_flag_densidad[contador]=flag_densidad
                                if numero_clases > 2:
                                    array_distancia_centroides[contador] = np.min(TOLERANCE_distancia_entre_centroides)[0]
                                else:
                                    array_distancia_centroides[contador] = np.min(TOLERANCE_distancia_entre_centroides)

                                Lista_nodos_anterior_GNG_transformados = GNG_Funciones.aplicar_movimiento(posiciones_anterioores_list,  tmp_clas_rot, tmp_clas_tras, DIMENSION,   numero_clases)
                                array_nodos_anteriores_transformados, etiquetas_anteriores_trans = GNG_Funciones.from_list_t_array(   Lista_nodos_anterior_GNG_transformados, DIMENSION, numero_clases)

                                #movimineto
                                nodos_anteriores_ordde_hung, nodos_t_act_ord, nodos_ant_no_comp, indicesno,nodos_act_no_ut = GNG_Funciones.merge_hungarianGNG( numero_clases, Lista_nodos_anterior_GNG_transformados, lista_ordenadostmp)
                                nodos_para_AÑADIR_listahung = GNG_Funciones.aplicar_movimiento(nodos_ant_no_comp,tmp_clas_rot, tmp_clas_tras, DIMENSION, numero_clases)

                                GNG_ACTUAL_lista = GNG_Funciones.concateno_listas(nodos_t_act_ord,nodos_para_AÑADIR_listahung, indicesno,proporciones_ACTUALES, numero_clases)
                                array_nodos_actuales_ordenados, etiquetas_actuales_orden = GNG_Funciones.from_list_t_array(GNG_ACTUAL_lista,  DIMENSION,  numero_clases)

                                lista_nodos_ACTUALES_transformados_para_test = GNG_Funciones.aplicar_movimiento(GNG_ACTUAL_lista, tmp_clas_rot, tmp_clas_tras, DIMENSION, numero_clases)
                                array_nodos_actuales_transformados, etiquetas_actuales_trans = GNG_Funciones.from_list_t_array( lista_nodos_ACTUALES_transformados_para_test, DIMENSION, numero_clases)

                                GNG_listaDEF=GNG_Funciones.check_list(GNG_ACTUAL_lista, numero_clusters, centroides_transformados, TOLERANCE_radio_clase,DIMENSION,dim_radio)

                                POSICIONESGNG[contador] = GNG_Funciones.flatten(GNG_listaDEF)
                                labelsGNG[contador] = etiquetas_actuales_orden#.tolist()
                            else:
                                t_c = []
                                for i in range(numero_clases):
                                    if DIMENSION<3:
                                        t_c.append(np.array([cent0[i][0],cent0[i][1],0]))
                                    else:
                                        t_c.append(np.array(cent0[i]))

                                centroides_transformados = t_c
                                lista_centroides.append(centroides_transformados)

                                nodos_anteriores_ordde_hung, nodos_t_act_ord, nodos_ant_no_comp, indicesno,nodos_act_no_ut = GNG_Funciones.merge_hungarianGNG(numero_clases, nodos_t_anterior_para_comparar, lista_ordenadostmp)
                                GNG_ACTUAL_lista = GNG_Funciones.concateno_listas(posiciones_anterioores_list, nodos_ant_no_comp,indicesno, proporciones_ACTUALES, numero_clases)
                                array_nodos_actuales_ordenados, etiquetas_actuales_orden = GNG_Funciones.from_list_t_array( GNG_ACTUAL_lista, DIMENSION, numero_clases)
                                array_nodos_actuales_transformados = array_nodos_actuales_ordenados
                                etiquetas_actuales_trans = etiquetas_actuales_orden

                                GNG_listaDEF = GNG_Funciones.check_list(GNG_ACTUAL_lista, numero_clusters, centroides_transformados, TOLERANCE_radio_clase,DIMENSION, dim_radio)

                                t_max=[]
                                for i in range(numero_clases):
                                    tmp_max = GNG_Funciones.distancias_al_centroide_INTRA_CLUSTER(
                                        GNG_listaDEF[i], centroides_transformados[i])
                                    t_max.append(tmp_max)
                                lista_distancias_maximas.append(t_max)

                                POSICIONESGNG[contador] = GNG_Funciones.flatten( GNG_listaDEF)
                                labelsGNG[contador] = etiquetas_actuales_trans#.tolist()

                            XSELECTED, ySELECTED = GNG_Funciones.SeleccionByClass(array_nodos_actuales_transformados, etiquetas_actuales_trans, numero_clases,including_percentage[i_name])  # param

                            labels_DEFINITIVA_P=GNG_Funciones.funcion_predecir(Xfeatures=XSELECTED, Xlabels=ySELECTED,XTRAIN=mytrain,YTRAIN=my_y_train,  test=XtestP, flag=flag_densidad,Tol_DENSITY=Tolerance_DENSITY,metrica=metrica_param[i_name],
                                                                  n_c=numero_clases, dim=DIMENSION,centroides=centroides_transformados, TOL_radio_clase=TOLERANCE_radio_clase,ZONA=ZONA_LABELIZADA,ident_Z=identificador_zona,feat_imp=feature_imp_num)


                            if len(set(labels_DEFINITIVA_P))>1:
                                distribucion_etiquetas = [len(XtestP[np.array(labels_DEFINITIVA_P) == c]) for c in  range(numero_clases)]
                                ratio_distribucion_etiquetas = np.max(distribucion_etiquetas) / np.min(distribucion_etiquetas)

                                cent_tr,cent_tst=GNG_Funciones.obtencion_ceniodes(numero_clases, DIMENSION, mytrain, my_y_train, XtestP, np.array(labels_DEFINITIVA_P))
                                distancia_train_test = GNG_Funciones.calculo_distancia_centroides(cent_tr,cent_tst, numero_clases)
                                flag_movimiento = GNG_Funciones.aumenta_distancia(cent_tr, cent_tst,ref_radios)#ref_radios diametro  #ESTA MARIA
                            else:
                                distribucion_etiquetas = [len(XtestP[np.array(labels_DEFINITIVA_P) == c]) for c in range(numero_clases)]
                                ratio_distribucion_etiquetas=np.max(distribucion_etiquetas)


                            dim_comparacion_prop=0
                            if numero_clusters != numero_clases_ORIGINAL:# numero_clusters > numero_clases_ORIGINAL:
                                if np.array(lista_proporciones[-2]).shape[0]!=np.array(lista_proporciones[-1]).shape[0]:
                                    dim_comparacion_prop=min(np.array(lista_proporciones[-2]).shape[0],np.array(lista_proporciones[-1]).shape[0])
                                elif np.array(lista_proporciones[-1]).shape[0]!=np.array(lista_proporciones[0]).shape[0]:
                                    dim_comparacion_prop = min(np.array(lista_proporciones[-1]).shape[0],np.array(lista_proporciones[0]).shape[0])

                            if ((dim_comparacion_prop>=numero_clases_ORIGINAL)&(dim_comparacion_prop!=0)):
                                dim_comparacion_prop = dim_comparacion_prop - 1#Comenzamos en cero
                            print('DIMENSION COMPARACION',dim_comparacion_prop)

                            if ((numero_clusters > numero_clases_ORIGINAL)and(cambio_clusters==1)and((np.array(lista_proporciones[-1])[1]<np.array(lista_proporciones[-2])[1]/factor_proporcionalidad_batch_clusters).any()))or\
                                    ((numero_clusters > numero_clases_ORIGINAL)and(cambio_clusters==0)and((np.array(lista_proporciones)[-1][dim_comparacion_prop]<np.array(lista_proporciones[-2])[dim_comparacion_prop]/factor_proporcionalidad_batch_clusters).any()))or\
                                    ((numero_clusters > numero_clases_ORIGINAL) and (cambio_clusters== 0) and ((np.abs(np.array(lista_proporciones)[-1][dim_comparacion_prop] - np.array(lista_proporciones)[0][dim_comparacion_prop])>cantidad_proporcionalidad_batch_clusters).any()))or\
                                    ((numero_clusters == numero_clases_ORIGINAL)and(ratio_distribucion_etiquetas>ratio_distribucion_INICIAL_etiquetas*3))or\
                                    ((numero_clusters == numero_clases_ORIGINAL)and((distancia_train_test>np.array(distancia_train_test_OROGINARIA)*9).any())):
                                if numero_clusters != numero_clases_ORIGINAL:# numero_clusters > numero_clases_ORIGINAL:
                                    limit = int((XtestP.shape[0] // 4) ** 0.5)
                                    scores = []
                                    scores.append(0.0)
                                    scores.append(0.1)
                                    for k in range(2, limit + 1):
                                        model = KMeans(n_clusters=k)
                                        model.fit(XtestP)
                                        pred = model.predict(XtestP)
                                        score = silhouette_score(XtestP, pred)
                                        scores.append(score)

                                    second_element = sorted(scores)[-2]
                                    index_second = np.where(np.array(scores) == second_element)[0][0]
                                    diferencia_clusteresNEW = scores[np.argmax(scores)] - second_element

                                    if np.argmax(scores) <= numero_clusters:
                                        numero_clusterstmp = np.argmax(scores)
                                    else:#
                                        segundo_maximo = np.sort(scores)[-2]
                                        indice_segundo = np.where(scores == segundo_maximo)[0][0]
                                        maximo = np.sort(scores)[-1]
                                        if (maximo - segundo_maximo) > SILHOUTE_TOLERANCE:
                                            numero_clusterstmp = np.argmax(scores)
                                        else:
                                            numero_clusterstmp = indice_segundo
                                else:
                                    numero_clusterstmp=numero_clases

                                numero_clasestmp=numero_clusterstmp
                                numero_clases=numero_clasestmp
                                kmeanstest = KMeans(n_clusters=numero_clases, random_state=0)
                                kmeanstest.fit(XtestP)
                                centroidestest = kmeanstest.cluster_centers_

                                _asociacion_clusters = kmeanstest.fit(XtestP).labels_
                                new_lab = kmeanstest.predict(centroidestest)
                                knn = KNeighborsClassifier(n_neighbors=1).fit(centroidestest, new_lab)
                                prediccion_knn = knn.predict(XtestP).astype(int)

                                permutations = GNG_Funciones.get_permutations(centroidestest)
                                for perm in permutations:
                                    centroidestesttmp = np.array(perm)
                                    kmeanstest = KMeans(n_clusters=numero_clases,init=centroidestesttmp,random_state=0)
                                    kmeanstest.fit(XtestP)

                                    _asociacion_clusters = kmeanstest.fit(XtestP).labels_
                                    new_lab = kmeanstest.predict(centroidestest)
                                    knn = KNeighborsClassifier(n_neighbors=1).fit(centroidestest, new_lab)
                                    prediccion_knn = knn.predict(XtestP).astype(int)

                                    size_save = len(lista_gng_guardados)
                                    for tam in range(size_save):
                                        if np.array(lista_numero_clusters_modelos_guardados)[tam] == numero_clases:
                                            distancia_comparativa = GNG_Funciones.calculo_distancia_centroides(np.array(lista_centroides_modelos_guardados[tam])[:,:DIMENSION].tolist(),centroidestesttmp.tolist(), numero_clases)
                                            #print(tam,distancia_comparativa,diametro+TOLERANCE_IOU)
                                            if (np.array(distancia_comparativa)<(diametro+TOLERANCE_IOU)).all():
                                                valores2 = GNG_Funciones.obtain_alpha_IOU_2025(contador, np.array(lista_numero_clusters_modelos_guardados)[tam],lista_guardados[tam], XtestP, prediccion_knn,including_percentage[i_name],dir_datset)
                                                list_iou.append([contador,np.array(valores2)])
                                                print(valores2,contour_percentage[i_name])
                                                if (np.array(valores2) >=(contour_percentage[i_name])).all():
                                                    print('CAMBIO',valores2)
                                                    SILHOUTE_TOLERANCE = 0.12
                                                    flag_cambio = 1
                                                    contador_cambio = contador
                                                    array_nodos_actuales_transformados = GNG_Funciones.flatten(lista_gng_guardados[tam])
                                                    array_nodos_actuales_transformados = np.array(array_nodos_actuales_transformados).reshape(numero_nodos_GNG,DIMENSION)
                                                    etiquetas_actuales_trans = np.array(lista_etiquetas_gng_guardados[tam])
                                                    POSICIONESGNG[contador] = array_nodos_actuales_transformados.tolist()
                                                    labelsGNG[contador] = np.array(etiquetas_actuales_trans)
                                                    labels_DEFINITIVA_P=prediccion_knn
                                                    if numero_clusters!=numero_clases_ORIGINAL:
                                                        lista_asignaciones.pop()
                                                        asignaciong=lista_asignaciones_modelos_guardados[tam]
                                                        lista_asignaciones.append(asignaciong)
                                                        lista_proporciones.pop()
                                                        proporciones_ACTUALES = lista_proporciones_modelos_guardados[tam]
                                                        lista_proporciones.append(proporciones_ACTUALES)
                                                    lista_centroides.pop()
                                                    if DIMENSION==2:
                                                        tmp=np.zeros((numero_clases,3))
                                                        for q in range(numero_clases):
                                                            tmp[q]=np.array([centroidestesttmp[q][0],centroidestesttmp[q][1],0])
                                                        lista_centroides.append(tmp)

                                                        t_max=[]
                                                        for i in range(numero_clases):
                                                            tmp_max = GNG_Funciones.distancias_al_centroide_INTRA_CLUSTER(
                                                                array_nodos_actuales_transformados[etiquetas_actuales_trans==i], tmp[i][:DIMENSION])
                                                            t_max.append(tmp_max)
                                                        lista_distancias_maximas.append(t_max)
                                                    else:
                                                        t_max=[]
                                                        for i in range(numero_clases):
                                                            tmp_max = GNG_Funciones.distancias_al_centroide_INTRA_CLUSTER(
                                                                lista_gng_guardados[tam][i], centroidestesttmp[i])
                                                            t_max.append(tmp_max)
                                                        lista_distancias_maximas.append(t_max)
                                                        lista_centroides.append(centroidestesttmp)
                                                lista_numero_clusters.pop()
                                                lista_numero_clusters.append(numero_clusterstmp)
                                                numero_clusters=numero_clases
                                                break
                                        if((flag_cambio==1) &(np.array(distancia_comparativa)<(diametro+TOLERANCE_IOU)).all()):
                                            break
                                    if (flag_cambio == 1):
                                        break

                                if  ((tam== size_save-1) and(flag_cambio== 0)):
                                    numero_clases=numero_clusters

                            if flag_EZ!=0:# numero_clusters > numero_clases_ORIGINAL:
                                for j in range(XtestP.shape[0]):
                                    labels_DEFINITIVA_P[j] = int(lista_asignaciones[-1][labels_DEFINITIVA_P[j]][1])

                            flag_wilcoxon_list = []
                            tmp = np.full((numero_clases_ORIGINAL, DIMENSION), -1.1)
                            if len(set(labels_DEFINITIVA_P)) <= 1:
                                for p in range(numero_clases_ORIGINAL):
                                    for f in range(DIMENSION):
                                        tmp[p][f] = 1
                                        flag_wilcoxon = 1
                                        flag_wilcoxon_list.append(flag_wilcoxon)
                                array_pvalor.append(tmp)
                            elif ratio_distribucion_etiquetas > ratio_distribucion_INICIAL_etiquetas * 3:
                                for p in range(numero_clases_ORIGINAL):
                                    for f in range(DIMENSION):
                                        tmp[p][f] = 1
                                        flag_wilcoxon = 1
                                        flag_wilcoxon_list.append(flag_wilcoxon)
                                array_pvalor.append(tmp)
                            else:
                                for p in range(numero_clases_ORIGINAL):
                                    for f in range(DIMENSION):
                                        flag_wilcoxon, pvalorc = GNG_Funciones.Wilcoxon_flag_NEW(
                                            mytrain[np.array(my_y_train) == p][:, f],
                                            XtestP[np.array(labels_DEFINITIVA_P) == p][:, f])
                                        tmp[p][f] = pvalorc
                                        flag_wilcoxon_list.append(flag_wilcoxon)
                                array_pvalor.append(tmp)

                            F1[contador] = (f1_score(ytestP, labels_DEFINITIVA_P, average='macro'))

                            minimos_abosulos = np.zeros((DIMENSION))
                            maximos_absolutos = np.zeros((DIMENSION))
                            if identificador_zona != 9:
                                minimos_abosulos = np.min(np.array(lista_centroides[contador]),axis=0)
                                maximos_absolutos = np.max(np.array(lista_centroides[contador]),axis=0)

                                ZONA_LABELIZADA = [minimos_abosulos[feature_imp_num], maximos_absolutos[feature_imp_num]]
                            else:
                                if DIMENSION == 2:
                                    for fea in range(DIMENSION):
                                        minimos_abosulos[fea] = min(min(np.array(lista_centroides[contador])[fea][0:DIMENSION]),
                                                                    min(np.array(lista_centroides[contador])[fea][0:DIMENSION]))
                                        maximos_absolutos[fea] = max(max(np.array(lista_centroides[contador])[fea][0:DIMENSION]),
                                                                     max(np.array(lista_centroides[contador])[fea][0:DIMENSION]))

                                        ZONA_LABELIZADA[fea] = [minimos_abosulos[fea], maximos_absolutos[fea]]
                                else:
                                    for fea in range(DIMENSION):
                                        minimos_abosulos[fea] = min(min(np.array(lista_centroides[contador])[fea]),
                                                                    min(np.array(lista_centroides[contador])[fea]))
                                        maximos_absolutos[fea] = max(max(np.array(lista_centroides[contador])[fea]),
                                                                     max(np.array(lista_centroides[contador])[fea]))

                                        ZONA_LABELIZADA[fea] = [minimos_abosulos[fea], maximos_absolutos[fea]]

                            lista_etiquetas_test.append(labels_DEFINITIVA_P)
                            comparativaP = ytestP

                            resDEF = np.abs(comparativaP - labels_DEFINITIVA_P)
                            errorDEFP = len(np.where(resDEF != 0)[0]) / len(resDEF)
                            totalDEFP = 100 * (1 - errorDEFP)

                            array_resultados[contador] = (1 - errorDEFP)
                            array_error[contador] = errorDEFP

                            acumulado = GNG_Funciones.calcularAcumuladonew(contador, array_resultados, array_resultados[0])#25 ,vol_batch)
                            array_resultados_acumulados[contador] =acumulado
                            array_error_acumulado[contador]  = GNG_Funciones.calcular_ERROR_Acumuladonew(contador, array_error, array_error[0])#25 ,vol_batch)

                            print('F1 ',F1[contador])
                            print(contador,'  RESULTADOS:  ',totalDEFP)

                            # F1 y Youden usan las ETIQUETAS (preds_test)
                            F = metrics.micro_f1(comparativaP, labels_DEFINITIVA_P, kkc_fitP)
                            youd = metrics.youden_index_evl(comparativaP, labels_DEFINITIVA_P, kkc_fitP)
                            Youden[contador]=youd
                            micro_F1_osr[contador]=F
                            print(f"microF1_OSR: {F},  youden: {youd}")

                            distancia_comparativa = GNG_Funciones.calculo_distancia_centroides(np.array(lista_centroides_modelos_guardados)[-1].tolist(),centroides_transformados, numero_clases)
                            if (np.array(distancia_comparativa)>=distacia_guardar_modelos).any():
                                inicio=0
                                seleccion_TMP=[]
                                todos_TMP = []
                                for g in range(numero_clases):
                                    lista_tmp = GNG_Funciones.flatten(POSICIONESGNG[contador][inicio:inicio + proporciones_ACTUALES[g]])
                                    tam_tmp = np.array(POSICIONESGNG[contador][inicio:inicio + proporciones_ACTUALES[g]]).shape[0]
                                    array_tmp = np.array(lista_tmp).reshape(tam_tmp, DIMENSION)
                                    out= GNG_Funciones.SeleccionBy1Class(array_tmp, np.array(centroides_transformados)[g,:DIMENSION],including_percentage[i_name])
                                    seleccion_TMP.append(out)
                                    todos_TMP.append(array_tmp)
                                    inicio += proporciones_ACTUALES[g]
                                    if g==0:
                                        lista_gng_guardados.append(POSICIONESGNG[contador].tolist())
                                        lista_etiquetas_gng_guardados.append(labelsGNG[contador])#.append(labelsGNG[contador].tolist())
                                        lista_centroides_modelos_guardados.append(np.array(centroides_transformados))
                                        lista_numero_clusters_modelos_guardados.append(numero_clases)
                                        lista_proporciones_modelos_guardados.append(proporciones_ACTUALES)
                                        if numero_clases != numero_clases_ORIGINAL:
                                            lista_asignaciones_modelos_guardados.append(asignacion)
                                            lista_proporciones_clusters_guardados.append(lista_proporciones[-1])#LISTA_PROPORCINES_CLUSTERS
                                    elif g==numero_clases-1:
                                        lista_guardados.append(todos_TMP)
                                        lista_seleccion_guardados.append(seleccion_TMP)
                    if DIMENSION==2:
                        if contador % 5 == 0:
                            if flag_situacion == 1:
                                if contador==0:
                                    GNG_Funciones.draw_image(lista_centroides[0],gng_slayer.graph, array_posicionestmp, array_posicionestmp,Array_etiquetastmp, numero_clases,DIMENSION, contador,Xtest,labels_DEFINITIVA,identificador_zona,ZONA_LABELIZADA,feature_imp_num, dir_datset ,nuevo_nombre)
                                else:
                                    GNG_Funciones.draw_image(lista_centroides[contador],gng_slayer.graph, array_nodos_actuales_ordenados, array_nodos_actuales_transformados, etiquetas_actuales_trans, numero_clases,DIMENSION, contador, Xtest, labels_DEFINITIVA,identificador_zona,ZONA_LABELIZADA,feature_imp_num,dir_datset ,nuevo_nombre)
                            else:
                                GNG_Funciones.draw_image(lista_centroides[contador],gng_slayer.graph, array_nodos_actuales_ordenados,array_nodos_actuales_transformados, etiquetas_actuales_trans, numero_clases,DIMENSION, contador,XtestP,labels_DEFINITIVA_P,identificador_zona,ZONA_LABELIZADA,feature_imp_num,dir_datset ,nuevo_nombre)

            while -1.1 in F1:
                F1 = np.delete(F1, -1)
                micro_F1_osr = np.delete(micro_F1_osr, -1)
                Youden = np.delete(Youden, -1)
                AUROC = np.delete(AUROC, -1)
                array_resultados= np.delete(array_resultados, -1)
                array_error= np.delete(array_error, -1)
                array_error_acumulado= np.delete(array_error_acumulado, -1)
                array_resultados_acumulados= np.delete(array_resultados_acumulados, -1)
                array_flag_densidad = np.delete(array_flag_densidad, -1)
                array_distancia_centroides = np.delete(array_distancia_centroides, -1)

            if dim_max > dim_max_comparacion:
                ratio = (round)(dim_max / dim_max_comparacion)
                for j in range(dim_max_comparacion):
                    if (ratio * j + ratio) <= dim_max:
                        tmp_F1 = 0
                        tmp_array_resultados = 0
                        tmp_array_error = 0
                        tmp_you = 0
                        tmp_micro = 0
                        tmp_auc = 0
                        for resto in range(ratio):
                            tmp_F1 += F1[ratio * j + resto]
                            tmp_you += Youden[ratio * j + resto]
                            tmp_micro += micro_F1_osr[ratio * j + resto]
                            tmp_auc += AUROC[ratio * j + resto]
                            tmp_array_resultados += array_resultados[ratio * j + resto]
                            tmp_array_error += array_error[ratio * j + resto]
                        F1_COMP[j] = tmp_F1 / ratio
                        AUROC_COMP[j] = tmp_auc / ratio
                        Youden_COMP[j] = tmp_you / ratio
                        micro_F1_osr_COMP[j] = tmp_micro / ratio
                        array_resultados_COMP[j] = tmp_array_resultados / ratio
                        array_error_COMP[j] = tmp_array_error / ratio
                        array_error_acumulado_COMP[j] = GNG_Funciones.calcular_ERROR_Acumuladonew(j, array_error_COMP,array_error_COMP[0])
                        array_resultados_acumulados_COMP[j] = GNG_Funciones.calcularAcumuladonew(j, array_resultados_COMP,array_resultados_COMP[0])
                    else:
                        F1_COMP = np.delete(F1_COMP, -1)
                        micro_F1_osr_COMP = np.delete(micro_F1_osr, -1)
                        Youden_COMP = np.delete(Youden, -1)
                        AUROC_COMP = np.delete(AUROC, -1)
                        array_resultados_COMP = np.delete(array_resultados_COMP, -1)
                        array_error_COMP = np.delete(array_error_COMP, -1)
                        array_error_acumulado_COMP = np.delete(array_error_acumulado_COMP, -1)
                        array_resultados_acumulados_COMP = np.delete(array_resultados_acumulados_COMP, -1)
            else:
                F1_COMP = F1
                micro_F1_osr_COMP = micro_F1_osr
                Youden_COMP = Youden
                AUROC_COMP = AUROC
                array_resultados_COMP = array_resultados
                array_error_COMP = array_error
                array_error_acumulado_COMP = array_error_acumulado
                array_resultados_acumulados_COMP = array_resultados_acumulados

            array_resultados_boxplot.append((array_error_acumulado[-1]*100))   
            
            pred_df_name=path + '/'+ nuevo_nombre + '_predicciones.res'
            tam=len(lista_etiquetas_test)
            total=NUM_instancias_poriteracion+vol_batch*tam
            df.loc[NUM_instancias_poriteracion:total-1, 'PREDICTION'] =np.array(lista_etiquetas_test).flatten().tolist()
            df.to_csv(pred_df_name,index=False) 

            tem = sys.stdout
            sys.stdout = fA = open(path+'/' + nuevo_nombre + 'RESULTADOS.res', 'w')
            GNG_Funciones.salvar_resultadosEVL('GNG_KDE', 1, array_resultados_COMP.tolist(),
                                               array_resultados_acumulados_COMP.tolist(), F1_COMP, micro_F1_osr_COMP,
                                               Youden_COMP, AUROC_COMP)
            GNG_Funciones.salvar_resultadosEVL('GNG_KDE', 2, array_error_COMP.tolist(),
                                               array_error_acumulado_COMP.tolist())
            sys.stdout = tem
            fA.close()

            sys.stdout = fA = open(path+'/'+ nuevo_nombre + 'parametros.res', 'w')
            GNG_Funciones.salvar_parametros(array_error_acumulado_COMP[-1], including_percentage[i_name],  dim_max, metrica_param[i_name],numero_reps[i_name],dist_save_mod[i_name],contour_percentage[i_name],tolerance_iou[i_name],array_resultados_boxplot,list_iou)
            sys.stdout = tem
            fA.close()

            fig1 = plt.figure()
            fig1.add_subplot(111)
            plt.title(nuevo_nombre)
            plt.ylabel('Accuracy')
            plt.xlabel('Timestep')
            plt.plot(array_resultados[0:dim_max - 1], color='b', label='ACCURACY')
            plt.legend()
            fig1.savefig(path + '/' + nuevo_nombre + '_ACURACY' + '.png')
            plt.show(block=False)
            plt.close('all')

            fig2 = plt.figure()
            fig2.add_subplot(111)
            plt.title(nuevo_nombre)
            plt.ylabel('Micro_F1_OSR')
            plt.xlabel('Timestep')
            plt.plot(micro_F1_osr[0:dim_max - 1], color='b', label='Micro F1 OSR')
            plt.legend()
            fig2.savefig(path + '/' + nuevo_nombre + '_Micro_F1' + '.png')
            plt.show(block=False)
            plt.close('all')

            fig3 = plt.figure()
            fig3.add_subplot(111)
            plt.title(nuevo_nombre)
            plt.ylabel('YOUDEN OSR')
            plt.xlabel('Timestep')
            plt.plot(Youden[0:dim_max - 1], color='b', label='YOUDEN OSR')
            plt.legend()
            fig3.savefig(path + '/' + nuevo_nombre + '_YOUDEN' + '.png')
            plt.show(block=False)
            plt.close('all')


        fig11 = plt.figure()
        fig11.add_subplot(111)
        plt.title(file[:-4])
        plt.ylabel('Prequential error')
        plt.xlabel(path+file[:-4])
        plt.boxplot(np.array(array_resultados_boxplot))
        fig11.savefig(path+'/'+file[:-4] + '_boxplot' + '.png')
        plt.show(block=False)
        plt.close('all')

        if len(list_iou)>=1:
            fig12 = plt.figure(figsize=(15, 15))
            numero_total_clusters = np.array(lista_numero_clusters_modelos_guardados).max()
            for f in range(numero_total_clusters):
                fig12.add_subplot(numero_total_clusters, 1, f + 1)
                fig12.subplots_adjust(hspace=0.7, wspace=0.133, top=0.979, bottom=0.03, right=0.976, left=0.055)
                valores = [np.round(datos[f],2) if len(datos) > f else 0 for datos in np.array(list_iou)[:, 1]]
                print(valores)
                name = 'IOU metric :' + str(f + 1)
                label_f = 'cluster_' + str(f + 1)
                plt.title(name)
                plt.ylabel('IOU ')
                plt.xlabel('Timestep')
                np.array(list_iou)
                plt.scatter(np.array(list_iou)[:, 0], valores, color=colors_SINGULARES[f], alpha=0.5, label=label_f)
                for i in range(len(valores)):
                    plt.annotate(f'({np.array(list_iou)[:,0][i]}, {str(valores[i])})', (np.array(list_iou)[:,0][i], valores[i]), textcoords="offset points", xytext=(0, 10), ha='center',fontsize=10)
                plt.legend()

            fig12.savefig(path + '/' + file[:-4] + '_IOU' + '.png')
            plt.show(block=False)
            plt.close('all')
        else:
            print('No se detecta ningun CD Abrupto.(Las GNG absorben bien las rotaciones de las clases alrededor de sus centros.)')
        
    print('FIN')


