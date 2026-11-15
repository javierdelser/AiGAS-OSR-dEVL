"""
Estima si un lote abre una clase nueva sin leer sus etiquetas.

La silueta de K frente a K+1 no sirve en todos los flujos: en FG y MG,
donde no hay clase nueva, abre una de todos modos, y en varios flujos
con clase nueva no la abre porque K+1 no mejora la silueta.

La regla de este fichero sigue el centroide y el radio (percentil 85)
de cada clase conocida. Un punto queda fuera si dista mas de
FACTOR_RADIO radios de todos los centroides. Ese grupo abre una clase
solo si los centroides conocidos siguen ocupados y el grupo nuevo no
parte una clase ya explicada. Si un centroide se vacia y los demas
siguen cubriendo el lote, paso() devuelve -1 y quita ese centroide.
Si los puntos se han ido lejos de todos, la clase se ha movido y K
no cambia.

Las etiquetas se usan al arrancar, en el prefijo, y al imprimir
DISCREPANCIA. No modifican K. Este fichero no modifica AIGAS_OSR_2026.py.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


FACTOR_RADIO = 3.2
MASA_MINIMA = 0.02
MASA_ABSOLUTA = 8
FRACCION_ABANDONO = 0.25


def tamano_prefijo(n, dim_total=200, techo=44000):
    """Mismo corte que AIGAS_OSR_2026: 1000 muestras, o el 2.5 por ciento si el flujo es largo."""
    if n > techo:
        pref = int(n * 5 / dim_total)
    else:
        pref = 1000
    vol = max(1, pref // 5)
    pref = max(vol, (pref // vol) * vol)
    return pref, vol


def _radio(pts, centro):
    if len(pts) < 5:
        return 1e-6
    return max(float(np.quantile(np.linalg.norm(pts - centro, axis=1), 0.85)), 1e-6)


class ClassTracker:
    """Centroides y radios de las clases vistas. paso() no recibe etiquetas del lote."""

    def __init__(self, X_prefijo, y_prefijo, factor=FACTOR_RADIO):
        self.factor = factor
        centros, radios, masas = [], [], []
        for clase in np.unique(y_prefijo):
            pts = X_prefijo[y_prefijo == clase]
            cen = pts.mean(axis=0)
            centros.append(cen)
            radios.append(_radio(pts, cen))
            masas.append(float(len(pts)))
        self.centros = np.vstack(centros)
        self.radios = np.asarray(radios, dtype=float)
        self.masas = np.asarray(masas, dtype=float)
        # Las masas salen del prefijo entero. Cada lote es mas corto, y la
        # comparacion de abandono usa esa proporcion.
        self.tam_ref = float(len(X_prefijo))
        self.k_minimo = len(self.centros)

    def _quitar(self, indice):
        mascara = np.ones(len(self.centros), dtype=bool)
        mascara[indice] = False
        self.centros = self.centros[mascara]
        self.radios = self.radios[mascara]
        self.masas = self.masas[mascara]

    def paso(self, X):
        """
        Devuelve (cambio, centroides).
        cambio es +1 si abre una clase, -1 si una clase seguida se vacia,
        y 0 si el lote solo mueve las clases que ya habia.
        X es solo el lote actual, sin etiquetas.
        """
        X = np.asarray(X, dtype=float)
        centros = self.centros
        radios = self.radios
        k = len(centros)
        dist = np.linalg.norm(X[:, None, :] - centros[None, :, :], axis=2)
        cercano = dist.argmin(axis=1)
        dist_cercana = dist[np.arange(len(X)), cercano]
        fuera = dist_cercana > self.factor * radios[cercano]
        masa_min = max(MASA_ABSOLUTA, int(MASA_MINIMA * len(X)))
        masa_dentro = np.array(
            [int(np.sum((cercano == i) & ~fuera)) for i in range(k)]
        )
        # masa_dentro es de este lote. self.masas puede ser del prefijo
        # entero o del lote anterior: se pasa a la misma escala.
        masa_esperada = self.masas * (len(X) / self.tam_ref)
        abandonadas = [
            i for i in range(k)
            if masa_esperada[i] >= MASA_ABSOLUTA
            and masa_dentro[i] < FRACCION_ABANDONO * masa_esperada[i]
        ]

        if int(fuera.sum()) >= masa_min and not abandonadas:
            sel = fuera
            cen = X[sel].mean(axis=0)
            dist_nueva = np.linalg.norm(X - cen, axis=1)
            robados = int(np.sum((dist_nueva < dist_cercana) & ~fuera))
            if robados <= int(fuera.sum()):
                self.centros = np.vstack([centros, cen])
                self.radios = np.append(radios, _radio(X[sel], cen))
                self.masas = np.append(masa_dentro, float(fuera.sum()))
                self.tam_ref = float(len(X))
                for i in range(k):
                    quedan = (cercano == i) & ~fuera
                    if int(quedan.sum()) >= 5:
                        self.centros[i] = X[quedan].mean(axis=0)
                        self.radios[i] = _radio(X[quedan], self.centros[i])
                return 1, self.centros

        if abandonadas and len(self.centros) > self.k_minimo:
            # Una clase se ha ido si el resto de centroides sigue cubriendo
            # el lote. Si los puntos quedan lejos de todos, la clase se ha
            # movido y K no cambia.
            for indice in sorted(abandonadas, key=lambda i: masa_dentro[i]):
                vivos = [j for j in range(k) if j != indice]
                dist_vivos = dist[:, vivos]
                cercano_vivo = dist_vivos.argmin(axis=1)
                dist_viva = dist_vivos[np.arange(len(X)), cercano_vivo]
                radio_vivo = radios[vivos][cercano_vivo]
                sin_cubrir = int(np.sum(dist_viva > self.factor * radio_vivo))
                if sin_cubrir < masa_min:
                    self._quitar(indice)
                    for posicion, j in enumerate(vivos):
                        sel = cercano_vivo == posicion
                        if int(sel.sum()) >= 5:
                            self.centros[posicion] = X[sel].mean(axis=0)
                            self.radios[posicion] = _radio(X[sel], self.centros[posicion])
                            self.masas[posicion] = float(sel.sum())
                    self.tam_ref = float(len(X))
                    return -1, self.centros
            self.centros, self.radios, self.masas = _recolocar(X, centros)
            self.tam_ref = float(len(X))
            return 0, self.centros
        if abandonadas:
            self.centros, self.radios, self.masas = _recolocar(X, centros)
            self.tam_ref = float(len(X))
            return 0, self.centros

        self.masas = np.array([float(np.sum(cercano == i)) for i in range(k)])
        self.tam_ref = float(len(X))
        for i in range(k):
            sel = cercano == i
            if int(sel.sum()) >= 5:
                self.centros[i] = X[sel].mean(axis=0)
                self.radios[i] = _radio(X[sel], self.centros[i])
        return 0, self.centros


def _recolocar(X, centros):
    """Empareja K modos del lote con los centroides anteriores. K no crece."""
    k = len(centros)
    modelo = KMeans(n_clusters=k, n_init=4, random_state=0).fit(X)
    nuevos = modelo.cluster_centers_
    dist = np.linalg.norm(centros[:, None, :] - nuevos[None, :, :], axis=2)
    usados = set()
    asignacion = np.full(k, -1)
    for i in np.argsort(dist.min(axis=1)):
        opciones = [(dist[i, j], j) for j in range(k) if j not in usados]
        _, j = min(opciones)
        asignacion[i] = j
        usados.add(j)
    centros_out = centros.copy()
    radios = np.ones(k)
    masas = np.zeros(k)
    for i in range(k):
        sel = modelo.labels_ == asignacion[i]
        masas[i] = float(sel.sum())
        if int(sel.sum()) >= 5:
            centros_out[i] = X[sel].mean(axis=0)
            radios[i] = _radio(X[sel], centros_out[i])
    return centros_out, radios, masas


def evaluar_fichero(path, max_detalle=8):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].to_numpy(dtype=float)
    y = df.iloc[:, -1].to_numpy()
    n = len(y)
    pref, vol = tamano_prefijo(n)
    if pref >= n - vol:
        print(f"{os.path.basename(path)}: flujo demasiado corto.")
        return

    rastreador = ClassTracker(X[:pref], y[:pref])
    k_modelo = len(rastreador.centros)
    k_prefijo = k_modelo
    discrepancias = 0
    mostradas = 0
    entrada_real = salida_real = None
    entrada_geom = salida_geom = None
    k_antes = k_prefijo
    inicio = pref
    batch = 0
    while inicio + vol <= n:
        fin = inicio + vol
        k_real = len(np.unique(y[inicio:fin]))
        if entrada_real is None and k_real > k_prefijo:
            entrada_real = batch
        if salida_real is None and entrada_real is not None and k_real < k_antes:
            salida_real = batch
        k_antes = k_real
        n_new, _ = rastreador.paso(X[inicio:fin])
        k_modelo = k_modelo + n_new
        if entrada_geom is None and n_new > 0:
            entrada_geom = batch
        if salida_geom is None and n_new < 0:
            salida_geom = batch
        if k_modelo != k_real:
            discrepancias += 1
            if mostradas < max_detalle:
                print(
                    f"DISCREPANCIA {os.path.basename(path)}  "
                    f"lote={batch}  K_geom={k_modelo}  K_lote={k_real}  "
                    f"cambio={n_new}"
                )
                mostradas += 1
        inicio = fin
        batch += 1

    if discrepancias > mostradas:
        print(
            f"  ... {discrepancias - mostradas} discrepancias mas en "
            f"{os.path.basename(path)} (no listadas)."
        )
    print(
        f"RESUMEN {os.path.basename(path)}  lotes={batch}  discrepancias={discrepancias}  "
        f"entrada_real={entrada_real} entrada_geom={entrada_geom}  "
        f"salida_real={salida_real} salida_geom={salida_geom}"
    )


def main():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets_OSR")
    ficheros = sorted(
        f for f in os.listdir(base)
        if f.endswith(".txt") and not f.endswith(".old") and f != "dim_UG.txt"
        and ("OSR" in f or f in ("FG_2C_2D_MOD.txt", "MG_2C_2D_MOD.txt"))
    )
    print(f"Ficheros: {len(ficheros)}")
    print(
        "K_real usa las etiquetas solo para la comparacion. "
        "K_geom sale de los puntos que caen a mas de "
        f"{FACTOR_RADIO} radios de las clases seguidas."
    )
    for nombre in ficheros:
        evaluar_fichero(os.path.join(base, nombre))


RastreadorClases = ClassTracker

if __name__ == "__main__":
    main()
