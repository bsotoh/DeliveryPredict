"""Contrato de inferencia y datos históricos de DeliveryPredict."""

import json
from hashlib import sha256
from itertools import product
from dataclasses import dataclass
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

NOMBRES_CLASE = ("Muy rápida", "Normal", "Lenta", "Muy lenta")
INTERVALOS = ("Hasta 20 min", "Más de 20 a 30 min", "Más de 30 a 40 min", "Más de 40 min")
CONTEXTO_SOPORTE = ("City", "Road_traffic_density", "Festival", "Vehicle_condition")
VEHICULO = "Type_of_vehicle"
CARGA = "multiple_deliveries"
CAMPOS_SOPORTE = (*CONTEXTO_SOPORTE, VEHICULO, CARGA)
UMBRAL_INICIAL = 30


class ErrorSimulador(ValueError):
    """Error con un mensaje apropiado para el operador."""


@dataclass
class Simulador:
    modelo: object
    meta: dict
    soporte: pd.DataFrame
    demo: dict
    procedencia: dict

    @property
    def referencia(self):
        return self.meta["pedido_referencia"].iloc[0].to_dict()

    @property
    def codigos_carga(self):
        return tuple(sorted(self.soporte.index.get_level_values(CARGA).unique().tolist()))

    def marco(self, pedidos):
        columnas = self.meta["columnas"]
        for pedido in pedidos:
            if set(pedido) != set(columnas):
                raise ErrorSimulador("El escenario debe incluir las 14 variables del modelo.")
            for col, opciones in self.meta["opciones_categoricas"].items():
                if pedido[col] not in opciones:
                    raise ErrorSimulador(f"Categoría fuera del entrenamiento en {col}.")
            for col in self.meta["numericas"]:
                if not np.isfinite(float(pedido[col])):
                    raise ErrorSimulador(f"El valor numérico de {col} debe ser finito.")
            if pedido[CARGA] not in self.codigos_carga:
                raise ErrorSimulador("El código de carga no aparece en los registros de entrenamiento.")
        marco = pd.DataFrame(pedidos, columns=columnas)
        return marco.astype(self.meta["pedido_referencia"].dtypes)

    def probabilidades(self, pedidos):
        matriz = self.modelo.predict_proba(self.marco(pedidos))
        if (
            matriz.shape != (len(pedidos), len(NOMBRES_CLASE))
            or not np.isfinite(matriz).all()
            or (matriz < 0).any()
            or (matriz > 1).any()
            or not np.allclose(matriz.sum(axis=1), 1, rtol=0, atol=1e-8)
        ):
            raise ErrorSimulador("El modelo devolvió probabilidades inválidas.")
        return matriz

    def registros(self, pedido):
        clave = tuple(pedido[col] for col in CAMPOS_SOPORTE)
        return int(self.soporte["registros"].get(clave, 0))

    def fuera_de_rango(self, pedido):
        return [
            col for col, limites in self.meta["rangos_numericos"].items()
            if col != CARGA and not limites["minimo"] <= pedido[col] <= limites["maximo"]
        ]

    def candidatos(self, pedido, vehiculos, probabil_actual, minimo, factibles, datos_confirmados):
        """Evalúa cada par elegido y conserva los ceros como diagnóstico exploratorio."""
        pares = list(product(dict.fromkeys([pedido[VEHICULO], *vehiculos]), self.codigos_carga))
        pedidos = [
            {**pedido, VEHICULO: vehiculo, CARGA: carga}
            for vehiculo, carga in pares
        ]
        opciones = pd.DataFrame(pares, columns=[VEHICULO, CARGA])
        opciones["registros"] = [self.registros(escenario) for escenario in pedidos]
        opciones["probabilidad"] = self.probabilidades(pedidos)[:, 3]
        opciones["actual"] = [par == (pedido[VEHICULO], pedido[CARGA]) for par in pares]
        opciones.loc[opciones["actual"], "probabilidad"] = probabil_actual
        opciones["delta"] = 100 * (opciones["probabilidad"] - probabil_actual)
        opciones["factible"] = [par in factibles for par in pares]
        opciones["evaluable"] = (
            (self.registros(pedido) >= minimo)
            & (not self.fuera_de_rango(pedido))
            & datos_confirmados
            & opciones["registros"].ge(minimo)
            & opciones["factible"]
            & ~opciones["actual"]
        )
        return opciones.sort_values(
            ["actual", "evaluable", "probabilidad", "registros"],
            ascending=[False, False, True, False], ignore_index=True,
        )


def cargar_simulador(carpeta: Path) -> Simulador:
    archivos = (
        "mlp_optimizada.joblib", "metadatos_simulador.joblib",
        "soporte_operativo.csv", "pedido_demostracion.json", "procedencia_soporte.json",
    )
    faltantes = [nombre for nombre in archivos if not (carpeta / nombre).is_file()]
    if faltantes:
        raise ErrorSimulador(f"Faltan archivos en Hito_2: {', '.join(faltantes)}.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            modelo = joblib.load(carpeta / archivos[0])
            meta = joblib.load(carpeta / archivos[1])
    except InconsistentVersionWarning as error:
        raise ErrorSimulador(
            f"El modelo requiere scikit-learn {error.original_sklearn_version}. "
            "Instala las versiones de requirements.txt en el entorno ML."
        ) from error
    except (ImportError, AttributeError, TypeError, ValueError) as error:
        raise ErrorSimulador(
            "Los archivos joblib no pudieron cargarse. Comprueba las versiones "
            "de requirements.txt y conserva los archivos originales."
        ) from error

    pre = modelo.named_steps["preprocesador"]
    if (
        list(modelo.classes_) != [0, 1, 2, 3]
        or tuple(meta["categorias_entrega"]) != NOMBRES_CLASE
        or list(pre.feature_names_in_) != meta["columnas"]
        or list(meta["pedido_referencia"].columns) != meta["columnas"]
        or len(meta["columnas"]) != 14
        or len(meta["numericas"]) != 5
        or len(meta["categoricas"]) != 9
    ):
        raise ErrorSimulador("Las clases o las columnas del modelo no coinciden con los metadatos.")
    grupos = {nombre: list(columnas) for nombre, _, columnas in pre.transformers_}
    if any(grupos[nombre] != meta[nombre] for nombre in ("numericas", "categoricas")):
        raise ErrorSimulador("El preprocesamiento no coincide con las variables del simulador.")
    categorias = pre.named_transformers_["categoricas"].named_steps["codificador"].categories_
    for columna, opciones in zip(meta["categoricas"], categorias, strict=True):
        if set(opciones) != set(meta["opciones_categoricas"][columna]):
            raise ErrorSimulador(f"El codificador no coincide con las categorías de {columna}.")

    soporte = pd.read_csv(carpeta / archivos[2])
    procedencia = json.loads((carpeta / archivos[4]).read_text(encoding="utf-8"))
    if (
        list(soporte.columns) != [*CAMPOS_SOPORTE, "registros"]
        or soporte.isna().any().any()
        or soporte.duplicated(list(CAMPOS_SOPORTE)).any()
        or not pd.api.types.is_integer_dtype(soporte["registros"])
        or not soporte["registros"].gt(0).all()
        or len(soporte) != procedencia["grupos"]
        or soporte["registros"].sum() != procedencia["filas_soporte"]
        or any(
            not set(soporte[col]).issubset(meta["opciones_categoricas"][col])
            for col in CAMPOS_SOPORTE if col != CARGA
        )
        or not pd.api.types.is_integer_dtype(soporte[CARGA])
        or not soporte[CARGA].ge(0).all()
        or list(CAMPOS_SOPORTE) != procedencia["campos_soporte"]
        or sha256(soporte.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()
        != procedencia["soporte_csv_sha256"]
    ):
        raise ErrorSimulador("La tabla de respaldo no coincide con su esquema o con el registro de auditoría.")
    demo, = json.loads((carpeta / archivos[3]).read_text(encoding="utf-8"))
    simulador = Simulador(modelo, meta, soporte.set_index(list(CAMPOS_SOPORTE)), demo, procedencia)
    simulador.marco([simulador.referencia, demo])
    return simulador
