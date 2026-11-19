"""Simulador what-if de duración de entrega. Iniciar con Streamlit."""

from html import escape
from math import ceil, floor, isclose
from pathlib import Path

import numpy as np
import streamlit as st

from simulador import (
    CARGA, INTERVALOS, NOMBRES_CLASE, UMBRAL_INICIAL, VEHICULO,
    ErrorSimulador, cargar_simulador,
)

RAIZ = Path(__file__).resolve().parent
ETIQUETAS = {
    "Delivery_person_Age": "Edad del repartidor",
    "Delivery_person_Ratings": "Calificación del repartidor",
    "Weather_conditions": "Clima",
    "Road_traffic_density": "Tráfico",
    "Vehicle_condition": "Condición del vehículo",
    "Type_of_order": "Tipo de pedido",
    VEHICULO: "Tipo de vehículo",
    CARGA: "Código de entregas simultáneas",
    "Festival": "Festividad",
    "City": "Zona urbana",
    "Dia_semana": "Día de la semana",
    "Mes": "Mes",
    "Hora_pedido": "Hora del pedido",
    "Haversine_km": "Distancia Haversine (km)",
}
OPCIONES = {
    "Weather_conditions": {
        "Cloudy": "Nublado", "Fog": "Niebla", "Sandstorms": "Tormenta de arena",
        "Stormy": "Tormenta", "Sunny": "Soleado", "Windy": "Ventoso",
    },
    "Road_traffic_density": {
        "Low": "Bajo", "Medium": "Medio", "High": "Alto", "Jam": "Congestión",
    },
    "Type_of_order": {
        "Buffet": "Buffet", "Drinks": "Bebidas", "Meal": "Comida", "Snack": "Snack",
    },
    VEHICULO: {
        "bicycle": "Bicicleta", "electric_scooter": "Scooter eléctrico",
        "motorcycle": "Motocicleta", "scooter": "Scooter",
    },
    "Festival": {"No": "No", "Yes": "Sí"},
    "City": {"Metropolitian": "Metropolitana", "Semi-Urban": "Semiurbana", "Urban": "Urbana"},
    "Dia_semana": dict(enumerate(
        ("Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo")
    )),
    "Mes": {2: "Febrero", 3: "Marzo", 4: "Abril"},
}
AYUDAS = {
    CARGA: "Código histórico observado de 0 a 3. Úsalo cuando se conozca la carga prevista. Con el código 0, el pedido sigue presente en el escenario.",
    "Vehicle_condition": "Código histórico de 0 a 3. La fuente utilizada no establece su interpretación operativa exacta. Se mantiene fijo en la ventana.",
    "Delivery_person_Age": "Dato del escenario. La ventana operativa mantiene fijo el perfil del repartidor.",
    "Haversine_km": "Distancia geográfica en línea recta sobre la superficie terrestre, expresada en kilómetros.",
    "Mes": "Meses observados en el entrenamiento.",
}
ENTERAS = {CARGA, "Delivery_person_Age", "Hora_pedido"}
REFERENCIA = "Pedido de referencia"
DEMO = "Ejemplo de demostración"
GUARDADO = "Escenario guardado"


def opcion(columna, valor):
    if columna == "Hora_pedido":
        return f"{int(valor):02d}:00"
    if columna in ENTERAS:
        return str(int(valor))
    if columna == "Haversine_km":
        return f"{decimal(valor, 6)} km"
    if columna == "Delivery_person_Ratings":
        return decimal(valor, 2)
    if columna == "Vehicle_condition":
        return f"Código {valor}"
    return str(OPCIONES.get(columna, {}).get(valor, valor))


def decimal(valor, precision=2):
    return f"{valor:.{precision}f}".replace(".", ",")


def porcentaje(valor):
    if 0 < valor < 0.00005:
        return "&lt;0,01 %"
    if 0.99995 < valor < 1:
        return "&gt;99,99 %"
    return f"{decimal(100 * valor)} %"


def puntos(valor):
    if abs(valor) < 0.005:
        return "≈ 0,00 pp" if abs(valor) > 1e-10 else "0,00 pp"
    return f"{'−' if valor < 0 else '+'}{decimal(abs(valor))} pp"


def clave(columna):
    return f"campo_{columna}"


@st.cache_resource
def recursos():
    return cargar_simulador(RAIZ / "Hito_2")


def limpiar_factibilidad():
    for key in list(st.session_state):
        if key in ("vehiculos_a_evaluar", "datos_confirmados") or key.startswith("factible_"):
            del st.session_state[key]


def cargar_pedido(pedido, origen):
    st.session_state.original = dict(pedido)
    st.session_state.origen = origen
    for col, valor in pedido.items():
        st.session_state[clave(col)] = int(valor) if col in ENTERAS else valor
    limpiar_factibilidad()


def guardar_original(columnas):
    st.session_state.original = {col: st.session_state[clave(col)] for col in columnas}
    st.session_state.origen = GUARDADO


def restablecer():
    limpiar_factibilidad()
    for col, valor in st.session_state.original.items():
        st.session_state[clave(col)] = int(valor) if col in ENTERAS else valor


def campo(simulador, columna):
    meta = simulador.meta
    etiqueta = ETIQUETAS[columna]
    argumentos = {"key": clave(columna), "help": AYUDAS.get(columna), "on_change": limpiar_factibilidad}
    if columna == CARGA:
        return st.select_slider(etiqueta, options=list(simulador.codigos_carga), **argumentos)
    if columna in meta["categoricas"]:
        opciones = meta["opciones_categoricas"][columna]
        if columna == "Road_traffic_density":
            opciones = [v for v in ("Low", "Medium", "High", "Jam") if v in opciones]
        return st.selectbox(etiqueta, opciones, format_func=lambda v: opcion(columna, v), **argumentos)

    limites = meta["rangos_numericos"][columna]
    minimo = min(limites["minimo"], st.session_state.original[columna])
    maximo = max(limites["maximo"], st.session_state.original[columna])
    if columna in ENTERAS:
        return st.select_slider(
            etiqueta, options=list(range(ceil(minimo), floor(maximo) + 1)),
            format_func=lambda v: opcion(columna, v), **argumentos,
        )
    return st.number_input(
        etiqueta, min_value=float(minimo), max_value=float(maximo),
        step=0.1, format="%.6f" if columna == "Haversine_km" else "%.2f", **argumentos,
    )


def estado_respaldo(n, minimo):
    if n == 0:
        return "sin", "Sin respaldo"
    if n < minimo:
        return "bajo", "Respaldo insuficiente"
    return "suficiente", "Cumple el mínimo"


def entero(n):
    return f"{n:,}".replace(",", ".")


def ficha_respaldo(n, minimo):
    nivel, etiqueta = estado_respaldo(n, minimo)
    return f'<span class="soporte {nivel}">{etiqueta} · {entero(n)} registros</span>'


def comparador(probabilidades, conteos, minimo, origen, fuera):
    original, actual = probabilidades
    delta = 100 * (actual[3] - original[3])
    estado = "Comparación descriptiva"
    aviso = "La factibilidad de cada opción se confirma en Ventana operativa."
    severidad = "info"
    if min(conteos) == 0:
        estado, severidad = "Exploración · Sin respaldo histórico", "sin"
        aviso = "Al menos un grupo no tiene registros comparables. La diferencia solo describe la salida del modelo."
    elif min(conteos) < minimo:
        estado, severidad = "Exploración · Respaldo insuficiente", "bajo"
        aviso = f"Al menos un grupo tiene menos de {entero(minimo)} registros comparables."
    if fuera:
        if severidad == "info":
            estado, severidad = "Exploración · Fuera del rango habitual", "bajo"
        aviso += " Variables fuera de P01–P99 · " + ", ".join(fuera) + "."

    fichas = []
    for nombre, detalle, probabilidades_escenario, registros in (
        ("Original", origen, original, conteos[0]),
        ("Modificado", "Escenario en edición", actual, conteos[1]),
    ):
        fichas.append(f"""
<div class="scenario"><div class="scenario-label">{nombre}<small>{escape(detalle)}</small></div>
<span class="eyebrow">Clase predicha</span>
<strong class="scenario-class">{NOMBRES_CLASE[int(np.argmax(probabilidades_escenario))]}</strong>
<span class="eyebrow">P(Muy lenta)</span>
<strong class="scenario-risk">{porcentaje(probabilidades_escenario[3])}</strong>
{ficha_respaldo(registros, minimo)}</div>
""")
    st.html(f"""
<section class="comparison" aria-label="Comparación de predicciones">
  <div class="panel-header"><h2>Comparación de escenarios</h2><span class="risk-tag">Muy lenta · &gt; 40 min</span></div>
  <div class="callout {severidad}" role="status"><strong>{estado}</strong><p>{escape(aviso)}</p></div>
  <div class="scenario-grid">
  {''.join(fichas)}
  <div class="scenario scenario-delta"><div class="scenario-label">Variación estimada<small>Modificado − original</small></div>
  <strong class="scenario-risk">{puntos(delta)}</strong>
  <span class="eyebrow">Puntos porcentuales</span></div></div>
  <p class="panel-caption">El respaldo agrupa por zona, tráfico, festividad, condición del vehículo, vehículo y código de carga.
  Mínimo por grupo · <strong>{entero(minimo)}</strong>.</p>
</section>
""")

    filas = []
    for i, (nombre, intervalo, p0, p1) in enumerate(zip(
        NOMBRES_CLASE, INTERVALOS, original, actual, strict=True
    )):
        alerta = " class='critica'" if i == 3 else ""
        filas.append(f"""
<li{alerta}><div class="chart-label"><strong>{nombre}</strong><small>{intervalo}</small></div>
<div class="tracks" aria-hidden="true"><div><b style="width:{100*p0:.7f}%"></b></div>
<div><b style="width:{100*p1:.7f}%"></b></div></div>
<div class="chart-values"><span aria-label="Original {porcentaje(p0)}">{porcentaje(p0)}</span>
<strong aria-label="Modificado {porcentaje(p1)}">{porcentaje(p1)}</strong></div></li>
""")
    st.html(f"""
<section class="prob-card">
<div class="chart-top"><h3>Probabilidades por clase</h3>
<div class="legend"><span>Original</span><span>Modificado</span></div></div>
<ul class="prob-chart" aria-label="Comparación de las cuatro probabilidades">
{''.join(filas)}</ul></section>
""")


def ventana(simulador, pedido, probabilidad, origen):
    st.subheader("Ventana operativa")
    st.caption("Usa el escenario modificado como base y conserva las otras 12 características del pedido.")
    configuracion, resultados = st.columns([1, 2.15], gap="medium")
    motor = pedido[VEHICULO]
    n_actual = simulador.registros(pedido)
    catalogo = simulador.meta["opciones_categoricas"][VEHICULO]
    with configuracion:
        st.markdown("#### Opciones de despacho")
        minimo = st.number_input(
            "Registros mínimos por grupo", min_value=1, step=1, key="minimo_soporte",
            help="Regla prudencial configurable. Alcanzar el umbral da respaldo descriptivo y no demuestra validez estadística ni causalidad.",
        )
        extras = st.multiselect(
            "Otros vehículos a evaluar",
            [v for v in catalogo if v != motor], format_func=lambda v: opcion(VEHICULO, v),
            key="vehiculos_a_evaluar", placeholder="Selecciona vehículos",
            help="Agregar un vehículo permite explorarlo. Confirma por separado los pares vehículo/código que son factibles.",
        )
        st.markdown("**Pares que confirmas como factibles**")
        st.caption("Selecciona los códigos posibles para cada vehículo según la ruta, el pedido, la disponibilidad y la condición indicada.")
        factibles = set()
        for vehiculo in [motor, *extras]:
            codigos = st.multiselect(
                f"Códigos factibles · {opcion(VEHICULO, vehiculo)}",
                [c for c in simulador.codigos_carga if (vehiculo, c) != (motor, pedido[CARGA])],
                key=f"factible_{vehiculo}", placeholder="Selecciona códigos",
            )
            factibles.update((vehiculo, c) for c in codigos)
        datos_confirmados = st.checkbox(
            "Conozco los datos requeridos al planificar el despacho", key="datos_confirmados",
            help="Incluye carga y vehículo del escenario actual, condición del vehículo, contexto y datos del repartidor.",
        )
        st.caption("Los códigos 0 a 3 provienen del dataset. Confirma su interpretación y disponibilidad antes de aplicar cualquier opción.")
        with st.expander("Qué cambia y qué se conserva"):
            st.markdown(
                "**Decisiones evaluadas.** Tipo de vehículo y código de entregas simultáneas.\n\n"
                "**Características fijas.** Condición del vehículo, zona, tráfico, festividad, "
                "clima, tipo de pedido, distancia, fecha y hora, edad y calificación del repartidor."
            )

    with resultados:
        tabla = simulador.candidatos(pedido, extras, probabilidad, minimo, factibles, datos_confirmados)
        mejores = tabla.loc[tabla["evaluable"] & (tabla["delta"] < -1e-10)]
        valores_contexto = " · ".join((
            opcion("City", pedido["City"]),
            opcion("Road_traffic_density", pedido["Road_traffic_density"]),
            "Con festividad" if pedido["Festival"] == "Yes" else "Sin festividad",
            f"Condición (código {pedido['Vehicle_condition']})",
        ))
        st.html(f"""
<section class="baseline">
<div><span class="eyebrow">BASE · ESCENARIO MODIFICADO</span>
<h3>{escape(opcion(VEHICULO, motor))} · Código {int(pedido[CARGA])}</h3>
<p>{escape(valores_contexto)}</p></div>
<div><strong>{porcentaje(probabilidad)}</strong><small>P(Muy lenta)</small></div>
{ficha_respaldo(n_actual, minimo)}
</section>
""")
        if not mejores.empty:
            mejor = mejores.iloc[0]
            st.html(f"""
<section class="recommend">
<span class="eyebrow">RECOMENDACIÓN CONDICIONADA PARA EVALUAR</span>
<h3>{escape(opcion(VEHICULO, mejor[VEHICULO]))} · Código {int(mejor[CARGA])}</h3>
<div class="recommend-metrics"><strong>{porcentaje(mejor["probabilidad"])} <small>P(Muy lenta)</small></strong>
<strong>{puntos(mejor["delta"])} <small>frente a la base</small></strong>
<strong>{entero(mejor["registros"])} <small>registros</small></strong></div>
<p>Es la opción con menor riesgo predicho entre las factibles confirmadas que cumplen el mínimo histórico.</p>
</section>
""")
        else:
            if n_actual < minimo:
                motivo = "La base no cumple el respaldo mínimo. Las diferencias se muestran como exploración."
            elif simulador.fuera_de_rango(pedido):
                motivo = "La base contiene valores numéricos fuera de P01–P99. Las diferencias se muestran como exploración."
            elif not factibles or not datos_confirmados:
                motivo = "Confirma los pares factibles y que dispones de los datos al planificar el despacho."
            elif not tabla["evaluable"].any():
                motivo = "Los pares confirmados no alcanzan el mínimo de registros comparables."
            else:
                motivo = "Las alternativas evaluables tienen una probabilidad igual o superior a la base."
            st.html(f"""
<div class="callout info" role="status"><strong>Sin recomendación operativa</strong><p>{motivo}</p></div>
""")

        filas = []
        for fila in tabla.to_dict("records"):
            if fila["actual"]:
                estado = "Escenario actual"
            elif fila["evaluable"]:
                estado = "Factible y evaluable"
            elif fila["factible"]:
                estado = "Exploración · Par confirmado"
            else:
                estado = "Exploración · Factibilidad pendiente"
            filas.append(f"""
<tr class="{'row-base' if fila['actual'] else ''}">
<th scope="row">{escape(opcion(VEHICULO, fila[VEHICULO]))}<small>Código {int(fila[CARGA])}</small></th>
<td class="oper-prob">{porcentaje(fila["probabilidad"])}</td>
<td>{puntos(fila["delta"])}</td>
<td>{ficha_respaldo(fila["registros"], minimo)}<small>{estado}</small></td>
</tr>
""")
        st.html(f"""
<section class="oper-table"><h3>Alternativas de despacho</h3>
<table aria-label="Probabilidades y respaldo de las alternativas de despacho">
<thead><tr><th scope="col">Vehículo y código</th><th scope="col">P(Muy lenta)</th>
<th scope="col">Variación</th><th scope="col">Respaldo y estado</th></tr></thead>
<tbody>{''.join(filas)}</tbody></table></section>
""")
        st.caption(
            "Cada conteo exige coincidencia en zona, tráfico, festividad, condición del vehículo, tipo "
            "de vehículo y código de carga. Las otras ocho variables intervienen únicamente en la predicción. "
            "El mínimo es una regla descriptiva y la factibilidad depende de tu confirmación."
        )
        if origen == DEMO:
            st.caption("El escenario parte de un ejemplo seleccionado para demostrar la sensibilidad del modelo.")


def metodologia(procedencia):
    st.subheader("Modelo y alcance")
    izquierda, derecha = st.columns(2)
    with izquierda:
        st.markdown("#### Evaluación del Hito 1")
        st.table({
            "Modelo": ["MLP base", "MLP optimizada", "TabPFN base", "TabPFN optimizado"],
            "F1 macro CV": ["0,7138", "0,7237", "0,7228", "0,7228"],
            "F1 macro test": ["0,7085", "0,7224", "0,7283", "0,7364"],
        }, border="horizontal")
        st.markdown(
            "La MLP optimizada detectó **473 de 807** entregas muy lentas en test "
            "(recall de **58,61 %**). TabPFN optimizado obtuvo el mayor F1 macro en test. "
            "La menor latencia de la MLP optimizada permite recalcular la predicción al editar el escenario."
        )
        st.caption(
             "La tabla final del notebook registra 0,07 s para la MLP optimizada y 50,45 s "
            "para TabPFN optimizado sobre las 9.117 observaciones de test. Son tiempos experimentales. "
            "Entrenamiento final: 36.467 registros para la MLP y 9.000 para TabPFN."
        )
    with derecha:
        st.markdown("#### Interpretación operacional")
        st.markdown(
            "- **Duración pronosticada.** Las cuatro clases son intervalos de tiempo. "
            "«Muy lenta» significa más de 40 minutos. Evaluar el cumplimiento requiere conocer el plazo prometido.\n"
            "- **Predicción.** Las probabilidades pueden estar descalibradas. "
            "Los cambios reflejan asociaciones aprendidas y el efecto de una intervención requiere validación independiente.\n"
            "- **Etapa de uso.** Evalúa opciones en la planificación del despacho cuando "
            "la carga prevista, el vehículo y el repartidor se conocen.\n"
            "- **Alcance de los datos.** Los rangos sugeridos son P01–P99 de entrenamiento, "
            "salvo la carga, que usa los códigos discretos observados. Las categorías provienen del codificador ajustado."
        )
        st.write(
            f"El respaldo se reconstruyó con **{entero(procedencia['filas_entrenamiento'])}** "
            f"filas de entrenamiento. Hay **{entero(procedencia['filas_soporte'])}** filas completas "
            f"en los seis campos y **{entero(procedencia['grupos'])}** grupos. "
            f"De ellos, **{entero(procedencia['grupos_menores_de_30'])}** tienen menos de 30 registros."
        )
        st.markdown(
            "[Datos históricos de Zomato en Kaggle]"
            "(https://www.kaggle.com/datasets/saurabhbadole/zomato-delivery-operations-analytics-dataset)"
        )


st.set_page_config(page_title="DeliveryPredict", layout="wide")
estilos = (RAIZ / "estilos.css").read_text(encoding="utf-8")
st.html(f"<style>{estilos}</style>")
st.html("""
<div class="brandbar"><div class="brand"><i aria-hidden="true"></i>Delivery<span>Predict</span></div>
<div class="brand-meta"><span>Simulador de última milla</span><b>Hito 2 · UDD</b></div></div>
<div class="hero">
  <div class="hero-copy"><span class="overline"><i></i> MODELO MLP OPTIMIZADO</span>
  <h1>Simula antes de despachar<span>.</span></h1>
  <p>Compara escenarios de entrega y consulta su respaldo histórico.</p></div>
  <div class="hero-art" aria-hidden="true"><div class="orbit orbit-a"></div><div class="orbit orbit-b"></div>
  <div class="orbit orbit-c"></div><div class="route"><i></i><i></i><i></i></div></div>
</div>
""")
try:
    simulador = recursos()
except (ErrorSimulador, OSError, EOFError, KeyError, TypeError, AttributeError, ValueError) as error:
    mensaje = str(error) if isinstance(error, ErrorSimulador) else (
        "No se pudieron leer los datos del simulador. Extrae nuevamente el ZIP y consulta README.md."
    )
    st.error(mensaje)
    st.stop()

columnas = simulador.meta["columnas"]
if "original" not in st.session_state:
    cargar_pedido(simulador.referencia, REFERENCIA)
if "minimo_soporte" not in st.session_state:
    st.session_state.minimo_soporte = UMBRAL_INICIAL
minimo = st.session_state.minimo_soporte
origen = st.session_state.origen

tab_sim, tab_ventana, tab_modelo = st.tabs(["Simulador", "Ventana operativa", "Modelo y alcance"])

with tab_sim:
    with st.container(key="resumen-movil"):
        resumen_movil = st.empty()
    editor, resultados = st.columns([0.87, 2.03], gap="medium")
    with editor:
        st.subheader("Edita el escenario")
        st.caption(f"Original · {origen}")
        if origen == DEMO:
            st.caption("Ejemplo de alto riesgo seleccionado para la demostración.")
        with st.expander("Decisiones de despacho", expanded=True):
            for col in (VEHICULO, CARGA):
                campo(simulador, col)
            st.caption("El código de carga de 0 a 3 proviene del dataset. Confirma su uso al planificar el despacho.")
        with st.expander("Contexto de la entrega"):
            for col in (
                "Vehicle_condition", "Road_traffic_density", "Weather_conditions", "City",
                "Haversine_km", "Hora_pedido", "Dia_semana", "Mes", "Type_of_order", "Festival",
            ):
                campo(simulador, col)
        with st.expander("Perfil del repartidor"):
            for col in ("Delivery_person_Age", "Delivery_person_Ratings"):
                campo(simulador, col)
        st.button("Fijar como original", on_click=guardar_original, args=(columnas,), width="stretch")
        st.button("Restablecer cambios", on_click=restablecer, width="stretch")
        with st.expander("Cargar un pedido de entrenamiento"):
            st.button(REFERENCIA, on_click=cargar_pedido, width="stretch",
                      args=(simulador.referencia, REFERENCIA))
            st.button(DEMO, on_click=cargar_pedido, width="stretch",
                      args=(simulador.demo, DEMO))
            st.caption("El pedido de referencia es el que se guardó con los metadatos. "
                       "La demostración usa un ejemplo de alto riesgo seleccionado para ilustrar los cambios.")

    original = st.session_state.original.copy()
    actual = {col: st.session_state[clave(col)] for col in columnas}
    try:
        probabilidades = simulador.probabilidades([original, actual])
    except (ErrorSimulador, ValueError, TypeError) as error:
        st.error(f"El modelo no pudo evaluar el escenario. {error}")
        st.stop()

    delta_movil = 100 * (probabilidades[1, 3] - probabilidades[0, 3])
    conteos = simulador.registros(original), simulador.registros(actual)
    fuera = list(dict.fromkeys([*simulador.fuera_de_rango(original), *simulador.fuera_de_rango(actual)]))
    nivel_movil, etiqueta_movil = estado_respaldo(min(conteos), minimo)
    if fuera and nivel_movil == "suficiente":
        nivel_movil, etiqueta_movil = "bajo", "Fuera del rango habitual"
    resumen_movil.html(
        f'<div class="mini-risk {nivel_movil}" aria-live="polite"><span>Muy lenta · Modificado</span>'
        f'<strong>{porcentaje(probabilidades[1, 3])}</strong>'
        f'<small>{puntos(delta_movil)} · {etiqueta_movil}</small></div>'
    )
    with resultados:
        comparador(probabilidades, conteos, minimo, origen, [ETIQUETAS[col] for col in fuera])
        cambios = [col for col in columnas if (
            not isclose(actual[col], original[col], abs_tol=1e-12, rel_tol=0)
            if col in simulador.meta["numericas"] else actual[col] != original[col]
        )]
        if cambios:
            with st.expander(f"Cambios en el escenario ({len(cambios)})", expanded=True):
                for col in cambios:
                    st.markdown(f"**{ETIQUETAS[col]}** · {opcion(col, original[col])} → {opcion(col, actual[col])}")
        else:
            st.caption("Ambos escenarios coinciden. Modifica una variable para ver el cambio en vivo.")
        st.html("""
<p class="note"><strong>Interpretación.</strong> La variación compara predicciones y el efecto de una
intervención requiere validación. La clase «Muy lenta» mide duración. Para evaluar cumplimiento se requiere el plazo prometido.</p>
""")

with tab_ventana:
    try:
        ventana(simulador, actual, probabilidades[1, 3], origen)
    except (ErrorSimulador, ValueError, TypeError):
        st.error("No fue posible evaluar la ventana operativa. Revisa la configuración del pedido.")

with tab_modelo:
    metodologia(simulador.procedencia)

st.html("<footer>DeliveryPredict · Machine Learning · Universidad del Desarrollo</footer>")
