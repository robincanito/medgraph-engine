"""Parseo y chunking canonicos de MedGraph: PDF -> paginas -> estructura -> chunks.

UNA SOLA FUENTE DE VERDAD (Fase 2, tanda 1, 6-sep-2026). Hasta hoy este codigo vivia
TRES veces: `parser_v2.py` (CLI, la version nueva con reconstruccion de tablas del
9-ago), `api/services/ingest.py` (14 copias MAS VIEJAS y un chunker propio) y
`medgraph-engine/parser_v2.py` (OSS, 209 lineas atras). El bug de embeddings de
agosto vivio exactamente en una de esas copias. Doctrina de la casa: "un numero
repetido en dos archivos siempre diverge en silencio".

Quien lo importa:
  - CLI:  `parser_v2.py` y `migrate_chunks.py` son shims que re-exportan de aca.
  - API:  `api/services/ingest.py::parse_and_chunk` = parse_pdf_v2 + normalize_chunks.
  - OSS:  `medgraph-engine/pipeline/parseo.py` es una copia identica (se replica a mano
          antes de publicar; un test compara los dos archivos).

El codigo de las funciones es el de parser_v2.py del 6-sep, movido verbatim por AST.
`fitz` (PyMuPDF) se importa PEREZOSO dentro de las funciones que lo usan: importarlo al
cargar el modulo le costaba ~10 s a cada arranque en frio de Cloud Run (1-sep-2026).
"""
import logging
import os
import re
import unicodedata
from collections import Counter

# Los parametros de chunkeo y los patrones de estructura viven en pipeline/estrategia.py:
# son del DOMINIO, no del codigo. Se re-exportan con los nombres de siempre para no romper a
# quien los importe (parser_v2 y los tests los usan).
from pipeline.estrategia import (  # noqa: E402, F401  (re-export deliberado)
    MAX_PARENT_WORDS,
    MAX_SIZE,
    MIN_SIZE,
    OVERLAP_SIZE,
    PARENT_WINDOW,
    POR_DEFECTO,
    TARGET_SIZE,
    Estrategia,
)
from pipeline.estrategia import PATRONES_MEDICINA as STRUCTURE_PATTERNS  # noqa: E402, F401

# LO QUE NO ES CONTENIDO (13-sep-2026, item 18 del roadmap). Tipos de chunk que el retrieval
# EXCLUYE: compiten por el top-10 con densidad de termino y sin decir nada. Medido el 12-sep sobre
# el gold de 101 consultas: ocupaban 10 de 1.010 puestos, y en "Obstructivo: EPOC / ASMA" el indice
# CIE-10 se llevaba 4 de 10. Es UNA tupla para las dos puntas -el parser la produce, vector.py la
# filtra- porque dos listas divergen y el sintoma seria un tipo nuevo que se marca y no se filtra.
# Bitacora del paquete: `pipeline/` no usa print (los orquestadores configuran el logging; en
# Cloud Run el print no lleva severity ni origen). tests/test_bitacora.py lo frena desde el 13-sep.
log = logging.getLogger(__name__)

NO_CONTENIDO = ("referencias", "indice")

# Una cita bibliografica: "1983;309:45-7", con o sin fasciculo "2018;391(10125):1023" (Lancet numera
# sus fasciculos con CINCO digitos: el test lo cazo con \d{1,4}). El fasciculo
# NO es opcional en la practica: sin el, DeVita -que cita asi- pasaba de 2.322 chunks a 292, y la
# medicion del 10-sep (4.517) no se podia reproducir. Un detector sin su regex anotada es un numero
# que nadie puede volver a obtener.
CITA_BIBLIOGRAFICA = re.compile(
    r"\b(?:1[89]|20)\d{2}\s?;\s?\d{1,4}(?:\s?\(\d{1,5}\))?\s?:\s?[A-Za-z]?\d{1,5}\b")
# >= 8 citas en un chunk de ~281 palabras (~una cada 35) no deja lugar para prosa, y la primera en
# el primer cuarto dice que el chunk ES la lista, no un parrafo que termina en ella. Los dos juntos
# separan las listas PURAS (2.337 chunks) de las MIXTAS (1.795): las mixtas son prosa clinica con
# la bibliografia del capitulo colgada al final -"...riesgo de provocar una fibrilacion
# ventricular..." con 5 citas-, y marcarlas enteras habria borrado ~466 chunks-equivalentes de
# texto real. Esas se arreglan cortando la seccion en su frontera, en el parseo; no aca.
MIN_CITAS_REFERENCIAS = 8
INICIO_MAX_REFERENCIAS = 0.25

# Un codigo de clasificacion tipo CIE-10: "J10.08", "O9A", "E11". El indice CIE-10 del corpus
# (libro `cie10`, 448 chunks) es un MANUAL DE CODIFICACION: reglas en prosa mezcladas con tablas de
# codigos, mediana 9 codigos por chunk. Medido el 13-sep-2026: fuera de ese libro, en 3.000 chunks
# con algun token parecido a un codigo, CERO llegan a 8. El umbral separa la parte enumerativa (la
# que se llevaba 4 de 10 puestos en "Obstructivo: EPOC / ASMA") sin tocar prosa clinica de nadie.
# Es una regla de CONTENIDO y no "todo el libro cie10 es indice" a proposito: no necesita tocar el
# perfil ni el contrato, y vale igual para una tabla ATC o SNOMED que aparezca en otro libro.
# CON PUNTO OBLIGATORIO, y esto se pago con una inspeccion (13-sep-2026): la forma corta "X00" choca
# con medio vocabulario medico -B12 (la vitamina, 9 veces en un chunk de Robbins), P53/P16/P21 (los
# oncogenes en Farreras), C16/C18 (acilcarnitinas en Meneghello), P90 (percentiles del PRUNAPE),
# A10-A16 (serotipos), "C D13" (CD13 partido por el PDF)-. Con ella, 30 chunks de contenido clinico
# real quedaban marcados como indice fuera de `cie10`. La subcategoria con punto (J10.08, C38.1,
# A04.3) no la escribe nadie que no este codificando: los dos indices legitimos que aparecieron
# fuera de cie10 (el manual de notificacion obligatoria del SIAJ, la tabla de la OMS) la usan.
CODIGO_CLASIFICACION = re.compile(r"\b[A-Z]\d{2}\.\d{1,2}\b")
MIN_CODIGOS_INDICE = 8

# ─────────────────────────────────────────────────────────────────────────────────────────
# EL INDICE ANALITICO DE UN TRATADO (16-sep-2026, §1.2 y §4.2 de
# nomos/knowledge/DISENO-uso-real-16sep.md).
#
# POR QUE. Los ocho pasajes que el informe de uso real reporto como "Capitulo 505 Educacion
# medica para el futuro" son TODOS de meneghello-t2, paginas 1380 a 1457, y su texto es el
# INDICE ANALITICO del tomo ("fenitoina, 535 fenobarbital, 535 ..."). No eran una fuente
# distinta ni un capitulo mal calculado: ~75 paginas de indice que la ingesta trata como
# CUERPO, porque hasta hoy la unica regla de `indice` era la del CIE-10 (codigo con punto,
# >= 8 por chunk) y un indice analitico no tiene ni un codigo asi. Tambien cae aca
# fundamentos-derma p.157 (indice alfabetico). Un solo origen, una sola cura.
#
# UN PAR "TERMINO, NUMERO DE PAGINA": al menos tres letras, coma, numero de pagina. El numero
# admite sufijo de columna ("1438c") y rango ("723-725"), que es como los imprimen los
# tratados. No se pide mayuscula inicial: en un indice a dos niveles la subentrada va en
# minuscula ("difusamente adherente, 723").
PAGINA_DE_INDICE = r"\d{1,4}[a-z]?(?:\s?-\s?\d{1,4}[a-z]?)?"
PAR_TERMINO_PAGINA = re.compile(rf"[^\W\d_]{{3,}},\s?{PAGINA_DE_INDICE}\b")

# UNA ENTRADA DE INDICE GENERAL (los PRELIMINARES: el sumario del principio del libro).
# "1.2 Carga de enfermedad 17". El diseño la describe por linea
# (`^\d+(\.\d+)*\s+.+\s+\d+$`) y ACA NO HAY LINEAS: `generate_chunks_v2` rearma el chunk con
# " ".join(palabras) (ver el docstring de classify_content_type), asi que la misma forma se
# reconoce sobre palabras — numeracion, un puñado de palabras sin digitos ni puntos, y el
# numero de pagina. El techo de 60 caracteres es lo que separa un titulo de un parrafo.
#
# LA NUMERACION LLEVA AL MENOS UN PUNTO, y esto NO estaba en el diseño: se midio el
# 16-sep-2026 y sin el la regla se comia una tabla de dosis entera. `(\.\d+)*` admite cero
# puntos, o sea la forma "numero palabra numero", que es EXACTAMENTE la forma de una columna
# de dosis ("... 50-75 Ceftazidima 150 Tobramicina 5-7 ..."): 18 falsos pares en un chunk de
# 72 palabras. Con el punto obligatorio, esa tabla da CERO y el sumario sigue dando 16 (sus
# subentradas 1.1, 1.2, 2.1... son las que lo declaran sumario). El costo esta dicho: un
# indice general de un solo nivel ("1 Introduccion 15 2 Metodos 20") no se reconoce por esta
# via y queda `body`, que es la conducta de hoy — marca, no frena.
ENTRADA_INDICE_GENERAL = re.compile(
    rf"\b\d{{1,3}}(?:\.\d{{1,3}}){{1,3}}\.?\s+[^\W\d_][^\d.]{{2,60}}?\s{PAGINA_DE_INDICE}\b")

# EL REFUERZO POR PALABRA: "(Cont.)" encabeza la continuacion de un indice partido entre
# paginas y "Vease" es la remision cruzada que solo existe en un indice. Ninguna de las dos
# alcanza sola (ver `_es_indice`): bajan el umbral de densidad, no lo reemplazan.
REFUERZO_INDICE = re.compile(r"(?i)\((?:cont\.?|continuaci[oó]n)\)|\bv[eé]ase\b")

# CUANTOS PARES HACEN UN INDICE. 12 en un chunk de 280 palabras (la mediana del corpus) es
# una entrada cada 23 palabras: un parrafo de prosa que cite tres trabajos no llega ni cerca,
# y el indice analitico real esta en 60-70 (una entrada cada 4 palabras). Se exige el numero
# ABSOLUTO y la DENSIDAD: el absoluto frena el fragmento corto con dos pares, la densidad
# frena el capitulo largo que arrastra una tabla de referencias cruzadas al final.
MIN_PARES_INDICE = 12
MIN_PARES_INDICE_CON_REFUERZO = 8
PALABRAS_DE_REFERENCIA_INDICE = 280

# UN INDICE *ES* SUS ENTRADAS, y esta es la condicion que de verdad separa (medida el
# 16-sep-2026, y no estaba en el diseño). La densidad sola no alcanzaba: un cuadro clinico con
# dosis escritas con coma —"Cefalexina, 25 a 50 mg/kg/dia ... Clindamicina, 30 mg/kg/dia"—
# llega a 34 pares por 280 palabras y pasaba como indice. La diferencia esta en cuanto del
# texto SON las entradas: en el indice analitico las entradas cubren el 43-81 % de los
# caracteres (las cuatro muestras reales de meneghello-t2 y fundamentos-derma) y en el cuadro,
# el 27 % — el resto es la prosa de la columna "tratamiento de eleccion". El umbral va en el
# medio de las dos mediciones.
MIN_COBERTURA_INDICE = 0.35

# LAS TRES CONDICIONES DURAS, y van JUNTAS con la densidad — nunca sueltas. La leccion es del
# 13-sep-2026: la forma corta del CIE-10 ("X00") tomada sola mordio 30 chunks de contenido
# clinico real (B12, P53, C16). Un indice analitico, ademas de tener pares, NO TIENE PROSA:
#   · un quinto de sus palabras son numeros de pagina (`MIN_RATIO_NUMERICO_INDICE`);
#   · no tiene oraciones — casi ningun punto final (`MAX_PUNTOS_POR_PALABRA_INDICE`: la prosa
#     clinica del corpus esta en ~0,05, o sea una oracion cada 20 palabras);
#   · no tiene verbos — `es`, `se`, `puede`, `debe`, `son` son las cinco palabras que aparecen
#     en cualquier parrafo medico y en ninguna entrada de indice.
# Una tabla de dosis reconstruida por `render_tabla` tiene el ratio numerico de un indice y
# ninguna de las otras dos cosas; por eso las tres se piden a la vez.
MIN_RATIO_NUMERICO_INDICE = 0.20
MAX_PUNTOS_POR_PALABRA_INDICE = 0.02
MAX_VERBOS_POR_PALABRA_INDICE = 0.02
VERBOS_FRECUENTES = frozenset({"es", "se", "puede", "debe", "son"})

# Un token que es SOLO un numero de pagina (con su sufijo, su rango y la puntuacion pegada).
TOKEN_NUMERICO = re.compile(rf"^{PAGINA_DE_INDICE}[.,;:]?$")
# Un final de oracion: la misma idea que `_find_sentence_boundary`, sin contar "1." ni "3.2.".
FIN_DE_ORACION = re.compile(r"[.?!]$")
SOLO_NUMERACION = re.compile(r"^\d+(?:\.\d+)*\.?$")

# LA POSICION COMO REFUERZO, NUNCA COMO CONDICION (16-sep-2026). El indice analitico vive al
# final del libro y el sumario al principio, pero un chunk no es indice POR ESTAR ahi: la
# ultima pagina de un tratado tambien puede ser el colofon de un capitulo. Estar en el ultimo
# 5 % o en las primeras `MAX_PAGINAS_PRELIMINARES` paginas baja el umbral de densidad de 12 a
# 8 pares; las tres condiciones duras se siguen exigiendo enteras.
FRACCION_FINAL_DEL_LIBRO = 0.05
MAX_PAGINAS_PRELIMINARES = 30

# LA MAQUETA NO ES CONTENIDO (13-sep-2026, items C-1/C-2 de la auditoria de exposicion). El
# encabezado y el pie de pagina se cuelan en CADA chunk: suman terminos al indice full-text, entran
# al texto que se embebe y no dicen nada del tema. Se los reconoce por la FORMA -repeticion y aviso
# legal-, nunca por el nombre del titular ni del sitio de donde salio un PDF: una regla que nombra a
# alguien hay que cambiarla con cada libro nuevo, y en un espejo publico cuenta de donde vino el
# corpus. Ver `clean_text` y `lineas_repetidas`.
#
# TRES paginas, y el numero es el minimo que distingue: con 2 alcanza una coincidencia entre dos
# paginas seguidas -frecuente en una tabla partida o en dos parrafos que arrancan igual-; con 3 la
# coincidencia ya es un patron de la maqueta. Los documentos de menos de 3 paginas quedan sin filtro
# de repeticion a proposito: ahi no hay evidencia de repeticion que valga.
MIN_PAGINAS_REPETIDAS = 3
# Cuantas lineas no vacias del principio y del final de cada pagina se consideran "borde". DOS, no
# una: un pie suele traer el titulo corrido y el pie de imprenta en dos lineas, y un encabezado, el
# nombre de la obra y el del capitulo. Mas que dos empieza a morder el primer parrafo.
LINEAS_DE_BORDE = 2

# EL FOLIO: la linea que es SOLO el numero de pagina impreso (16-sep-2026, tanda 6 del diseño
# nomos/knowledge/DISENO-uso-real-16sep.md, §4.6 y la bitacora "Experimento de T6").
#
# QUE SE ROMPIO. La regla vivia como `re.sub(r'^\s*\d{1,4}\s*$', '', text, re.MULTILINE)`: borraba
# CUALQUIER linea de 1 a 4 digitos, estuviera donde estuviera. Y PyMuPDF emite cada celda de una
# tabla en su propio renglon, asi que toda celda que sea un numero corto sin separador desaparecia.
# Medido sobre la Guia ITU SAP 2022 (19 pags, `find_tables()` no ve ni una tabla porque la maqueta no
# tiene lineas): de la Tabla 5, "Amikacina 15 12-24 IV-IM" perdia el 15 -la dosis- y conservaba el
# 12-24 y el 99,8, que tienen separador. Ese es el patron "tablas con celdas vacias" del informe de
# uso real, y NO era la rama de tablas: era esta regex.
#
# LA INTENCION ERA EL FOLIO, y el folio esta en un BORDE. Asi que la regla se acota a lo que quiso
# decir: sale a lo sumo UNA linea de digitos por borde, y solo si cae en el BORDE de la pagina — el
# mismo borde que ya usa el filtro de maqueta, `LINEAS_DE_BORDE` lineas no vacias arriba y abajo. Una
# celda de tabla vive en el medio de la pagina y sobrevive; el numero impreso arriba o abajo se va.
#
# POR QUE DOS LINEAS Y NO UNA (medido el 16-sep-2026; el borde de una sola linea era lo primero que
# se probo). El folio casi nunca es la PRIMERA linea: viene detras del titulo corrido, que es
# exactamente la razon por la que `LINEAS_DE_BORDE` ya valia 2 para la maqueta. En PRONAP
# neurodesarrollo el orden de lectura es "horacio lejarraga • EVALUACION DEL DESARROLLO" / "12" /
# el cuerpo: con borde de 1 linea sobrevivian 148 de las 180 lineas de digitos del libro -y ~120 de
# ellas eran folios de verdad, uno por pagina-; con borde de 2, salen 152 y quedan 28, que son
# celdas. En Sanguinetti, 938 lineas de digitos, 201 quitadas y 737 conservadas (antes: 938
# borradas). El precio medido y aceptado: en las 8 paginas del INDICE de Sanguinetti la primera y la
# ultima entrada pierden su numero de pagina (~12 numeros en todo el libro), porque ahi el borde de
# la pagina es una entrada de indice. Son chunks que `classify_content_type` marca `indice` y que el
# retrieval ya excluye, asi que el dano cae donde menos cuesta.
#
# LO QUE ESTA REGLA NO ATRAPA, dicho para que nadie lo redescubra: el folio que el orden de lectura
# de PyMuPDF deja mas adentro que el borde (una pagina a dos columnas donde el bloque del pie sale
# antes que el final de la segunda columna). Queda una linea de digitos en el medio del texto. Es el
# precio de no comerse las celdas, y es el lado barato del error: un numero de mas en un chunk se
# lee; una dosis de menos, no se ve.
#
# NO SE SOLAPA con la segunda pasada de maqueta (`lineas_repetidas` + `quitar_lineas`, que
# `parse_pdf_v2` corre despues sobre el documento entero): esa mira tambien el borde pero exige que
# la linea sea IDENTICA en >= 3 paginas, y un folio nunca lo es -cambia en cada pagina-. Lo que esa
# pasada saca es el encabezado/pie con TEXTO, incluido el que lleva el numero pegado ("Cap. 3 -
# Editorial X - 45"), que este filtro tampoco ve. Las dos reglas se reparten el borde sin pisarse.
FOLIO_SUELTO = re.compile(r'^\s*\d{1,4}\s*$')
# Mas largo que esto no es un encabezado ni un pie: es un PARRAFO que se repite, y borrar un parrafo
# es borrar contenido. El techo se pago con una medicion (13-sep-2026): los PDF sinteticos de la QA
# meten el mismo cuerpo en cada pagina en UNA linea larguisima -PyMuPDF no hace wrap-, asi que sin
# el techo esa linea quedaba "repetida en el borde de 6 paginas" y el filtro se comia el cuerpo
# entero del documento. Un encabezado real -titulo de obra, titulo corrido, pie de imprenta- entra
# de sobra en 120 caracteres.
MAX_CHARS_MAQUETA = 120
# Y ADEMAS TIENE QUE SER MAS CORTA QUE EL CUERPO. Esta es la condicion que de verdad distingue, y la
# pago el escenario P10 del banco de QA (13-sep-2026): su PDF es un parrafo de 3.000 palabras tomadas
# de un vocabulario de 15 que CICLA, asi que reportlab produce paginas periodicas y la primera y la
# ultima linea de la pagina 1 son identicas a las de la 3 y la 5 -medido: 4 lineas de 90 a 99
# caracteres repetidas en 3 paginas-. Son lineas de CUERPO justificadas al ancho de la caja: el
# filtro se comia dos por pagina y el documento pasaba de 13 chunks a 4. Un encabezado o un pie, en
# cambio, NUNCA llena la caja: es una linea corta suelta arriba o abajo. Se compara contra la linea
# mas larga del documento -no contra la mediana, que se hunde en un documento de listas- y el 0,75
# deja pasar un titulo corrido largo (70 de 110) y frena una linea de cuerpo (90 de 99).
FRACCION_ANCHO_MAQUETA = 0.75
# Un aviso de derechos POR SU FORMA: el simbolo (© o "(c)") seguido de año, o una de las palabras
# con las que se escribe un aviso. Sin un solo nombre propio: no nombra editorial, autor ni sitio,
# asi que sirve igual para cualquier libro y no hay que editar el codigo cuando entra uno nuevo.
# El simbolo SOLO no alcanza -"©" suelto aparece en simbolos de unidades y en notas al pie-: lo que
# convierte a la linea en un aviso es el año o la palabra.
# Y LA LINEA TIENE QUE SER CORTA (el mismo techo que la maqueta, `MAX_CHARS_MAQUETA`): un aviso
# legal es una linea suelta del pie de imprenta. Una linea de CUERPO que use la palabra -un tratado
# de derecho hablando de "los derechos reservados a las provincias", un manual que diga "no
# fotocopiar la receta"- es una linea justificada al ancho de la caja, y se conserva. Sin el techo,
# la palabra sola borraba contenido en el dominio derecho.
AVISO_DE_DERECHOS = re.compile(
    r'(?i)^(?=.{1,' + str(MAX_CHARS_MAQUETA) + r'}$).*'
    r'(?:(?:©|\(c\))\s*\d{4}|copyright|derechos\s+reservados|fotocopiar).*$',
    re.MULTILINE)

# ─────────────────────────────────────────────────────────────────────────────────────────
# LA CALIDAD DEL TEXTO, POR CHUNK Y POR FORMA (16-sep-2026, §1.6 y §4.2 del diseño).
#
# QUE HUECO CIERRA. `text_quality` NO lo calculaba el pipeline: cero ocurrencias en
# `pipeline/` y en `api/services/ingest.py`. Lo escribio UNA vez `backfill_books.py` sobre una
# MUESTRA DE SEIS CHUNKS por libro (25/50/75 % del libro) con dos firmas —letras sueltas y
# palabras largas sin vocales— que NO VEN bytes de control, ni `U+FFFD`, ni `(cid:NN)`:
# `.isalpha()` y `[^\W\d_]+` los descartan antes de contar. Medido sobre los 2.735 chunks que
# el uso real toco: `sanguinetti-semiologia` 13 de 13 con bytes de control (CID desplazado,
# "HVWH\x03PRWLYR") y `castano-lopez-laboratorio` 11 de 11 — y el catalogo daba las dos por
# `ok`. La calidad pasa a calcularse ACA, por unidad, con las cinco firmas, y el `:Book` la
# deriva de sus chunks (`pipeline/carga.py`) en vez de adivinarla con seis muestras.
#
# DOCTRINA (§3.3 del diseño de ingesta, y §4.2 de este): MARCA, NO FRENA. Nada se excluye del
# corpus ni del retrieval por calidad; la etiqueta viaja para que quien lee sepa que tiene
# delante. Por eso no hay un umbral de "descartar".
#
# LO QUE NUNCA ES TEXTO: un byte de control (el PDF con la fuente CID sin mapear los escupe),
# el caracter de reemplazo U+FFFD (una decodificacion que fallo) y el literal "(cid:NN)" (el
# extractor que se rindio y escribio el numero de glifo). Las tres son de FORMA y valen igual
# en cualquier idioma y cualquier dominio.
BYTES_DE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
CARACTER_DE_REEMPLAZO = "�"   # U+FFFD, escapado: un literal invisible en el fuente no se revisa
CID_LITERAL = re.compile(r"\(cid:\s?\d+\)")

# LAS ETIQUETAS y donde corta cada una. Son los MISMOS numeros con los que `assess_quality`
# venia juzgando libros enteros desde el 29-jul-2026 (corrupto > 0,30; sospechoso > 0,15), a
# proposito: la escala no cambia de significado por mudarse de la fuente a la unidad.
UMBRAL_CALIDAD_CORRUPTA = 0.30
UMBRAL_CALIDAD_DUDOSA = 0.15

# FIRMAS DURAS Y FIRMAS BLANDAS: QUIEN PUEDE DECIR "CORRUPTA" (16-sep-2026, dry-run de
# `reclasificar_contenido.py --calidad` contra el grafo vivo — solo lectura, `data/reclasificacion/
# dryrun.json`, bitacora del diseño nomos/knowledge/DISENO-uso-real-16sep.md).
#
# QUE MOSTRO LA MEDICION. Sobre los 91.032 chunks del corpus: corrupta 9.480 / dudosa 612 / ok
# 80.940. Las corruptas son casi todas de las tres fuentes que de verdad estan rotas (atlas
# ginecologico 7.489, Castaño 1.705, Sanguinetti 173). Pero las pocas que aparecian en fuentes
# SANAS no eran corrupcion: eran TABLAS NUMERICAS y EPIGRAFES DE FIGURA — una tabla de valores de
# referencia de Meneghello (0,34-0,50), la tabla de vacunas de Farreras con sus "x" y sus "Si"
# (0,45), una formula quimica de DeVita (0,33), una tabla de tasas (0,62), un epigrafe de Abbas con
# los marcadores A/B/C de la figura (0,81)—. En todas disparaban SOLAS las dos firmas que miden
# forma tipografica y no decodificacion: letras sueltas y proporcion no alfabetica.
#
# LA REGLA. Esas dos, solas, llegan a lo sumo a `dudosa`. `corrupta` exige que la que cruce el
# umbral sea una firma DURA: un byte de control, un U+FFFD, un "(cid:NN)", una palabra larga sin
# vocales o texto ESPACIADO. Las cinco duras dicen "la decodificacion fallo" y ninguna aparece en
# texto sano; las dos blandas dicen "esto no parece prosa", que es exactamente lo que es una tabla
# de dosis. La doctrina no cambia —MARCA, NO FRENA—, pero una tabla de dosis no le puede decir
# "texto corrupto" a quien la lee: la etiqueta viaja al resultado de la busqueda.
#
# LETRAS ESPACIADAS *NO* ES LETRAS SUELTAS, y la diferencia es la que salva el caso del
# 29-jul-2026. `ratio_letras_sueltas` cuenta tokens de una letra ESTEN DONDE ESTEN: los
# marcadores A/B/C de un epigrafe de figura y la columna de "x" de una tabla de vacunas puntuan
# igual que un PDF extraido caracter a caracter. `ratio_letras_espaciadas` cuenta solo las que
# vienen en CORRIDA deletreando una palabra ("l a i n s u f i c i e n c i a"), que es el sintoma
# de diagnostic-pathology-gynecological-3ed y no le pasa a ninguna tabla. Ver esa funcion para las
# tres condiciones de la corrida.
#
# LO QUE NO CAMBIA: la calidad de la FUENTE. `carga.CALIDADES_DEGRADADAS` cuenta `dudosa` Y
# `corrupta` por igual, asi que bajar un chunk de corrupta a dudosa no mueve el `text_quality` de
# ningun libro. Y `calidad_valor` sigue siendo la PEOR de las firmas, sin recortar: la etiqueta se
# modera, la medicion no se toca. Tampoco cambia `calidad_valor` por sumar la firma nueva: toda
# letra espaciada es tambien una letra suelta, asi que `espaciadas <= letras_sueltas` siempre y el
# maximo es el mismo.
FIRMAS_DURAS = ("control", "reemplazo", "cid", "sin_vocales", "espaciadas")
FIRMAS_BLANDAS = ("letras_sueltas", "no_alfabetico")

# COMO SE COMPARAN FIRMAS QUE NO MIDEN LO MISMO. `calidad_valor` es la PEOR de todas,
# y para que "peor" signifique algo cada una se normaliza contra su propia REFERENCIA:
# la proporcion a la que esa firma sola ya dice "esto es basura". Un 10 % de bytes de control
# es basura total; un 10 % de letras sueltas es un corpus sano (el corpus real mide 3-9 %).
# Sin la normalizacion, la firma de los bytes de control —la que destapo sanguinetti— quedaba
# debajo del umbral justo en el caso que vino a cazar (el chunk medido: 8 bytes en 78
# caracteres, 0,10 crudo).
REFERENCIA_CONTROL = 0.10
REFERENCIA_REEMPLAZO = 0.05
REFERENCIA_CID = 0.10
# Las dos firmas historicas se miden en su propia escala (referencia 1,0): asi `dudosa` y
# `corrupta` caen exactamente donde caian `suspect` y `corrupt`. La de letras espaciadas va en la
# misma escala por la misma razon: es la mitad dura de la de letras sueltas.
REFERENCIA_LETRAS_SUELTAS = 1.0
REFERENCIA_SIN_VOCALES = 1.0
REFERENCIA_ESPACIADAS = 1.0

# LAS TRES CONDICIONES DE UNA CORRIDA ESPACIADA (16-sep-2026). Una corrida es una tira de tokens
# de UNA sola letra, seguidos. Para que cuente como "el PDF se extrajo caracter a caracter":
#   · LARGO >= 4. Tres letras sueltas seguidas las produce el español normal ("vitamina A y D") y
#     tambien las produce una lista de incisos; cuatro ya es deletreo. Es el mismo numero de
#     tokens que la corrida minima de `RE_TEXTO_ESPACIADO` (2 pares + 1), que resuelve el mismo
#     problema en los titulos (`_clean_spaced_text`).
#   · >= 3 LETRAS DISTINTAS. La columna de "x" de una tabla de vacunas —"x x PCV13 Si x x x x x x
#     x"— es la corrida mas larga de todo el corpus sano y tiene UNA letra distinta.
#   · MAYORITARIAMENTE MINUSCULA. Los marcadores de panel de una figura ("A B C D V D J C") y los
#     simbolos de una formula quimica ("N A B O N N O") son MAYUSCULAS; una palabra deletreada,
#     no. El empate (dos y dos, "A y D e") NO cuenta: se exige mayoria estricta.
# Las tres salieron de los cinco falsos positivos del dry-run del 16-sep y del caso verdadero del
# 29-jul ("l a i n s u f i c i e n c i a c a r d i a c a"), que las cumple las tres.
MIN_CORRIDA_ESPACIADA = 4
MIN_LETRAS_DISTINTAS_ESPACIADA = 3

# LA PROPORCION NO ALFABETICA, con piso. Una tabla de dosis es legitimamente no alfabetica
# —"Ceftriaxona 50-75 Ceftazidima 150" mide 0,30— y marcarla degradada fue justo el falso
# positivo que se pago el 3-ago-2026 con los valores hematimetricos del PRONAP. Por eso esta
# firma no cuenta desde cero: solo lo que pasa el PISO, reescalado hasta la REFERENCIA.
PISO_NO_ALFABETICO = 0.45
REFERENCIA_NO_ALFABETICO = 0.95

# CUANTA EVIDENCIA HACE FALTA PARA JUZGAR POR TOKENS. Las firmas de caracteres (control,
# reemplazo, cid) valen siempre: un byte de control en un texto de diez palabras ya es un
# texto roto. Las de tokens necesitan muestra, o el ruido decide: un titulo de tres palabras
# con una inicial suelta daria 33 % de letras sueltas.
MIN_TOKENS_CALIDAD = 30
MIN_PALABRAS_LARGAS_CALIDAD = 10
#: Desde cuantos caracteres una palabra es "larga" para la firma CID (sin vocales).
LARGO_PALABRA_SIN_VOCALES = 7
VOCALES = frozenset("aeiouáéíóúüàèìòùâêîôûäëïöÿ")

MIN_FILAS_TABLA = 2
MIN_COLS_TABLA = 2
SOLAPE_BLOQUE_TABLA = 0.5          # fraccion del bloque que debe caer dentro de la tabla
MAX_CHARS_ENCABEZADO = 60          # mas largo que esto no es un nombre de columna
MIN_RATIO_LETRAS_ENCABEZADO = 0.5  # ver _parece_encabezado
MAX_PERDIDA_TOKENS = 0.02          # ver _pierde_contenido

# CUANTO PUEDE PERDER UNA TABLA ANTES DE QUE NO SEA UNA TABLA (16-sep-2026, tanda 6, §4.6 del
# diseño nomos/knowledge/DISENO-uso-real-16sep.md). Es la valvula POR TABLA: ver
# `_pierde_la_tabla` y `extraer_texto_pagina`.
#
# SE MIDE EN PALABRAS DISTINTAS, no en ocurrencias, y esa es la diferencia con la guarda de
# pagina. Motivo medido: PRONAP sobreimprime los recuadros destacados —el mismo parrafo escrito
# DOS VECES dentro del mismo bloque de PyMuPDF, ni siquiera en dos bloques— y la grilla lo
# rinde una sola vez. Contando ocurrencias, 33 de las 105 tablas del libro "perdian" exactamente
# 0,500 y se descartaban; contando palabras distintas, pierde UNA sola tabla en todo el libro (la
# p. 160, donde `find_tables` agarro el pie de imprenta y rindio "159."). Lo mismo en Sanguinetti:
# 8 tablas "perdian" por ocurrencias y 3 por palabras distintas.
#
# EL NUMERO. Entre 0,05 y 0,30 los dos libros dan el MISMO resultado (una tabla descartada cada
# uno), asi que el umbral se pone en el medio. Por debajo de 0,05 empieza a morder tablas sanas
# a las que la grilla les dejo afuera el rotulo de una columna (Sanguinetti p. 61 pierde
# "diastolica", 0,036; p. 142 pierde "h2o", 0,042): perder una palabra de un encabezado y ganar
# la estructura de ocho filas es un buen canje, y la palabra sigue en el chunk vecino.
MAX_PERDIDA_TABLA = 0.10


def normalize_for_search(text: str) -> str:
    """Quita acentos y pasa a minúsculas para full-text search."""
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c)).lower()


def quitar_folio(texto: str, borde: int = LINEAS_DE_BORDE) -> str:
    """Saca el número de página suelto, y SÓLO si está en un borde de la página.

    A lo sumo UNA línea de dígitos por borde: se miran las `borde` primeras líneas no vacías y
    las `borde` últimas, y en cada zona sale la primera que sea sólo dígitos. Ver
    `FOLIO_SUELTO` arriba para el porqué (16-sep-2026, §4.6 del diseño de uso real): la regla
    vieja borraba cualquier línea de 1-4 dígitos de cualquier parte de la página y se comía las
    celdas numéricas de las tablas, que PyMuPDF emite una por renglón.

    Es PURA y recibe el texto de UNA página: el borde es el de esa página, no el del documento.
    El `borde` es parámetro para poder medirlo en un test sin tocar la constante del módulo.
    """
    lineas = texto.split('\n')
    no_vacias = [i for i, linea in enumerate(lineas) if linea.strip()]
    if not no_vacias:
        return texto
    # La zona de abajo se recorre de afuera hacia adentro: el folio del pie es la ÚLTIMA línea
    # antes que la anteúltima, igual que el del encabezado es la primera antes que la segunda.
    # Y NO SE SOLAPAN: en una página de pocas líneas las dos zonas comparten renglones, y sin
    # esto una página que empieza con dos números —una columna de tabla— los perdía los dos.
    arriba = no_vacias[:borde]
    abajo = [i for i in no_vacias[-borde:] if i not in set(arriba)]
    for zona in (arriba, abajo[::-1]):
        for i in zona:
            if FOLIO_SUELTO.match(lineas[i]):
                lineas[i] = ''
                break   # a lo sumo una por borde: la de al lado ya es contenido
    return '\n'.join(lineas)


def clean_text(text: str) -> str:
    """Limpia texto extraído de PDF: lo que es MAQUETA y no contenido.

    EL FILTRO ES DE FORMA, NO DE TITULAR (13-sep-2026, items C-1/C-2 de la auditoría de
    exposición). Hasta hoy esta función borraba líneas por una lista de regex que NOMBRABAN a
    un editor concreto y al sitio del que había salido un PDF; eso viajaba al espejo OSS, que
    es público, y decía de dónde venía el corpus de esta instancia. Lo que se quería sacar no
    era "las líneas de tal editorial": era el pie de imprenta, que es de la maqueta y se cuela
    en cada chunk. Así que ahora se reconoce por su FORMA, y la forma es doble:

      · REPETICIÓN — el encabezado/pie se repite idéntico en muchas páginas. Eso no se puede
        ver desde acá: esta función recibe UNA página sin contexto del documento. Lo hace
        `lineas_repetidas` + `quitar_lineas`, que `parse_pdf_v2` aplica en una pasada previa.
      · AVISO DE DERECHOS — `AVISO_DE_DERECHOS`, abajo: el símbolo con año, o la palabra con la
        que se escribe un aviso. Sin un solo nombre propio, así que vale para cualquier libro
        de cualquier editorial y no hay que tocar el código cuando entra un libro nuevo.

    EL FOLIO SE ACOTÓ AL BORDE el 16-sep-2026 (tanda 6, §4.6 del diseño de uso real): ver
    `FOLIO_SUELTO` y `quitar_folio`. Antes era un `re.sub` con `re.MULTILINE` sobre la página
    entera y borraba las celdas numéricas de las tablas. El encabezado de tratado ("… medicina
    interna … edición …") queda como estaba: es de las de siempre y sacarlo cambiaría el
    parseo sobre texto que no es un encabezado repetido, que es justo lo que esta tanda no toca.
    """
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = quitar_folio(text)
    text = re.sub(r'(?i)^.*medicina interna.*edici[oó]n.*$', '', text, flags=re.MULTILINE)
    text = AVISO_DE_DERECHOS.sub('', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = '\n'.join(line.strip() for line in text.split('\n'))
    return text.strip()


def _clave_de_linea(linea: str) -> str:
    """La forma comparable de una línea: espacios normalizados y sin bordes.

    "Idénticas" se decide sobre ESTO y no sobre los bytes: el mismo pie extraído de dos
    páginas distintas suele diferir en un espacio doble o en un espacio fino, porque el
    espaciado lo pone la maqueta y no el texto.
    """
    return re.sub(r'\s+', ' ', linea).strip()


def lineas_repetidas(paginas: list, minimo: int = MIN_PAGINAS_REPETIDAS,
                     borde: int = LINEAS_DE_BORDE) -> set:
    """Las líneas de encabezado/pie que se repiten en >= `minimo` páginas del documento.

    Args:
        paginas: textos de página YA limpiados con `clean_text`, en orden.
        minimo: en cuántas páginas tiene que aparecer una línea para tratarla como maqueta.
        borde: cuántas líneas no vacías de arriba y de abajo de cada página se miran.

    Returns:
        Conjunto de claves (`_clave_de_linea`) a descartar. Vacío si no hay ninguna.

    ES PURA: recibe texto, devuelve un conjunto. No abre el PDF, no toca disco y no depende
    del libro, del dominio ni del perfil — por eso se puede testear sin PyMuPDF.

    POR QUÉ SÓLO EL BORDE Y NO TODA LA PÁGINA. "Línea repetida en N páginas" también describe
    a una frase de cuerpo que un libro repite (una advertencia al pie de cada tabla, un
    estribillo de un manual), y borrarla sería borrar contenido. Un encabezado o un pie, además
    de repetirse, está SIEMPRE en el mismo lugar: arriba o abajo. Mirar sólo las `borde`
    primeras y últimas líneas no vacías de cada página es lo que separa una cosa de la otra con
    el orden de lectura, que es lo único que este módulo ve del PDF.

    Y ADEMÁS TIENE QUE SER CORTA: corta en absoluto (`MAX_CHARS_MAQUETA`) y corta RELATIVO al
    ancho del cuerpo (`FRACCION_ANCHO_MAQUETA` de la línea más larga del documento). Las dos
    condiciones salieron de dos falsos positivos medidos el 13-sep-2026 y están anotadas arriba:
    el documento cuyo cuerpo entero es una sola línea repetida, y el párrafo periódico de P10,
    cuyas líneas de cuerpo justificadas se repiten entre páginas. Un pie de imprenta no llena la
    caja de texto; una línea de cuerpo sí.

    LO QUE ESTE FILTRO NO ATRAPA, dicho para que nadie lo descubra de nuevo: el pie que lleva el
    número de página EN LA MISMA LÍNEA ("Cap. 3 — Editorial X — 45") no es idéntico entre
    páginas y sobrevive. La línea de número suelto sí la saca `clean_text`, que es el caso
    frecuente en los PDF del corpus. Normalizar dígitos acá haría colapsar filas de tablas.
    """
    por_pagina = []
    ancho = 0
    for texto in paginas:
        lineas = [x for x in (_clave_de_linea(l) for l in texto.split('\n')) if x]
        if not lineas:
            continue
        por_pagina.append(lineas)
        ancho = max(ancho, max(len(x) for x in lineas))

    techo = min(MAX_CHARS_MAQUETA, int(ancho * FRACCION_ANCHO_MAQUETA))
    cuenta = Counter()
    for lineas in por_pagina:
        # `set`: en una página de menos de 2*borde líneas las zonas se solapan, y una línea no
        # puede contar dos veces por la misma página.
        zona = set(lineas[:borde] + lineas[-borde:])
        cuenta.update(x for x in zona if len(x) <= techo)
    return {clave for clave, n in cuenta.items() if n >= minimo}


def quitar_lineas(texto: str, descartar: set) -> str:
    """Saca del texto las líneas cuya forma normalizada esté en `descartar`.

    Las líneas vacías se conservan: `lineas_repetidas` nunca las devuelve (no son maqueta) y
    los saltos separan párrafos, que es lo que `detect_structure` lee.
    """
    if not descartar:
        return texto
    return '\n'.join(l for l in texto.split('\n') if _clave_de_linea(l) not in descartar)


def _area(r) -> float:
    return max(0.0, r.x1 - r.x0) * max(0.0, r.y1 - r.y0)


def _dentro_de_alguna(rect, cajas: list) -> bool:
    """El bloque cae mayormente dentro de alguna tabla ya reconstruida?"""
    import fitz  # PyMuPDF: perezoso, le costaba ~10 s al arranque en frio de Cloud Run
    area = _area(rect)
    if area <= 0:
        return False
    for caja in cajas:
        inter = fitz.Rect(rect)
        inter.intersect(caja)
        if _area(inter) / area > SOLAPE_BLOQUE_TABLA:
            return True
    return False


def _celda_limpia(v) -> str:
    if not v:
        return ""
    return re.sub(r'\s+', ' ', str(v)).strip()


def _fusionar_continuaciones(filas: list) -> list:
    """Une con la fila anterior las filas que son continuacion de una celda.

    Una celda multilinea se parte en varias filas, con None en las columnas que
    no siguen:
        ['VCM', '80-94 u3', 'Macrocitosis']
        [None,  None,       'Normocitosis']   <- continuacion de la de arriba
    Sin esto, 'Normocitosis' quedaria como una fila propia sin su parametro.
    """
    salida = []
    for fila in filas:
        celdas = [_celda_limpia(c) for c in fila]
        if salida and not celdas[0] and any(celdas):
            for i, c in enumerate(celdas):
                if c and i < len(salida[-1]):
                    salida[-1][i] = (salida[-1][i] + " " + c).strip()
        else:
            salida.append(celdas)
    return salida


def _parece_encabezado(fila: list) -> bool:
    """La fila sirve como encabezado, o es una fila de datos mas?

    Un encabezado es texto: nombra la columna. Una fila de datos trae valores,
    unidades y simbolos. Se mide la proporcion de letras de cada celda, que es
    lo que mejor separa un caso del otro:
        'VALOR NORMAL' -> 100% letras   sirve
        '0 puntos'     ->  86% letras   sirve (el digito es parte del nombre)
        '80-94 um3'    ->  14% letras   NO sirve, es un valor
    Basta una celda que no pase para descartar la fila entera: etiquetar mal las
    columnas es peor que no etiquetarlas.
    """
    llenas = [c for c in fila if c]
    if len(llenas) < MIN_COLS_TABLA:
        return False
    for celda in llenas:
        if len(celda) > MAX_CHARS_ENCABEZADO:
            return False
        cuerpo = re.sub(r'\s', '', celda)
        if not cuerpo:
            return False
        letras = sum(1 for c in cuerpo if c.isalpha())
        if letras / len(cuerpo) < MIN_RATIO_LETRAS_ENCABEZADO:
            return False
    return True


def render_tabla(tabla) -> str:
    """Tabla detectada -> una fila autocontenida por linea. '' si no sirve.

    NO se usa tabla.header: cuando decide que el encabezado esta fuera de la
    tabla agarra el texto de arriba, y ahi puede levantar un pedazo suelto de la
    linea anterior. En el hemograma de SIAJ convirtio 'Serie Roja:' en 'oja:' y
    etiqueto las tres columnas con eso. El unico encabezado confiable es la
    primera fila de la propia grilla, y solo si parece un encabezado.
    """
    try:
        filas = _fusionar_continuaciones(tabla.extract())
    except Exception:
        return ""
    if len(filas) < MIN_FILAS_TABLA:
        return ""

    # Una tabla partida entre dos paginas deja la continuacion sin encabezado.
    # En ese caso se emiten las filas sin nombre de columna: se conserva el corte
    # entre celdas, que ya es mas de lo que daba get_text() aplanando todo.
    encabezados = None
    if _parece_encabezado(filas[0]):
        encabezados = filas[0]
        filas = filas[1:]
    if len(filas) < MIN_FILAS_TABLA:
        return ""

    lineas = []
    for fila in filas:
        partes = []
        for i, celda in enumerate(fila):
            if not celda:
                continue
            col = encabezados[i] if encabezados and i < len(encabezados) else ""
            partes.append(f"{celda} ({col})" if col else celda)
        if not partes:
            continue
        linea = "; ".join(partes)
        lineas.append(linea if linea.endswith(('.', ':', '?', '!')) else linea + ".")

    return "\n".join(lineas)


def _tokens(texto: str) -> Counter:
    return Counter(re.findall(r'\w{2,}', texto.lower()))


def _palabras_perdidas(viejo: str, nuevo: str) -> float:
    """Fraccion de PALABRAS DISTINTAS de `viejo` que no aparecen en `nuevo`. 0 si no hay texto.

    POR QUE DISTINTAS Y NO OCURRENCIAS (16-sep-2026, tanda 6, §4.6 del diseño de uso real).
    Hasta hoy se contaban ocurrencias, y eso hacia que el texto SOBREIMPRESO —el mismo parrafo
    escrito dos veces, que PyMuPDF devuelve dos veces y la grilla rinde una— se leyera como
    perdida. Medido en PRONAP neurodesarrollo: de las 74 paginas con reconstruccion aceptada,
    53 quedaban vetadas por ocurrencias (hasta 0,168 de "perdida") y NINGUNA perdia una sola
    palabra distinta —`faltan=[]` en las 53—. Contando palabras distintas, la guarda sigue
    cazando lo que vino a cazar (la pagina de prosa a dos columnas que `find_tables` cree
    grilla y deja en un renglon: ahi faltan casi todas las palabras) y deja de castigar una
    maqueta que repite tinta.
    """
    tenia = set(_tokens(viejo))
    if not tenia:
        return 0.0
    return len(tenia - set(_tokens(nuevo))) / len(tenia)


def _pierde_contenido(viejo: str, nuevo: str) -> bool:
    """La reconstruccion se comio texto que get_text() si traia? (guarda de PAGINA)

    find_tables() a veces marca como tabla una region que en realidad es prosa
    maquetada en columnas. Al excluir los bloques de esa region y reemplazarlos
    por las pocas filas que la grilla logra extraer, la pagina pierde parrafos
    enteros. Visto en el Manual de Pediatria: una pagina de 3.375 caracteres
    quedaba en 193.

    Perder texto es peor que aplanarlo, asi que ante cualquier perdida se vuelve
    al texto plano. El cambio solo puede agregar estructura, nunca sacar contenido.

    DESDE EL 16-sep-2026 ES EL ULTIMO RECURSO, no la unica valvula: la perdida se evalua
    primero POR TABLA (`_pierde_la_tabla`) y esta guarda cubre lo que esa medicion no ve —el
    bloque que solapaba dos cajas y quedo huerfano, el reordenamiento por `y0`—. Ver
    `extraer_texto_pagina`.
    """
    return _palabras_perdidas(viejo, nuevo) > MAX_PERDIDA_TOKENS


def _pierde_la_tabla(dentro: str, render: str) -> bool:
    """La grilla dejo afuera palabras que el PDF si tenia dentro de la caja de la tabla?

    LA VALVULA POR TABLA (16-sep-2026, §4.6 del diseño de uso real). Mide lo mismo que
    `_pierde_contenido` pero contra su propio umbral: el de la pagina esta calibrado sobre
    miles de palabras y el de una tabla, sobre decenas, donde el 2 % es UNA palabra. Ver
    `MAX_PERDIDA_TABLA`.
    """
    return _palabras_perdidas(dentro, render) > MAX_PERDIDA_TABLA


def _bloques_de_texto(page) -> list:
    """Los bloques de la pagina como [(Rect, texto)], en el orden que los da PyMuPDF.

    Se leen UNA sola vez y se usan para las dos cosas: medir la perdida de cada tabla y
    armar la pagina. `get_text("blocks")` devuelve (x0, y0, x1, y1, texto, nro, tipo).
    """
    import fitz  # PyMuPDF: perezoso, le costaba ~10 s al arranque en frio de Cloud Run
    return [(fitz.Rect(b[:4]), b[4], (b[6] if len(b) > 6 else 0)) for b in page.get_text("blocks")]


def _texto_de_la_caja(bloques: list, caja) -> str:
    """El texto que el PDF trae DENTRO de la caja de una tabla: contra esto se mide la perdida.

    DOS FILTROS, y los dos se pagaron midiendo (16-sep-2026):

      · SOLO BLOQUES DE TEXTO (tipo 0). La descripcion que PyMuPDF emite por un bloque de
        imagen ("<image: DeviceRGB, width ...>") aporta tokens que ninguna grilla puede tener:
        una figura dentro de la caja haria "perder" a una tabla sana.
      · SIN BLOQUES REPETIDOS. PyMuPDF emite DOS VECES el mismo bloque -misma caja, mismo
        texto- cuando el PDF sobreimprime el parrafo (los recuadros destacados de PRONAP lo
        hacen en todo el modulo de vacunas). La tabla lo rinde UNA vez, asi que sin deduplicar
        toda esa familia media exactamente 0,500 de perdida y se descartaba entera. El armado
        de la pagina NO deduplica a proposito: `page.get_text()` tambien trae el bloque dos
        veces, y sacar uno de los dos haria que la guarda de pagina viera una perdida que no
        existe.
    """
    vistos = set()
    piezas = []
    for rect, texto, tipo in bloques:
        if tipo != 0 or not _dentro_de_alguna(rect, [caja]):
            continue
        clave = (round(rect.y0, 1), round(rect.x0, 1), texto)
        if clave in vistos:
            continue
        vistos.add(clave)
        piezas.append(texto)
    return "\n".join(piezas)


def extraer_texto_pagina(page, decodificador=None) -> str:
    """Texto de la pagina con las tablas reconstruidas en su lugar de lectura.

    `decodificador` es opcional y lo arma `parse_pdf_v2` UNA vez por libro
    (`pipeline/decodificacion.py`, tanda 6, 16-sep-2026): cuando el PDF trae el mapeo de glifos
    roto, reemplaza los spans ilegibles por su version decodificada DESPUES de armar la pagina,
    asi la reconstruccion de tablas y el orden de lectura no cambian. Sin el —el caso normal—
    esta funcion hace exactamente lo de siempre.

    LA VALVULA DESCARTA LA TABLA QUE PIERDE, NO LA PAGINA (16-sep-2026, tanda 6 del diseño
    nomos/knowledge/DISENO-uso-real-16sep.md, §4.6 y la bitacora "Experimento de T6").

    QUE SE ROMPIO. La guarda `_pierde_contenido` se evaluaba UNA sola vez, sobre la pagina
    entera: si la reconstruccion completa perdia mas de `MAX_PERDIDA_TOKENS`, se tiraba TODA la
    reconstruccion y la pagina volvia a texto plano. Y alcanza UNA tabla espuria —una region de
    prosa a dos columnas que `find_tables()` cree grilla y que rinde una sola linea— para que
    la pagina entera pierda sus tablas buenas. Medido en PRONAP neurodesarrollo (162 pags): 83
    paginas con tabla detectada, 75 con render util y 54 de esas 75 (el 72 %) terminaban
    PLANAS. El caso del informe de uso real es la p. 28, la continuacion de la tabla de pautas
    del PRUNAPE: la grilla de 24x4 sale entera, pero una segunda "tabla" de 4x5 que es un
    recuadro de texto arrastraba a las dos al piso.

    LA CURA. La perdida se mide POR TABLA: el texto de los bloques que caen dentro de la caja
    contra las filas que esa tabla rindio. La tabla que pierde se descarta —sus bloques quedan
    planos, en su lugar de lectura— y las demas se conservan. La guarda de PAGINA queda como
    ULTIMO RECURSO al final: cubre lo que la medicion por tabla no ve (el bloque que queda
    huerfano porque solapaba dos cajas, el reordenamiento por `y0`).
    """
    def salida(texto):
        return decodificador.aplicar(page, texto) if decodificador is not None else texto

    plano = page.get_text()
    try:
        detectadas = list(page.find_tables().tables)
    except Exception:
        return salida(plano)

    if not detectadas:
        return salida(plano)

    import fitz  # PyMuPDF: perezoso, le costaba ~10 s al arranque en frio de Cloud Run
    bloques = _bloques_de_texto(page)

    reconstruidas = []
    for t in detectadas:
        texto = render_tabla(t)
        if not texto:
            continue
        caja = fitz.Rect(t.bbox)
        if _pierde_la_tabla(_texto_de_la_caja(bloques, caja), texto):
            continue
        reconstruidas.append((caja, texto))

    if not reconstruidas:
        return salida(plano)

    # La prosa se toma por bloques para poder descartar los que son la tabla — si
    # no, el contenido quedaria duplicado: una vez aplanado y otra reconstruido.
    cajas = [caja for caja, _ in reconstruidas]
    piezas = [(rect.y0, rect.x0, txt) for rect, txt, _ in bloques
              if not _dentro_de_alguna(rect, cajas)]

    piezas.extend((caja.y0, caja.x0, texto) for caja, texto in reconstruidas)
    piezas.sort(key=lambda p: (round(p[0], 1), p[1]))
    armado = "\n".join(p[2] for p in piezas)

    return salida(plano if _pierde_contenido(plano, armado) else armado)


def _cerca_de_un_borde(page_start, total_pages) -> bool:
    """El chunk esta en el ultimo 5 % del libro o en sus paginas preliminares?

    ES UN REFUERZO Y NUNCA UNA CONDICION (16-sep-2026): un chunk no es indice por estar al
    final, y por eso esta funcion solo baja el umbral de densidad en `_es_indice`. Sin las dos
    paginas no dice nada: el material sin paginas (DOCX, HTML, EPUB — `pipeline/conversion.py`
    las emite en None a proposito) simplemente no tiene refuerzo posicional.
    """
    if not page_start or not total_pages or total_pages <= 0:
        return False
    if page_start <= MAX_PAGINAS_PRELIMINARES:
        return True
    return page_start >= total_pages * (1 - FRACCION_FINAL_DEL_LIBRO)


def _es_indice(text: str, palabras: list, page_start=None, total_pages=None) -> bool:
    """Firma de INDICE (analitico de tratado o sumario del principio), sobre palabras.

    Las CINCO condiciones van JUNTAS, y ese es todo el diseño de esta regla (§4.2 del diseño
    del 16-sep-2026). La leccion esta pagada: el 13-sep la forma corta del CIE-10, tomada
    sola, marco como indice 30 chunks de contenido clinico real. Aca:

      1. DENSIDAD de pares "termino, pagina" (o de entradas de sumario): >= 12 por cada 280
         palabras, y >= 12 en absoluto. Con refuerzo —"(Cont.)", "Vease", o estar en un borde
         del libro— baja a 8, nunca a cero.
      2. COBERTURA: las entradas son >= 35 % del texto. Un indice ES sus entradas; un cuadro
         clinico las tiene adentro de otra cosa. Es la condicion que separa de verdad.
      3. Un QUINTO de los tokens son numeros de pagina sueltos.
      4. Casi ningun FIN DE ORACION: un indice no tiene oraciones.
      5. Casi ningun VERBO frecuente: no tiene predicados.

    Una tabla de dosis reconstruida cumple (3) y ninguna de las otras; un cuadro clinico con
    cifras cumple (1) y (3) y falla (2); una lista de referencias falla (1), (2) y (4).
    """
    if not palabras:
        return False
    entradas = PAR_TERMINO_PAGINA.findall(text) + ENTRADA_INDICE_GENERAL.findall(text)
    pares = len(entradas)
    if not pares:
        return False
    if sum(len(e) for e in entradas) / max(len(text), 1) < MIN_COBERTURA_INDICE:
        return False

    numericos = sum(1 for p in palabras if TOKEN_NUMERICO.match(p))
    if numericos / len(palabras) < MIN_RATIO_NUMERICO_INDICE:
        return False

    puntos = sum(1 for p in palabras if FIN_DE_ORACION.search(p) and not SOLO_NUMERACION.match(p))
    if puntos / len(palabras) > MAX_PUNTOS_POR_PALABRA_INDICE:
        return False

    verbos = sum(1 for p in palabras if p.strip(".,;:()").lower() in VERBOS_FRECUENTES)
    if verbos / len(palabras) > MAX_VERBOS_POR_PALABRA_INDICE:
        return False

    refuerzo = bool(REFUERZO_INDICE.search(text)) or _cerca_de_un_borde(page_start, total_pages)
    minimo = MIN_PARES_INDICE_CON_REFUERZO if refuerzo else MIN_PARES_INDICE
    densidad = pares / len(palabras) * PALABRAS_DE_REFERENCIA_INDICE
    return pares >= minimo and densidad >= minimo


def classify_content_type(text: str, page_start=None, total_pages=None) -> str:
    """Clasifica el tipo de contenido de un chunk.

    Args:
        text: el texto del chunk, tal como lo arma el chunker.
        page_start: pagina donde empieza el chunk (opcional). None en el material sin
            paginas y en cualquier llamador viejo.
        total_pages: paginas del documento (opcional). Las dos juntas son un REFUERZO
            posicional para la regla de indice, nunca una condicion (ver `_cerca_de_un_borde`).

    OJO: generate_chunks_v2 rearma el chunk con " ".join(palabras), asi que el
    texto que llega aca NO tiene saltos de linea. Todo criterio que cuente lineas
    ve una sola linea y no se dispara nunca. Por eso las tablas se reconocen por
    la firma que deja render_tabla, no por estructura de lineas.

    LA RAMA `lista` SE ELIMINO EL 16-sep-2026, y no se reemplazo. Contaba lineas que empiezan
    con viñeta o numeracion y pedia >= 3 de ellas; con el texto en UNA sola linea la cuenta
    nunca pasaba de 1, asi que la rama no corrio NUNCA desde que existe el chunker v2 (el
    diseño §4.2 la llama "codigo muerto"). Reescribirla sobre palabras habria reetiquetado
    miles de chunks del corpus ya cargado, y `tipo_contenido` forma parte del TEXTO CANONICO
    del embedding (`pipeline/embeddings.build_embedding_text` antepone "Tipo: X"): cambiar la
    etiqueta obliga a re-embeber —que se paga— para un tipo que el retrieval NO excluye, o
    sea sin ningun efecto sobre lo que se recupera. `lista` sigue en el enum de `chunk/v1`
    porque hay chunks en el grafo que la llevan; el parser ya no la produce.
    """
    if not text.strip():
        return "body"
    palabras = text.split()

    # Lista de referencias bibliograficas: ver NO_CONTENIDO y CITA_BIBLIOGRAFICA arriba. Va PRIMERO
    # porque una lista de citas tambien tiene la firma de "lista" y a veces la de "tabla".
    citas = CITA_BIBLIOGRAFICA.findall(text)
    if len(citas) >= MIN_CITAS_REFERENCIAS:
        primera = CITA_BIBLIOGRAFICA.search(text).start() / max(len(text), 1)
        if primera < INICIO_MAX_REFERENCIAS:
            return "referencias"

    # Tabla o indice de codigos de clasificacion: ver CODIGO_CLASIFICACION arriba.
    if len(CODIGO_CLASIFICACION.findall(text)) >= MIN_CODIGOS_INDICE:
        return "indice"

    # Indice analitico de tratado o sumario de preliminares (16-sep-2026): ver `_es_indice`.
    # Va ANTES de `tabla` porque un indice a dos columnas puede dejar alguna firma de tabla.
    if _es_indice(text, palabras, page_start, total_pages):
        return "indice"

    # Tablas: filas emitidas por render_tabla -> "valor (COLUMNA); valor (COLUMNA)."
    if len(re.findall(r'\([^()]{2,40}\);', text)) >= 3:
        return "tabla"

    # Definiciones: empieza con patrones típicos
    first_100 = text[:200].lower()
    if any(p in first_100 for p in ['concepto', 'definición', 'se define como', 'se denomina', 'es la']):
        return "definicion"

    return "body"


# ─────────────────────────────────────────────────────────────────────────────────────────
# LA FRONTERA DE NO_CONTENIDO SE DETECTA POR LINEAS (18-sep-2026, hallazgo V4 del banco,
# `docs/DISENO-deudas-del-banco-18sep.md` §1).
#
# POR QUE. `classify_content_type` etiqueta el CHUNK ya cortado, y el chunker cortaba por tamaño
# sin mirar que habia adentro: un chunk `indice` arrastraba las ultimas filas de la tabla de dosis
# (contenido, perdido para el retrieval porque `indice` esta en NO_CONTENIDO) y el arranque de las
# referencias; y las 1.795 "listas mixtas" medidas el 13-sep --prosa clinica con la bibliografia
# del capitulo colgada al final-- eran exactamente esto: la nota de ese dia decia "esas se
# arreglan cortando la seccion en su frontera, en el parseo; no aca".
#
# COMO. Antes de aplanar las paginas en palabras, cada LINEA recibe una naturaleza: `contenido`,
# `referencias` o `indice`. Se usan las MISMAS firmas duras que ya clasifican chunks
# (`CITA_BIBLIOGRAFICA`, `CODIGO_CLASIFICACION`) y los MISMOS minimos absolutos
# (`MIN_CITAS_REFERENCIAS`, `MIN_CODIGOS_INDICE`): una corrida de lineas con firma, con huecos de
# a lo sumo `HUECO_MAX_*` lineas (una cita parte en dos o tres lineas y la firma "2018;391:1023"
# esta al final), y el TOTAL de firmas de la corrida tiene que llegar al minimo del chunk. Un
# parrafo con tres citas no llega y no abre nada: sigue siendo contenido, que es lo que es.
#
# LO QUE NO ENTRA, a proposito: la tabla (`tabla` es contenido; cortar en su borde daria chunks de
# quince palabras embebidos solos) y el indice analitico (`_es_indice`: firma blanda y posicional,
# en los tratados ocupa paginas enteras y la etiqueta por chunk lo resuelve; entra el dia que una
# medicion sobre meneghello lo pida). `generate_chunks_v2` es quien consume la naturaleza: no deja
# que un chunk --ni un `:ParentChunk`-- cruce la frontera, no solapa a traves de ella y le pone a
# los chunks de un bloque el tipo del bloque.
# ─────────────────────────────────────────────────────────────────────────────────────────
NATURALEZA_CONTENIDO = "contenido"

# El encabezado que abre una lista de referencias, si esta en las lineas previas a la primera
# firma: "Bibliografia", "Referencias bibliograficas", "Lecturas recomendadas", "Apendice 2.
# Referencias". Linea CORTA (un titulo, no una oracion que empieza con "Referencias a la ley...").
ENCABEZADO_REFERENCIAS = re.compile(
    r"(?i)^\W*(?:ap[eé]ndice\s+\w+[.:]?\s*)?"
    r"(?:bibliograf[ií]a|referencias?|lecturas\s+recomendadas|bibliography|references)\b[^.]{0,40}$")
MAX_LARGO_ENCABEZADO_REFERENCIAS = 60
LINEAS_PREVIAS_ENCABEZADO = 3
# Lo que una cita puede dejar DESPUES de su firma, en linea aparte: el DOI, el PMID, la URL.
COLA_REFERENCIAS = re.compile(r"(?i)^(?:doi\b|pmid\b|https?://|www\.|disponible\s+en\b)")
# Huecos admitidos DENTRO de una corrida. Dos para las citas (autores + titulo + revista suelen
# ocupar dos lineas sin firma antes de la linea con el año;volumen:pagina); uno para los codigos
# (cada linea de una tabla CIE-10 lleva codigos, y una linea de prosa entre medio es una
# aclaracion, no un cambio de bloque).
HUECO_MAX_REFERENCIAS = 2
HUECO_MAX_INDICE = 1


def _corridas(firmas: list, hueco_max: int, minimo: int) -> list:
    """Intervalos [a, b] (inclusive) de lineas: arrancan y terminan en una linea con firma, admiten
    hasta `hueco_max` lineas seguidas sin firma adentro, y suman al menos `minimo` firmas."""
    salida = []
    n = len(firmas)
    i = 0
    while i < n:
        if firmas[i] <= 0:
            i += 1
            continue
        a = b = i
        total = firmas[i]
        hueco = 0
        j = i + 1
        while j < n:
            if firmas[j] > 0:
                b = j
                total += firmas[j]
                hueco = 0
            else:
                hueco += 1
                if hueco > hueco_max:
                    break
            j += 1
        if total >= minimo:
            salida.append((a, b))
        i = b + 1
    return salida


def naturaleza_de_tipo(tipo: str) -> str:
    """La naturaleza de un chunk a partir de su `tipo_contenido`: `contenido`, `referencias` o
    `indice`. Es la vara UNICA de "esto es otro bloque".

    Existe porque la pregunta "estos dos chunks son del mismo bloque?" se hace en varios lados
    (los tests de la frontera, y manana cualquier consumidor que agrupe) y la respuesta facil
    --"uno esta en NO_CONTENIDO y el otro no"-- se equivoca justo en el par `indice` /
    `referencias`: los dos son NO_CONTENIDO y son DOS bloques distintos, entre los que tampoco
    hay solape.
    """
    return tipo if tipo in NO_CONTENIDO else NATURALEZA_CONTENIDO


def naturaleza_por_linea(lineas: list) -> list:
    """Una naturaleza por linea: `contenido`, `referencias` o `indice`. Pura: solo mira el texto.

    Las referencias ganan sobre los codigos si una corrida cae adentro de otra (una lista de
    citas puede nombrar un codigo; una tabla de codigos no cita nada).
    """
    n = len(lineas)
    naturaleza = [NATURALEZA_CONTENIDO] * n
    citas = [len(CITA_BIBLIOGRAFICA.findall(linea)) for linea in lineas]
    for a, b in _corridas(citas, HUECO_MAX_REFERENCIAS, MIN_CITAS_REFERENCIAS):
        # El encabezado, si esta cerca, es del bloque: arranca ahi y se lleva las lineas de
        # autores y titulo de la primera cita, que no tienen firma propia.
        vistas = 0
        k = a - 1
        while k >= 0 and vistas < LINEAS_PREVIAS_ENCABEZADO:
            linea = lineas[k].strip()
            if linea:
                vistas += 1
                if (len(linea) <= MAX_LARGO_ENCABEZADO_REFERENCIAS
                        and ENCABEZADO_REFERENCIAS.match(linea)):
                    a = k
                    break
            k -= 1
        while b + 1 < n and COLA_REFERENCIAS.match(lineas[b + 1].strip()):
            b += 1
        for k in range(a, b + 1):
            naturaleza[k] = "referencias"
    codigos = [len(CODIGO_CLASIFICACION.findall(linea)) for linea in lineas]
    for a, b in _corridas(codigos, HUECO_MAX_INDICE, MIN_CODIGOS_INDICE):
        for k in range(a, b + 1):
            if naturaleza[k] == NATURALEZA_CONTENIDO:
                naturaleza[k] = "indice"
    return naturaleza


def ratio_palabras_sin_vocales(text: str) -> float:
    """Fraccion de palabras largas SIN ninguna vocal: la firma del CID desplazado.

    LA FUNCION CANONICA (16-sep-2026). Vivia en `backfill_books.py::_ratio_sin_vocales`, que
    es un script de una corrida; ahora vive en el pipeline y el script la importa de aca. Dos
    detectores de lo mismo en dos archivos divergen en silencio, que es la doctrina de la casa
    y el motivo por el que existe este modulo.

    Con la fuente embebida sin mapear, el texto sale con las letras desplazadas y produce
    palabras largas sin una sola vocal: "ODV DFWLYLGDGHV GLDULDV" era "las actividades
    diarias". En español normal esto es ~0 (alguna sigla suelta). La firma de letras sueltas
    NO lo ve, porque las palabras son largas. Detectado el 3-ago-2026 en
    sina-up11-psicooncologia-pediatrica y en los titulos de sanguinetti-semiologia.
    """
    palabras = [w for w in re.findall(r"[^\W\d_]+", text, re.UNICODE)
                if len(w) >= LARGO_PALABRA_SIN_VOCALES]
    if len(palabras) < MIN_PALABRAS_LARGAS_CALIDAD:
        return 0.0
    return sum(1 for w in palabras if not (set(w.lower()) & VOCALES)) / len(palabras)


def ratio_letras_sueltas(text: str) -> float:
    """Fraccion de tokens que son UNA sola letra: la firma del PDF extraido caracter a caracter.

    Sintoma: "e v e n l ar g e v ess els". Corpus sano: 3-9 %; el libro que lo destapo el
    29-jul-2026 (diagnostic-pathology-gynecological-3ed) media 63,7 %.

    SOLO TOKENS DE UNA LETRA, y esto se pago el 3-ago-2026: contar cualquier token de un
    caracter daba falsos positivos en las tablas numericas, donde '<', '>', '•' y las unidades
    ('g', 'L') son tokens legitimos — los valores hematimetricos del PRONAP daban 21,2 % con
    el texto intacto, y marcarlos degradados deprioritizaba justo las tablas de referencia
    que uno quiere.
    """
    tokens = re.findall(r"\S+", text)
    if len(tokens) < MIN_TOKENS_CALIDAD:
        return 0.0
    return sum(1 for t in tokens if len(t) == 1 and t.isalpha()) / len(tokens)


def ratio_letras_espaciadas(text: str) -> float:
    """Fraccion de tokens que estan en una CORRIDA de letras sueltas que deletrea una palabra.

    LA MITAD DURA DE `ratio_letras_sueltas` (16-sep-2026, dry-run contra el grafo vivo; ver
    `FIRMAS_DURAS`). Sintoma que caza: el PDF extraido caracter a caracter —"l a i n s u f i c
    i e n c i a", "e v e n l ar g e v ess els"— que el 29-jul-2026 destapo
    diagnostic-pathology-gynecological-3ed con 63,7 % de tokens de una letra.

    Lo que NO caza, y por eso existe separada de las letras sueltas: los marcadores A/B/C de un
    epigrafe de figura, la columna de "x" de una tabla de vacunas y los simbolos sueltos de una
    formula quimica. Las tres condiciones de la corrida —largo, letras distintas y minusculas—
    y su porque estan en `MIN_CORRIDA_ESPACIADA`.
    """
    tokens = re.findall(r"\S+", text)
    if len(tokens) < MIN_TOKENS_CALIDAD:
        return 0.0

    def es_letra(token: str) -> bool:
        return len(token) == 1 and token.isalpha()

    def cuenta(corrida: list) -> bool:
        if len(corrida) < MIN_CORRIDA_ESPACIADA:
            return False
        if len({c.lower() for c in corrida}) < MIN_LETRAS_DISTINTAS_ESPACIADA:
            return False
        return sum(1 for c in corrida if c.islower()) * 2 > len(corrida)

    espaciados = 0
    corrida = []
    for token in tokens + [""]:      # el centinela cierra la ultima corrida
        if es_letra(token):
            corrida.append(token)
            continue
        if cuenta(corrida):
            espaciados += len(corrida)
        corrida = []
    return espaciados / len(tokens)


def ratio_no_alfabetico(text: str) -> float:
    """Fraccion de caracteres visibles que no son letras (digitos, simbolos, puntuacion)."""
    visibles = [c for c in text if not c.isspace()]
    if not visibles:
        return 0.0
    return sum(1 for c in visibles if not c.isalpha()) / len(visibles)


def _escalar(ratio: float, referencia: float, piso: float = 0.0) -> float:
    """Una firma cruda -> 0-1, para que las seis se puedan comparar entre si.

    `referencia` es la proporcion a la que esa firma SOLA ya significa "esto es basura", y
    `piso` la que se tolera sin contar nada (solo la usa la proporcion no alfabetica, donde una
    tabla de dosis es legitima). Sin esta normalizacion, la firma de los bytes de control
    quedaba debajo del umbral justo en el caso que vino a cazar.
    """
    if ratio <= piso or referencia <= piso:
        return 0.0
    return min(1.0, (ratio - piso) / (referencia - piso))


def firmas_de_calidad(text: str) -> dict:
    """Las siete firmas, ya NORMALIZADAS a una escala comun (0-1). Ver los umbrales arriba.

    Se devuelven todas —y no solo la peor— porque el diagnostico necesita saber CUAL disparo:
    un chunk 'corrupta' por bytes de control se cura re-ingestando el PDF con OCR, y uno
    'corrupta' por letras espaciadas, re-extrayendo con otro extractor. Es la misma razon por la
    que `admin-unit/v1` publica el estado del embedding y no un booleano.

    CINCO SON DURAS Y DOS BLANDAS (`FIRMAS_DURAS` / `FIRMAS_BLANDAS`, 16-sep-2026): las blandas
    solas no alcanzan para decir `corrupta`. Esta funcion no lo aplica —devuelve la medicion
    cruda—; lo aplica `calidad_de_texto`, que es quien pone la etiqueta.
    """
    largo = max(len(text), 1)
    tokens = max(len(re.findall(r"\S+", text)), 1)
    return {
        "control": _escalar(len(BYTES_DE_CONTROL.findall(text)) / largo, REFERENCIA_CONTROL),
        "reemplazo": _escalar(text.count(CARACTER_DE_REEMPLAZO) / largo, REFERENCIA_REEMPLAZO),
        "cid": _escalar(len(CID_LITERAL.findall(text)) / tokens, REFERENCIA_CID),
        "letras_sueltas": _escalar(ratio_letras_sueltas(text), REFERENCIA_LETRAS_SUELTAS),
        "espaciadas": _escalar(ratio_letras_espaciadas(text), REFERENCIA_ESPACIADAS),
        "sin_vocales": _escalar(ratio_palabras_sin_vocales(text), REFERENCIA_SIN_VOCALES),
        "no_alfabetico": _escalar(ratio_no_alfabetico(text), REFERENCIA_NO_ALFABETICO,
                                  PISO_NO_ALFABETICO),
    }


def calidad_de_texto(text: str) -> tuple:
    """(etiqueta, valor) de la calidad de UN chunk: `ok` | `dudosa` | `corrupta` y 0-1.

    El valor es la PEOR de las firmas (`firmas_de_calidad`), ya normalizadas para que se
    puedan comparar entre si. Es pura: recibe texto y devuelve una tupla — sin grafo, sin red y
    sin el libro al que pertenece el chunk, que es lo que permite correrla en el parseo y otra
    vez en un backfill.

    `CORRUPTA` PIDE UNA FIRMA DURA (16-sep-2026, dry-run contra el grafo vivo; ver
    `FIRMAS_DURAS` arriba para la medicion). Las dos firmas blandas —letras sueltas y
    proporcion no alfabetica— describen una tabla numerica o un epigrafe de figura igual de
    bien que un texto roto, asi que solas techan en `dudosa`. Para decir `corrupta` tiene que
    cruzar el umbral una firma que signifique "la decodificacion fallo".

    MARCA, NO FRENA: ningun llamador de esta funcion excluye nada del corpus por lo que
    devuelva. La etiqueta viaja hasta el resultado para que quien lee sepa que tiene delante.
    """
    if not text or not text.strip():
        return ("ok", 0.0)
    firmas = firmas_de_calidad(text)
    valor = max(firmas.values())
    dura = max(firmas[nombre] for nombre in FIRMAS_DURAS)
    if valor > UMBRAL_CALIDAD_CORRUPTA and dura > UMBRAL_CALIDAD_CORRUPTA:
        return ("corrupta", round(valor, 3))
    if valor > UMBRAL_CALIDAD_DUDOSA:
        return ("dudosa", round(valor, 3))
    return ("ok", round(valor, 3))


# Letras sueltas separadas por un espacio: al menos 2 pares "letra espacio" seguidos de una
# letra final. Compilado al importar porque `_clean_spaced_text` corre por LINEA de titulo.
LETRA_SUELTA = r'[A-ZÁÉÍÓÚÑa-záéíóúñ]'
RE_TEXTO_ESPACIADO = re.compile(rf'({LETRA_SUELTA} ){{2,}}{LETRA_SUELTA}')
_RE_ES_LETRA = re.compile(LETRA_SUELTA)

# Cuantas letras de la corrida tienen que estar SUELTAS (sin otra letra pegada ni antes ni
# despues) para tratarla como texto espaciado. DOS, y el numero es el hallazgo P11 de la QA
# (13-sep-2026): el patron de arriba tambien matchea "ultima letra de una palabra + palabra
# española de UNA letra + primera letra de la siguiente", y por eso
# "Capítulo 2. Fisiopatología y mecanismos" salia "Capítulo 2. Fisiopatologíaymecanismos" en
# el titulo de capitulo de todo libro con una "y" (o una "o", una "u", una "a", una "e") en el
# titulo. En esa corrida -"a y m"- hay UNA letra suelta, la "y"; en "A R T E" (de "PA R T E")
# hay TRES, y por eso se compacta junto con la "A" pegada a la "P" y da "PARTE".
#
# LO QUE SIGUE SIN RESOLVER, y es una ambiguedad real: "Vitaminas A y D en el adulto" colapsa
# igual ("VitaminasAyDen"), porque son tres letras sueltas seguidas. Sin diccionario no hay
# forma de distinguirlo mirando solo los caracteres; un humano tampoco puede.
MIN_LETRAS_SUELTAS = 2


def _clean_spaced_text(text: str) -> str:
    """Reconstruye texto con letras espaciadas del PDF.

    "PA R T E 3 : A M E T R O P Í A S" → "PARTE 3 : AMETROPÍAS"
    "B Á S I C O" → "BÁSICO"

    Una palabra española de una sola letra entre palabras normales NO es texto espaciado:
    "Fisiopatología y mecanismos" queda igual (ver MIN_LETRAS_SUELTAS).
    """
    def collapse_spaced(match):
        crudo = match.group(0)
        # La corrida es "L L L ...": una letra cada dos caracteres.
        sueltas = (len(crudo) + 1) // 2
        inicio, fin = match.start(), match.end()
        if inicio > 0 and _RE_ES_LETRA.match(text[inicio - 1]):
            sueltas -= 1   # la primera letra es la cola de una palabra normal
        if fin < len(text) and _RE_ES_LETRA.match(text[fin]):
            sueltas -= 1   # la ultima es la cabeza de la palabra siguiente
        if sueltas < MIN_LETRAS_SUELTAS:
            return crudo
        return crudo.replace(' ', '')

    return RE_TEXTO_ESPACIADO.sub(collapse_spaced, text)


def detect_structure(pages: list, libro_id: str, estrategia: Estrategia = POR_DEFECTO) -> list:
    """Detecta títulos de capítulo y sección en las páginas, POR POSICIÓN.

    Args:
        pages: Lista de {page, text}
        libro_id: ID del libro para seleccionar patrones

    Returns:
        Lista de {page, text, titulo_capitulo, titulo_seccion}. Una página con más de un
        encabezado devuelve VARIAS entradas con el MISMO `page` (ver abajo).

    EL TÍTULO ES DE LA POSICIÓN, NO DE LA PÁGINA (hallazgo P14 del banco de QA, curado el
    13-sep-2026 por decisión de Iván). Hasta hoy esta función decidía UN
    `titulo_capitulo`/`titulo_seccion` por página y ganaba el ÚLTIMO encabezado que
    aparecía en ella; después `generate_chunks_v2` le ponía a cada palabra la metadata de
    SU PÁGINA. Con dos encabezados en una página —lo normal en un tratado maquetado a dos
    columnas— el chunk que ARRANCA en "Capítulo 1" salía etiquetado "Capítulo 2", y ese
    título viaja en el prefijo que se embebe (`pipeline/embeddings.build_embedding_text`):
    el error no era cosmético, corría el vector.

    LA CURA, Y POR QUÉ ES ASÍ. La página se parte en SEGMENTOS: cada encabezado abre uno
    nuevo, que arranca en la línea del encabezado (el encabezado pertenece a lo que abre) y
    llega hasta el siguiente. Cada segmento sale como una entrada propia con el MISMO
    `page`, así que:
      - `generate_chunks_v2` no cambia ni una línea: ve "páginas" consecutivas con el mismo
        número, y `page_start`/`page_end` siguen saliendo del número de página;
      - el TEXTO no cambia: los segmentos de una página, concatenados, son la página (sólo
        se descartan los que son puro espacio, que no aportan palabras);
      - los ids `{libro}_v2_{NNNNN}` siguen siendo secuenciales y los cortes del chunker,
        los mismos.
    Lo único que cambia son los títulos. Alternativa descartada: mover la decisión al
    chunker (buscar el encabezado más cercano hacia atrás desde cada chunk). Sería duplicar
    el conocimiento de los patrones en dos funciones, y el chunker no ve líneas — trabaja
    sobre una lista plana de palabras.

    UN SEGMENTO SE ABRE SÓLO CUANDO EL TÍTULO SE ASIGNA DE VERDAD: un patrón de sección que
    matchea pero no pasa el filtro de "línea corta sin oración" no abre nada, igual que
    antes no cambiaba el título.
    """
    # Los patrones son del DOMINIO (pipeline/estrategia.py), no de este archivo.
    patterns = estrategia.patrones_de(libro_id)
    cap_patterns = [re.compile(p, re.MULTILINE) for p in patterns["capitulo"]]
    sec_patterns = [re.compile(p, re.MULTILINE) for p in patterns["seccion"]]

    current_capitulo = ""
    current_seccion = ""
    structured = []

    for page_data in pages:
        text = page_data["text"]
        lines = text.split('\n')
        # [linea donde arranca, capitulo vigente, seccion vigente]. El primero hereda los
        # títulos de la página anterior: un párrafo partido entre dos páginas sigue
        # perteneciendo a su capítulo.
        segmentos = [[0, current_capitulo, current_seccion]]

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped or len(line_stripped) < 3:
                continue

            # Detectar capítulo
            is_capitulo = False
            for pat in cap_patterns:
                if pat.match(line_stripped):
                    # Limpiar: reconstruir texto con espacios intercalados
                    # "PA R T E 3 : A M E T R O P Í A S" → "PARTE 3: AMETROPÍAS"
                    clean_cap = _clean_spaced_text(line_stripped)
                    clean_cap = re.sub(r'\s{2,}', ' ', clean_cap).strip()
                    current_capitulo = clean_cap[:120]
                    current_seccion = ""
                    is_capitulo = True
                    break

            # Detectar sección (solo si no fue detectado como capítulo)
            asigno = is_capitulo
            if not is_capitulo:
                for pat in sec_patterns:
                    if pat.match(line_stripped):
                        # Solo aceptar como sección si es línea corta (título, no contenido)
                        # y no contiene punto seguido de más texto (indica oración, no título)
                        if len(line_stripped) < 80 and line_stripped.count('.') <= 2:
                            current_seccion = line_stripped[:100]
                            asigno = True
                            break

            if not asigno:
                continue
            if segmentos[-1][0] == i:
                # El encabezado ES la primera línea del segmento en curso (el caso del
                # encabezado al tope de la página): se le corrigen los títulos en vez de
                # abrir un segmento vacío.
                segmentos[-1][1:] = [current_capitulo, current_seccion]
            else:
                segmentos.append([i, current_capitulo, current_seccion])

        for j, (inicio, cap, sec) in enumerate(segmentos):
            fin = segmentos[j + 1][0] if j + 1 < len(segmentos) else len(lines)
            trozo = '\n'.join(lines[inicio:fin])
            if not trozo.strip():
                continue   # sólo espacio: no aporta palabras y ensuciaría la lista
            structured.append({
                "page": page_data["page"],
                "text": trozo,
                "titulo_capitulo": cap,
                "titulo_seccion": sec,
            })

    return structured


def _find_sentence_boundary(words: list, target_idx: int) -> int:
    """Busca el final de oración más cercano al target_idx.

    Retorna el índice del último word que termina una oración,
    buscando en un rango de ±30 palabras del target.
    """
    search_start = max(0, target_idx - 30)
    search_end = min(len(words), target_idx + 30)

    best = target_idx
    best_dist = 999

    for i in range(search_start, search_end):
        word = words[i]
        if word.endswith(('.', '?', '!', ':')) and not re.match(r'^\d+\.$', word):
            dist = abs(i - target_idx)
            if dist < best_dist:
                best = i
                best_dist = dist

    return best + 1  # Retorna posición después del punto


def generate_chunks_v2(structured_pages: list, libro_id: str,
                       target_size: int | None = None,
                       overlap: int | None = None,
                       estrategia: Estrategia = POR_DEFECTO) -> tuple:
    """Genera child chunks y parent chunks a partir de páginas estructuradas.

    Args:
        structured_pages: Lista de {page, text, titulo_capitulo, titulo_seccion}
        libro_id: ID del libro
        target_size: Palabras objetivo por chunk. None = lo que diga la estrategia.
        overlap: Palabras de solape. None = lo que diga la estrategia.
        estrategia: Estrategia del dominio (tamaños). Default: la de medicina de siempre.

    Returns:
        (child_chunks, parent_chunks)

    `target_size` y `overlap` siguen aceptandose sueltos porque hay llamadores viejos que los
    pasan; cuando vienen, ganan sobre la estrategia (override puntual de una corrida).

    NO_CONTENIDO ES UNA FRONTERA DURA (18-sep-2026, hallazgo V4 del banco; ver el comentario
    largo sobre `naturaleza_por_linea`). Cada palabra llega con la naturaleza de su linea
    (`contenido` / `referencias` / `indice`) y el chunker:
      · nunca deja que un chunk cruce de una naturaleza a otra: el corte se adelanta a la frontera;
      · no solapa A TRAVES de la frontera: el chunk que arranca un bloque arranca en su primera
        palabra, sin las 60 del bloque anterior;
      · le pone a los chunks de un bloque el TIPO del bloque, sin pasar por
        `classify_content_type`: el clasificador de chunk pide >= 8 citas y una cola de 60 palabras
        con tres citas volvia a salir `body`, o sea contenido;
      · funde una cola mas chica que `min_size` con el chunk anterior SOLO si comparten
        naturaleza (antes se fundia siempre, y solo podia pasar al final del documento; ahora
        pasa al final de cada bloque); si no, la cola queda sola --una `referencias` de 60
        palabras es inofensiva, el retrieval la excluye--;
      · los parents tampoco cruzan: `agrupar: padre` devolvia prosa con la bibliografia pegada.
    Un documento sin bloques de NO_CONTENIDO sale EXACTAMENTE igual que antes: los cortes, los
    ids y los parents no cambian (lo fijan los goldens de los formatos sin bloques).
    """
    target_size = estrategia.target_size if target_size is None else target_size
    overlap = estrategia.overlap if overlap is None else overlap
    # Construir un buffer continuo con metadata por página
    all_words = []       # Lista plana de palabras
    word_meta = []       # Metadata por palabra: (page, capitulo, seccion)
    naturaleza = []      # Naturaleza por palabra: la de su linea (ver naturaleza_por_linea)

    # LA NATURALEZA SE DECIDE SOBRE LAS LINEAS DE TODO EL DOCUMENTO, no pagina por pagina: una
    # lista de referencias que cruza de pagina es UNA corrida, y un salto de pagina no la cierra.
    lineas_por_entrada = [sp["text"].split("\n") for sp in structured_pages]
    naturaleza_lineas = naturaleza_por_linea(
        [linea for lineas in lineas_por_entrada for linea in lineas])

    k = 0
    for sp, lineas in zip(structured_pages, lineas_por_entrada, strict=True):
        page = sp["page"]
        cap = sp["titulo_capitulo"]
        sec = sp["titulo_seccion"]
        for linea in lineas:
            nat = naturaleza_lineas[k]
            k += 1
            for w in linea.split():
                all_words.append(w)
                word_meta.append((page, cap, sec))
                naturaleza.append(nat)

    if not all_words:
        return [], []

    n = len(all_words)
    # fin_de_bloque[i]: la primera posicion despues de i con OTRA naturaleza (o n). Es el tope
    # que ningun chunk que arranque en i puede pasar.
    fin_de_bloque = [n] * n
    for i in range(n - 2, -1, -1):
        fin_de_bloque[i] = fin_de_bloque[i + 1] if naturaleza[i + 1] == naturaleza[i] else i + 1

    # LAS PAGINAS DEL DOCUMENTO, para el refuerzo posicional del detector de indice
    # (16-sep-2026). Sale de las paginas que efectivamente llegaron —no de doc.page_count—
    # porque `parse_pdf_v2` descarta las que no tienen texto, y porque el material sin
    # paginas (DOCX, HTML, EPUB) las trae en None: ahi queda None y no hay refuerzo, que es
    # exactamente lo que corresponde.
    total_paginas = max((sp["page"] for sp in structured_pages if sp.get("page")), default=None)

    # Generar child chunks con overlap
    children = []
    naturaleza_de_chunk = []   # paralela a `children`; no viaja en el chunk (no es de chunk/v1)
    pos = 0
    chunk_index = 0

    while pos < n:
        nat = naturaleza[pos]
        limite = fin_de_bloque[pos]
        # Determinar fin del chunk
        end_target = pos + target_size

        if end_target >= limite:
            # Último chunk del bloque (o del documento): tomar todo lo que queda hasta la frontera
            end = limite
        else:
            # Buscar límite de oración cerca del target
            end = _find_sentence_boundary(all_words, end_target)

            # Si el chunk es demasiado grande, forzar corte
            if end - pos > estrategia.max_size:
                end = _find_sentence_boundary(all_words, pos + estrategia.max_size)
                if end - pos > estrategia.max_size + 50:
                    end = pos + estrategia.max_size  # Corte duro como último recurso
            # La oracion mas cercana puede estar del otro lado de la frontera: no se cruza.
            end = max(min(end, limite), pos + 1)

        # LA COLA CORTA QUE NO ES DE NADIE SE VA CON EL BLOQUE SIGUIENTE. Pasa en el encabezado
        # que abre un bloque y que `naturaleza_por_linea` no reconocio --"Apendice 2.
        # Referencias" con la maqueta de una celda de planilla, "<!-- Slide number: 16 -->"--:
        # son cinco o siete palabras de `contenido` encajonadas entre dos bloques, que no se
        # pueden fundir hacia atras (otra naturaleza) y solas serian un chunk de siete palabras
        # con su embedding. Se las absorbe hacia ADELANTE, que ademas es donde pertenecen.
        if (end - pos < estrategia.min_size and limite < n
                and (not children or naturaleza_de_chunk[-1] != nat)):
            nat_siguiente = naturaleza[limite]
            for i in range(pos, limite):
                naturaleza[i] = nat_siguiente
                fin_de_bloque[i] = fin_de_bloque[limite]
            continue

        # Cola demasiado pequeña: merge con el anterior, si es de la misma naturaleza.
        if end - pos < estrategia.min_size and children and naturaleza_de_chunk[-1] == nat:
            prev = children[-1]
            prev["text"] = prev["text"] + " " + " ".join(all_words[pos:end])
            prev["word_count"] = len(prev["text"].split())
            prev["page_end"] = word_meta[end - 1][0]
            # La calidad se vuelve a medir sobre el texto FINAL del chunk fusionado: es una
            # medicion nueva (16-sep-2026) y medirla sobre un prefijo seria mentir barato.
            # `tipo_contenido` NO se recalcula a proposito: es la conducta historica, y
            # cambiarla reetiquetaria el ultimo chunk de cada libro del corpus —con su texto
            # canonico de embedding— por una razon ajena a esta tanda.
            prev["calidad"], prev["calidad_valor"] = calidad_de_texto(prev["text"])
            pos = end   # una cola es siempre el final de un bloque: no hay solape que dar
            continue

        chunk_text = " ".join(all_words[pos:end])
        page_start = word_meta[pos][0]
        page_end = word_meta[end - 1][0]

        # Metadata: tomar la del inicio del chunk (más representativa)
        capitulo = word_meta[pos][1]
        seccion = word_meta[pos][2]

        # Si hay cambio de capítulo/sección dentro del chunk, usar el más nuevo
        for i in range(pos, min(end, pos + 50)):
            if word_meta[i][1] and word_meta[i][1] != capitulo:
                capitulo = word_meta[i][1]
            if word_meta[i][2] and word_meta[i][2] != seccion:
                seccion = word_meta[i][2]

        # El tipo de un chunk de bloque es el del bloque (ver el docstring); el de contenido lo
        # decide el clasificador de siempre.
        if nat == NATURALEZA_CONTENIDO:
            tipo = classify_content_type(chunk_text, page_start, total_paginas)
        else:
            tipo = nat
        # LA CALIDAD SE MIDE ACA Y NO EN UN BACKFILL (16-sep-2026, §1.6 del diseño): es del
        # chunk, se calcula con el texto que se va a guardar y viaja con el. El `:Book` la
        # deriva de sus chunks en `pipeline/carga.py`; no hay una segunda medicion por muestreo.
        calidad, calidad_valor = calidad_de_texto(chunk_text)

        children.append({
            "id": f"{libro_id}_v2_{chunk_index:05d}",
            "libro_id": libro_id,
            "page_start": page_start,
            "page_end": page_end,
            "text": chunk_text,
            "word_count": end - pos,
            "titulo_capitulo": capitulo,
            "titulo_seccion": seccion,
            "tipo_contenido": tipo,
            "calidad": calidad,
            "calidad_valor": calidad_valor,
            "parent_id": None,  # Se asigna después
            "chunk_index": chunk_index,
            "version": 2,
        })
        naturaleza_de_chunk.append(nat)

        chunk_index += 1

        # Avanzar con overlap, salvo en la frontera: el bloque siguiente arranca limpio.
        if end >= limite:
            next_pos = limite
        else:
            next_pos = end - overlap
            if next_pos <= pos:
                next_pos = end  # Evitar loop infinito
        pos = next_pos

    # Generar parent chunks (ventana de parent_window children, sin cruzar la frontera)
    parents = []
    parent_idx = 0
    i = 0

    while i < len(children):
        window_end = min(i + estrategia.parent_window, len(children))
        for j in range(i + 1, window_end):
            if naturaleza_de_chunk[j] != naturaleza_de_chunk[i]:
                window_end = j
                break
        window = children[i:window_end]

        # Concatenar textos de los children (sin overlap duplicado)
        parent_text_parts = []
        for j, child in enumerate(window):
            if j == 0:
                parent_text_parts.append(child["text"])
            else:
                # Quitar overlap del inicio de este child (ya está en el anterior)
                child_words = child["text"].split()
                # El overlap son las últimas ~OVERLAP_SIZE palabras del child anterior
                skip = min(overlap, len(child_words) // 3)  # No skipear más de 1/3
                parent_text_parts.append(" ".join(child_words[skip:]))

        parent_text = " ".join(parent_text_parts)
        parent_words = len(parent_text.split())

        # Si el parent es muy grande, solo tomar lo necesario
        if parent_words > estrategia.max_parent_words:
            parent_text = " ".join(parent_text.split()[:estrategia.max_parent_words])
            parent_words = estrategia.max_parent_words

        parent_id = f"{libro_id}_v2_parent_{parent_idx:05d}"

        parents.append({
            "id": parent_id,
            "libro_id": libro_id,
            "page_start": window[0]["page_start"],
            "page_end": window[-1]["page_end"],
            "text": parent_text,
            "word_count": parent_words,
            "titulo_capitulo": window[0]["titulo_capitulo"],
            "titulo_seccion": window[0]["titulo_seccion"],
            "child_ids": [c["id"] for c in window],
            "version": 2,
        })

        # Asignar parent_id a los children
        for child in window:
            child["parent_id"] = parent_id

        parent_idx += 1
        i = window_end

    return children, parents


def parse_pdf_v2(pdf_path: str, libro_id: str, estrategia: Estrategia = POR_DEFECTO) -> tuple:
    """Parsea un PDF completo a chunks v2.

    Args:
        pdf_path: Ruta al archivo PDF
        libro_id: ID del libro
        estrategia: Como parsear y chunkear este material (pipeline/estrategia.py).
            El default es medicina, o sea el comportamiento historico.

    Returns:
        (children, parents) — listas de dicts
    """
    import fitz  # PyMuPDF: perezoso, le costaba ~10 s al arranque en frio de Cloud Run

    # EL MAPEO DE GLIFOS ROTO, ANTES DE LEER NINGUNA PAGINA (tanda 6, 16-sep-2026; §4.6 del
    # diseño de uso real). Hay PDF cuyo texto sale ilegible aunque el texto este ahi porque la
    # fuente embebida no trae con que traducir sus glifos. `decodificacion.resolver` mira el
    # documento COMPLETO —la clave se deduce del propio PDF: las palabras de su texto sano— y
    # devuelve None cuando no hay nada roto o cuando no llega a la confianza con nombre, que es
    # el caso normal y deja el parseo exactamente como estaba. Se importa aca adentro por la
    # misma razon que `fitz`: el modulo arrastra el arranque si se carga siempre.
    from pipeline import decodificacion

    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    log.info(f"  Parseando {os.path.basename(pdf_path)} ({total_pages} págs)...")

    decodificador = decodificacion.resolver(doc, libro_id)
    if decodificador is not None:
        log.info("  %d fuente(s) con el mapeo roto, decodificadas por forma",
                 len(decodificador.tablas))

    # Extraer texto por página
    pages = []
    for i in range(total_pages):
        page = doc[i]
        text = extraer_texto_pagina(page, decodificador)
        text = clean_text(text)

        if len(text.strip()) > 20:
            pages.append({"page": i + 1, "text": text})

        if (i + 1) % 500 == 0:
            log.info(f"    ... {i + 1}/{total_pages} páginas")

    doc.close()
    log.info(f"  {len(pages)} páginas con texto extraído")

    # SEGUNDA PASADA: el encabezado/pie repetido. Necesita el documento COMPLETO —una línea es
    # maqueta porque aparece en varias páginas, y eso no se ve mirando una sola—, así que no puede
    # vivir en `clean_text`, que corre por página. Va ANTES de `detect_structure` para que el pie no
    # pueda hacerse pasar por un título, y antes del chunker para que no entre a ningún chunk.
    descartar = lineas_repetidas([p["text"] for p in pages])
    if descartar:
        log.info(f"  {len(descartar)} línea(s) de encabezado/pie repetido descartadas")
        for p in pages:
            p["text"] = quitar_lineas(p["text"], descartar)

    # Detectar estructura
    structured = detect_structure(pages, libro_id, estrategia)

    # Generar chunks
    children, parents = generate_chunks_v2(structured, libro_id, estrategia=estrategia)
    log.info(f"  Resultado: {len(children)} children, {len(parents)} parents")

    return children, parents


def normalize_chunks(children: list) -> None:
    """Agrega propiedades normalizadas a los chunks (in-place).

    Agrega: text_busqueda, titulo_seccion_busqueda, titulo_capitulo_busqueda, keywords
    """
    for chunk in children:
        chunk["text_busqueda"] = normalize_for_search(chunk["text"])
        chunk["titulo_seccion_busqueda"] = normalize_for_search(chunk.get("titulo_seccion", ""))
        chunk["titulo_capitulo_busqueda"] = normalize_for_search(chunk.get("titulo_capitulo", ""))

        # Keywords: combinar título + primeras 50 palabras
        kw_parts = []
        if chunk.get("titulo_capitulo"):
            kw_parts.append(chunk["titulo_capitulo"])
        if chunk.get("titulo_seccion"):
            kw_parts.append(chunk["titulo_seccion"])
        first_words = " ".join(chunk["text"].split()[:50])
        kw_parts.append(first_words)
        chunk["keywords"] = normalize_for_search(" ".join(kw_parts))
