"""Decodificador de PDF con el mapeo de glifos roto: recupera el texto sin volver a la imagen.

QUE PROBLEMA CIERRA (16-sep-2026, tanda 6 del diseño
nomos/knowledge/DISENO-uso-real-16sep.md, §4.6 y la bitacora "Experimento de T6").

Hay PDF cuyo texto sale ilegible aunque el texto ESTE ahi: el extractor devuelve el codigo del
glifo en vez del caracter, porque la fuente embebida no trae con que traducirlo. El experimento
de T6 midio dos familias en el corpus, y las dos son de FORMA, no de titular:

  · Sanguinetti (227 pags, 19,2 % del texto): fuentes Type0 `Identity-H` SIN `ToUnicode`. El
    codigo es el indice de glifo, y en el bloque ASCII el orden de glifos es contiguo: sale un
    CORRIMIENTO CONSTANTE (-29 en la Times del libro: `D`->`a`, `H`->`e`, `\\x03`->espacio; -32
    en la Helvetica de versalitas). Los acentos y las ligaduras viven despues de ese bloque y NO
    siguen el corrimiento: se resuelven aparte.
  · Castaño (1.262 pags, 38,4 % del texto): fuentes Type1/CFF CON `ToUnicode`, pero su
    `codespacerange` deja fuera los codigos 0x02-0x1f. No hay corrimiento que sirva —el orden de
    glifos del subset es el de primer uso en el documento—: es una SUSTITUCION monoalfabetica de
    ~190 codigos (`\\x03`->`a`, `\\x05`->`e`, `\\x0b`->`o`, `\\t`->espacio).

La alternativa era OCR (Cloud Vision, US$2,23 los dos libros) y ademas PIERDE la reconstruccion
de tablas (`_legacy/ocr_ingest_v2.py:42` usa `text_detection` y reemplaza `extraer_texto_pagina`).
Esto es gratis, offline y conserva el layout.

COMO. Por FORMA y por PDF, nunca por titulo ni por nombre de fuente — igual que `clean_text`:

  1. QUE ESTA ROTO. Se reusan las firmas de la tanda 2 (`parseo.firmas_de_calidad`), no un
     segundo detector: un span esta roto si alguna de las firmas de decodificacion (bytes de
     control, U+FFFD, `(cid:NN)`) dispara. Dos detectores del mismo sintoma divergen en silencio.
  2. LA CLAVE SALE DEL PROPIO PDF. El lexico con el que se puntua es el de las PALABRAS DEL
     TEXTO SANO DEL MISMO LIBRO (el 81 % de Sanguinetti, el 62 % de Castaño), mas una lista corta
     de palabras frecuentes del español para el caso en que casi todo el libro este roto. No hay
     diccionario externo, ni descarga, ni modelo.
  3. PRIMERO UN CORRIMIENTO. Para cada fuente rota se prueban todos los `k` de un rango y se
     puntua el texto decodificado. Si alguno llega al umbral, su tabla cubre el bloque ASCII.
  4. DESPUES, VOTO CON EL LEXICO. Lo que el corrimiento no explica —y todo, si no hubo
     corrimiento— se resuelve por isomorfismo y voto: cada palabra cifrada propone letras para
     sus codigos desconocidos mirando que palabras del lexico podrian ser, y se asigna el codigo
     cuando un candidato domina. Se itera hasta que no entra ninguno nuevo.
  5. SOLO SI HAY CONFIANZA. La tabla se aplica si la fraccion de palabras reconocidas llega a
     `UMBRAL_CONFIANZA`, y ADEMAS span por span: un span se reemplaza solo si su version
     decodificada se reconoce mejor que la original. Esa segunda puerta es la que permite que una
     fuente tenga spans rotos y spans sanos —Sanguinetti lo hace: la misma Times tiene 96,5 % de
     caracteres bien— sin romper los sanos.
  6. QUEDA ESCRITO. Cada fuente resuelta (o descartada) emite `decodificacion_fuente` en la
     bitacora con el metodo, el tamaño de la tabla y la confianza. Si no se aplica, el texto
     queda como estaba y `parseo.calidad_de_texto` lo marca, que es la doctrina de la tanda 2:
     MARCA, NO FRENA.

NO DEPENDE DE `fitz` AL IMPORTAR (igual que `parseo`): PyMuPDF se importa perezoso. Y las
funciones que resuelven son PURAS —reciben textos y devuelven tablas—, asi que se testean con un
cifrado sintetico sin abrir un PDF.
"""
import logging
import re
from collections import Counter, defaultdict

from pipeline import eventos
from pipeline.parseo import (
    BYTES_DE_CONTROL,
    CARACTER_DE_REEMPLAZO,
    CID_LITERAL,
    firmas_de_calidad,
)

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────────────────
# QUE CUENTA COMO PALABRA. La misma forma que usa `parseo` para sus firmas: letras Unicode, sin
# digitos ni guiones bajos. Tres caracteres es el minimo con el que una palabra dice algo: con
# dos, "de" y "el" hacen que cualquier tabla equivocada puntue.
PALABRA = re.compile(r"[^\W\d_]{3,}", re.UNICODE)

#: Menos texto roto que esto en todo el documento y no se intenta nada: no hay con que resolver
#: una tabla ni con que medir si salio bien. Son ~dos paginas de un libro.
MIN_CHARS_ROTOS = 4000
#: Y el MISMO corte, en la pasada barata previa: cuantas MARCAS de decodificacion rota (bytes de
#: control, U+FFFD, "(cid:NN)") tiene que haber en el texto plano del documento para que valga
#: la pena la pasada cara. Medido: `get_text()` cuesta 0,3 s en las 162 paginas de PRONAP y
#: `get_text("dict")` 1,8 s — seis veces mas—, y la enorme mayoria de los libros del corpus no
#: tiene ni una marca. 500 marcas son ~un parrafo roto: por debajo de eso no hay nada que
#: resolver ni con que medirlo.
MIN_MARCAS_ROTAS = 500
#: Y por FUENTE: una fuente con menos de esto aporta demasiado poco para resolver su propia
#: tabla. Queda sin decodificar y `calidad` la marca (en Sanguinetti son 400 de 141.000
#: caracteres rotos: los titulos de los graficos).
MIN_CHARS_FUENTE = 500
#: Cuantas fuentes se intentan, de mayor a menor texto roto. Un PDF con 40 fuentes tiene 3 o 4
#: con texto y el resto son simbolos y wingdings, donde no hay palabras que reconocer.
MAX_FUENTES = 8

# EL CORRIMIENTO. El rango cubre de sobra los dos casos medidos (-29 y -32 en la notacion del
# PDF, o sea +29 y +32 al decodificar) y cualquier orden de glifos que arranque el bloque ASCII
# en un indice bajo. Se prueba entero: son ~220 pasadas sobre una muestra, milisegundos.
RANGO_CORRIMIENTO = range(-96, 129)
#: Puntaje (caracteres dentro de palabras reconocidas / caracteres del texto) a partir del cual
#: un corrimiento es creible. Medido: los cuatro corrimientos correctos del corpus dan 0,23 a
#: 0,49 y el mejor corrimiento EQUIVOCADO de una fuente con sustitucion da 0,03. El umbral va
#: lejos de los dos.
UMBRAL_CORRIMIENTO = 0.20
#: El corrimiento SOLO explica el bloque ASCII imprimible. Los codigos que caen fuera —acentos,
#: ligaduras, comillas tipograficas— se dejan sin asignar para que los resuelva el voto. Sin
#: este recorte, `y`->chr(150) y `¿`->'Ü' entraban a la tabla como si fueran buenos.
ASCII_MIN, ASCII_MAX = 32, 126

# EL VOTO CON EL LEXICO.
#: Cuantas palabras cifradas distintas se miran por ronda, de la mas frecuente a la menos.
TOPE_PALABRAS_VOTO = 4000
#: Una palabra con mas codigos desconocidos que esto no restringe nada: propone cualquier cosa.
MAX_DESCONOCIDOS = 2
#: Si una palabra cifrada admite mas candidatos que esto, no vota: no distingue.
MAX_CANDIDATOS = 40
#: Un codigo se asigna cuando su candidato mas votado le saca esta ventaja al segundo Y junta
#: al menos `MIN_VOTOS`. Los dos frenan la asignacion temprana equivocada, que despues arrastra.
DOMINANCIA_VOTO = 1.4
MIN_VOTOS = 3.0
#: Cuantas rondas de voto. Cada ronda usa lo asignado en la anterior; en los dos libros medidos
#: converge en 2 o 3 y nunca paso de 6.
MAX_RONDAS = 10
#: Para la semilla por isomorfismo: si el patron de repeticion de la palabra cifrada admite mas
#: de estas palabras del lexico, no dice nada.
MAX_CANDIDATOS_ISOMORFOS = 25

# LA CONFIANZA: fraccion de PALABRAS reconocidas despues de aplicar la tabla.
#: Debajo de esto no se aplica: el texto queda como esta y la calidad lo marca. Medido: las
#: fuentes principales de los dos libros dan 0,72 (Castaño) y 0,66-0,85 (Sanguinetti); una
#: fuente sin resolver da menos de 0,10.
UMBRAL_CONFIANZA = 0.50
#: Y sobre cuantas palabras, como minimo. Sin esto una fuente con 17 palabras daba confianza
#: 1,000 y se aplicaba una tabla de tres codigos.
MIN_PALABRAS_CONFIANZA = 100

#: Cuanto tiene que MEJORAR un span para reemplazarlo. La puerta por span es la que protege al
#: texto sano de una fuente que tambien tiene spans rotos.
MARGEN_SPAN = 0.20

# EL GUARDA DE LA PUNTUACION (ver `codigos_que_no_son_letra`). Un codigo que en >= 85 % de sus
# apariciones viene seguido de un espacio y que empieza palabra en <= 5 % de ellas no es una
# letra. Medido en español: el punto y la coma estan arriba del 0,95 y nunca empiezan palabra;
# la "s" —la letra final mas frecuente— esta en 0,4 y empieza palabra todo el tiempo.
FRACCION_PEGADA_AL_ESPACIO = 0.85
FRACCION_INICIAL_DE_PALABRA = 0.05
#: Con menos apariciones que esto la firma es ruido.
MIN_OCURRENCIAS_NO_LETRA = 30

# LO QUE NO FUNCIONO, anotado para que nadie lo vuelva a intentar (16-sep-2026). Se probo una
# DEPURACION FINAL: sacar de la tabla todo codigo cuyas palabras se reconocieran mucho menos que
# el promedio del libro (< 0,4 del global, con 20 apariciones minimo). La idea era cazar los
# codigos de DIGITOS y de SIGLAS EN MAYUSCULA, a los que el voto les pone una letra porque el
# lexico es de palabras en minuscula ("80 %" salio "rios"). Medido en Castaño: saco 16 codigos,
# la confianza subio de 0,815 a 0,844 —porque se queda con lo facil— y el texto roto del libro
# volvio de 1,7 % a 14,9 %. O sea: tiraba nueve veces mas texto bueno del que arreglaba. Se
# descarto. La limitacion queda: en una sustitucion resuelta por lexico, los digitos y las
# siglas pueden salir como una palabra equivocada.

# ─────────────────────────────────────────────────────────────────────────────────────────
# LAS PALABRAS FRECUENTES DEL ESPAÑOL, EMBEBIDAS. Son el piso del lexico para el caso en que el
# libro este roto casi entero y su texto sano no alcance. Es una lista de FORMA —articulos,
# preposiciones, verbos y conectores, mas el vocabulario clinico y juridico mas comun—, sin un
# solo nombre propio: vale igual para cualquier libro y no hay que tocarla cuando entra uno
# nuevo. No reemplaza al lexico del PDF: se suma.
FRECUENTES_ES = frozenset("""
que con por para una las los del como más pero sobre este esta estos estas entre cuando
todo toda todos todas otro otra otros otras hasta desde donde también puede pueden debe deben
ser son era eran fue fueron sido estar está están estaba estaban tiene tienen tenía tenían
hacer hace hacen hecho haber había habían hay caso casos forma formas parte partes
vez veces modo manera tipo tipos nivel niveles grupo grupos número números año años día días
mes meses hora horas tiempo lugar lugares punto puntos línea líneas cada mismo misma mismos
mismas cual cuales dicha dicho dichos dichas tanto tantas según además entonces luego después
antes durante mientras aunque porque sino solo sólo bien mayor menor mejor peor alto alta
altos altas bajo baja bajos bajas nuevo nueva nuevos nuevas primer primera primero segundo
segunda tercer tercera último última grande grandes pequeño pequeña general generales
posible posibles necesario necesaria importante importantes diferente diferentes distinto
distinta siguiente siguientes anterior anteriores presente presentes total totales
paciente pacientes enfermedad enfermedades tratamiento tratamientos diagnóstico diagnósticos
síntoma síntomas signo signos clínico clínica clínicos clínicas médico médica médicos médicas
sangre corazón pulmón pulmonar renal hepático arterial venoso cardíaco cardiaca respiratorio
respiratoria muscular óseo nervioso digestivo urinario dolor dolores fiebre infección
infecciones inflamación crónica crónico aguda agudo grave graves leve leves riesgo riesgos
causa causas efecto efectos dosis fármaco fármacos medicamento medicamentos terapia terapias
estudio estudios examen exámenes prueba pruebas resultado resultados valor valores normal
normales anormal anormales aumento disminución presencia ausencia función funciones
célula células tejido tejidos órgano órganos sistema sistemas proceso procesos
evaluación seguimiento control controles manejo prevención evolución pronóstico
ley leyes norma normas artículo artículos derecho derechos deber deberes obligación
obligaciones persona personas sujeto sujetos objeto objetos acto actos hecho hechos
proceso procesal juez jueces tribunal tribunales sentencia sentencias prueba pruebas
contrato contratos parte partes acción acciones responsabilidad daño daños
público pública privado privada nacional nacionales estado estados
información datos texto textos capítulo capítulos sección secciones página páginas
figura figuras tabla tablas cuadro cuadros anexo anexos nota notas ejemplo ejemplos
""".split())


# ─────────────────────────────────────────────────────────────────────────────────────────
# FUNCIONES PURAS: reciben texto, devuelven tablas. Se testean sin abrir un PDF.


def es_espacio(caracter: str) -> bool:
    """El caracter es un separador de verdad, y no un codigo de glifo?

    ES UNA TRAMPA PAGA (16-sep-2026): `str.isspace()` de Python dice True para `\\x0b`, `\\x0c` y
    `\\x1c`-`\\x1f`, y en estos PDF esos codigos son GLIFOS (el parentesis, los dos puntos). Al
    reves, en Castaño `\\t` y `\\n` SON glifos (`\\t` es el espacio y `\\n` es la "f"). Asi que el
    unico separador intocable es el espacio de verdad —U+0020 y los espacios Unicode de arriba—;
    todo lo que esta debajo de 0x20 es candidato a codigo.
    """
    o = ord(caracter)
    return o == 32 or (o > 32 and caracter.isspace())


def span_roto(texto: str) -> bool:
    """El span viene con el mapeo roto? Se decide con las firmas de `parseo`, no con otra regla.

    Solo las firmas de DECODIFICACION (bytes de control, U+FFFD, `(cid:NN)`). Las de forma
    tipografica —letras sueltas, proporcion no alfabetica— describen igual de bien una tabla
    numerica, y una tabla numerica no hay que decodificarla.
    """
    firmas = firmas_de_calidad(texto)
    return max(firmas["control"], firmas["reemplazo"], firmas["cid"]) > 0


def marcas_rotas(texto: str) -> int:
    """Cuantas MARCAS de decodificacion rota tiene el texto: bytes de control, U+FFFD, "(cid:NN)".

    Las tres definiciones son las de `parseo` —importadas, no copiadas—, que es la misma regla
    que usa `span_roto`; lo unico que cambia es que aca se cuentan marcas en vez de normalizar a
    una firma, porque este conteo decide si vale la pena la pasada cara sobre el documento.
    """
    return (len(BYTES_DE_CONTROL.findall(texto)) + texto.count(CARACTER_DE_REEMPLAZO)
            + len(CID_LITERAL.findall(texto)))


def construir_lexico(textos_sanos) -> set:
    """Las palabras del texto SANO del propio documento, mas las frecuentes del español."""
    lexico = set(FRECUENTES_ES)
    for texto in textos_sanos:
        for palabra in PALABRA.findall(texto):
            lexico.add(palabra.lower())
    return lexico


def puntaje(texto: str, lexico: set) -> float:
    """Caracteres que caen dentro de palabras reconocidas / caracteres del texto.

    SE MIDE POR CARACTER Y NO POR PALABRA, y esto se pago midiendo: con la fraccion de palabras,
    un corrimiento equivocado que deja el texto en tres palabras sueltas —una de ellas del
    lexico— puntuaba 1,000 y le ganaba al corrimiento correcto. Por caracter, un texto que casi
    no produce palabras puntua casi cero, que es lo que corresponde.
    """
    if not texto:
        return 0.0
    dentro = sum(len(p) for p in PALABRA.findall(texto) if p.lower() in lexico)
    return dentro / len(texto)


def confianza(texto: str, lexico: set) -> tuple:
    """(fraccion de palabras reconocidas, cuantas palabras). Es lo que decide si se aplica."""
    palabras = [p.lower() for p in PALABRA.findall(texto)]
    if not palabras:
        return (0.0, 0)
    return (sum(1 for p in palabras if p in lexico) / len(palabras), len(palabras))


def aplicar_tabla(texto: str, tabla: dict) -> str:
    return "".join(tabla.get(c, c) for c in texto)


def _corrida(caracter: str, k: int) -> str:
    """Un caracter corrido `k`, dejando el espacio de verdad donde esta.

    El espacio NO se corre, y es la misma regla que `tabla_de_corrimiento`: un span roto trae
    espacios de verdad —PyMuPDF los mete entre dos tiradas de glifos— y correrlos convertia el
    separador en un signo ("=" con k=29), que parte palabras y hunde el puntaje del corrimiento
    CORRECTO. Medido el 16-sep-2026: con la muestra sintetica, 0,19 antes y 0,28 despues.
    """
    if es_espacio(caracter):
        return caracter
    destino = ord(caracter) + k
    return chr(destino) if 0 < destino < 0x1000 else caracter


def mejor_corrimiento(muestra: str, lexico: set) -> tuple:
    """(puntaje, k) del corrimiento constante que mejor decodifica la muestra."""
    mejor = (0.0, 0)
    for k in RANGO_CORRIMIENTO:
        if k == 0:
            continue
        p = puntaje("".join(_corrida(c, k) for c in muestra), lexico)
        if p > mejor[0]:
            mejor = (p, k)
    return mejor


def tabla_de_corrimiento(codigos, k: int) -> dict:
    """La tabla de un corrimiento, acotada a lo que el corrimiento explica.

    SOLO EL BLOQUE ASCII (ver `ASCII_MIN`/`ASCII_MAX`) y nunca un espacio de verdad.
    """
    tabla = {}
    for c in codigos:
        if es_espacio(c):
            continue
        destino = ord(c) + k
        if ASCII_MIN <= destino <= ASCII_MAX:
            tabla[c] = chr(destino)
    return tabla


def codigo_separador(textos, tabla: dict):
    """El codigo que hace de espacio: el mas frecuente que no este ya resuelto como letra.

    En un texto en cualquier idioma el caracter mas frecuente es el espacio, y en un texto
    cifrado por sustitucion eso no cambia. Sin el separador no hay palabras que votar.
    """
    cuenta = Counter()
    for texto in textos:
        cuenta.update(texto)
    for codigo, _ in cuenta.most_common():
        if codigo in tabla:
            if tabla[codigo].isspace():
                return codigo
            continue
        return codigo
    return None


def palabras_cifradas(textos, tabla: dict, separador) -> Counter:
    """Las 'palabras' del texto cifrado: se corta por el separador y por lo ya sabido no-letra."""
    cuenta = Counter()
    for texto in textos:
        buffer = []
        for c in texto:
            corta = (c == separador or es_espacio(c)
                     or (c in tabla and not tabla[c].isalpha()))
            if corta:
                if len(buffer) >= 3:
                    cuenta["".join(buffer)] += 1
                buffer = []
            else:
                buffer.append(c)
        if len(buffer) >= 3:
            cuenta["".join(buffer)] += 1
    return cuenta


def codigos_que_no_son_letra(textos, separador, tabla: dict) -> set:
    """Los codigos que NO pueden ser una letra, por donde caen: pegados a un separador.

    POR QUE HACE FALTA (16-sep-2026, medido en Castaño). Sin esto, el punto y la coma —que el
    voto ve como la ultima letra de la palabra— se llevaban una letra: "ingreso." salia
    "ingresos" y "por ello," salia "por ellos". El voto no puede distinguirlos solo, porque
    "ingresos" ES una palabra del lexico; la diferencia esta en la POSICION, no en la palabra.

    LA FIRMA. Un signo de puntuacion viene casi siempre seguido de un espacio y casi nunca
    empieza una palabra. Una letra final frecuente del español —la "s"— tambien va seguida de
    espacio, pero en menos de la mitad de sus apariciones, y empieza palabras a cada rato. Los
    dos umbrales estan lejos de las dos poblaciones.

    QUE SE HACE CON ELLOS: se mapean a un ESPACIO, no a una letra. Se puede demostrar que no
    son una letra; no se puede demostrar CUAL signo son (el punto y la coma tienen exactamente
    la misma firma). Convertirlos en espacio deja las palabras bien y pierde el signo;
    convertirlos en letra, que es lo que pasaba antes, INVENTA texto.
    """
    seguidas = Counter()
    iniciales = Counter()
    total = Counter()
    for texto in textos:
        anterior = None
        for i, c in enumerate(texto):
            if es_espacio(c) or c == separador:
                anterior = c
                continue
            total[c] += 1
            siguiente = texto[i + 1] if i + 1 < len(texto) else None
            if siguiente is None or siguiente == separador or es_espacio(siguiente):
                seguidas[c] += 1
            if anterior is None or anterior == separador or es_espacio(anterior):
                iniciales[c] += 1
            anterior = c
    fuera = set()
    for c, n in total.items():
        if c in tabla or n < MIN_OCURRENCIAS_NO_LETRA:
            continue
        if (seguidas[c] / n >= FRACCION_PEGADA_AL_ESPACIO
                and iniciales[c] / n <= FRACCION_INICIAL_DE_PALABRA):
            fuera.add(c)
    return fuera


def isomorfo(palabra: str) -> tuple:
    """El patron de repeticion de la palabra: "abcadc" -> (0,1,2,0,3,2).

    Es lo unico que se puede saber de una palabra cifrada SIN tabla ninguna, y alcanza: en un
    lexico de 15.000 palabras, un patron de 8 letras con dos repeticiones lo cumplen tres o
    cuatro palabras. De ahi sale la semilla del caso Castaño, donde no hay corrimiento.
    """
    vistos, salida = {}, []
    for c in palabra:
        salida.append(vistos.setdefault(c, len(vistos)))
    return tuple(salida)


def _indices_del_lexico(lexico: set) -> tuple:
    """(por largo, por isomorfismo) — los dos indices que el voto consulta."""
    por_largo = defaultdict(list)
    por_isomorfo = defaultdict(list)
    for palabra in lexico:
        por_largo[len(palabra)].append(palabra)
        por_isomorfo[(len(palabra), isomorfo(palabra))].append(palabra)
    return por_largo, por_isomorfo


def votos_por_isomorfismo(cifradas: Counter, por_isomorfo: dict) -> dict:
    """La semilla cuando no hay corrimiento: cada palabra cifrada vota por su patron."""
    votos = defaultdict(Counter)
    for palabra, frecuencia in cifradas.most_common(TOPE_PALABRAS_VOTO):
        candidatas = por_isomorfo.get((len(palabra), isomorfo(palabra)), ())
        if not candidatas or len(candidatas) > MAX_CANDIDATOS_ISOMORFOS:
            continue
        peso = frecuencia / len(candidatas)
        for candidata in candidatas:
            for codigo, letra in zip(palabra, candidata, strict=True):
                votos[codigo][letra] += peso
    return votos


def votos_por_lexico(cifradas: Counter, tabla: dict, por_largo: dict) -> dict:
    """Cada palabra cifrada con pocos codigos sin resolver propone letras para ellos.

    La palabra se convierte en una expresion regular: lo ya sabido va literal y cada codigo
    desconocido es `(.{1,2})` —uno o DOS caracteres, porque una ligadura (`ﬁ`, `fl`) es un solo
    glifo y dos letras—, con retrovisor para que el mismo codigo de siempre lo mismo dentro de
    la palabra. Se busca contra las palabras del lexico de los largos posibles.
    """
    votos = defaultdict(Counter)
    for palabra, frecuencia in cifradas.most_common(TOPE_PALABRAS_VOTO):
        desconocidos = [c for c in dict.fromkeys(palabra) if c not in tabla]
        if not desconocidos or len(desconocidos) > MAX_DESCONOCIDOS:
            continue
        partes, grupos = [], {}
        for c in palabra:
            if c in tabla:
                partes.append(re.escape(tabla[c]))
            elif c in grupos:
                partes.append(f"\\{grupos[c]}")
            else:
                grupos[c] = len(grupos) + 1
                partes.append("(.{1,2})")
        patron = re.compile("(?i)" + "".join(partes) + r"\Z")
        fijos = sum(len(tabla[c]) for c in palabra if c in tabla)
        libres = sum(1 for c in palabra if c not in tabla)
        encontradas = []
        for largo in range(fijos + libres, fijos + 2 * libres + 1):
            for candidata in por_largo.get(largo, ()):
                encaje = patron.match(candidata)
                if encaje:
                    encontradas.append(encaje)
                    if len(encontradas) > MAX_CANDIDATOS:
                        break
            if len(encontradas) > MAX_CANDIDATOS:
                break
        if not encontradas or len(encontradas) > MAX_CANDIDATOS:
            continue
        peso = frecuencia / len(encontradas)
        for encaje in encontradas:
            for codigo, grupo in grupos.items():
                votos[codigo][encaje.group(grupo)] += peso
    return votos


def asignar_votos(votos: dict, tabla: dict) -> int:
    """Pasa a la tabla los codigos cuyo candidato domina. Devuelve cuantos entraron."""
    nuevos = 0
    for codigo, cuenta in votos.items():
        if codigo in tabla:
            continue
        top = cuenta.most_common(2)
        if not top or top[0][1] < MIN_VOTOS:
            continue
        if len(top) > 1 and top[0][1] < DOMINANCIA_VOTO * top[1][1]:
            continue
        tabla[codigo] = top[0][0]
        nuevos += 1
    return nuevos


def resolver_fuente(textos, lexico: set, indices=None) -> dict:
    """La tabla de UNA fuente rota: corrimiento si sirve, y voto con el lexico para el resto.

    Args:
        textos: los spans ROTOS de esa fuente, en orden.
        lexico: `construir_lexico` del documento.
        indices: `(por_largo, por_isomorfo)`; se calculan si no se pasan (util en los tests).

    Returns:
        `{"tabla": dict, "metodo": "corrimiento"|"sustitucion", "k": int|None,
          "confianza": float, "palabras": int}`. La tabla puede venir vacia.
    """
    por_largo, por_isomorfo = indices if indices else _indices_del_lexico(lexico)
    muestra = " ".join(textos[:600])
    punto, k = mejor_corrimiento(muestra, lexico)

    if punto >= UMBRAL_CORRIMIENTO:
        metodo = "corrimiento"
        tabla = tabla_de_corrimiento({c for t in textos for c in t}, k)
    else:
        metodo, k = "sustitucion", None
        tabla = {}

    separador = codigo_separador(textos, tabla)
    if separador is not None and separador not in tabla:
        tabla[separador] = " "

    # La puntuacion, ANTES de votar: si entra al voto se lleva una letra y la palabra sale mal.
    for codigo in codigos_que_no_son_letra(textos, separador, tabla):
        tabla[codigo] = " "

    cifradas = palabras_cifradas(textos, tabla, separador)
    if not any(v.isalpha() for v in tabla.values()):
        # Ni una letra todavia: no hubo corrimiento que sembrara nada, asi que la semilla sale
        # del isomorfismo — el patron de repeticion de la palabra, que no necesita tabla.
        asignar_votos(votos_por_isomorfismo(cifradas, por_isomorfo), tabla)
        cifradas = palabras_cifradas(textos, tabla, separador)

    for _ in range(MAX_RONDAS):
        if not asignar_votos(votos_por_lexico(cifradas, tabla, por_largo), tabla):
            break
        cifradas = palabras_cifradas(textos, tabla, separador)

    valor, palabras = confianza(aplicar_tabla(muestra, tabla), lexico)
    return {"tabla": tabla, "metodo": metodo, "k": k,
            "confianza": round(valor, 3), "palabras": palabras}


# ─────────────────────────────────────────────────────────────────────────────────────────
# EL OBJETO QUE VIAJA AL PARSEO


class Decodificador:
    """Las tablas resueltas de un documento, y como aplicarlas a una pagina.

    SE APLICA POR SPAN Y CON PUERTA. `aplicar` no traduce la pagina caracter a caracter: busca
    los spans rotos, decodifica cada uno y lo reemplaza en el texto ya armado SOLO si la version
    decodificada se reconoce mejor. Por eso puede convivir con una fuente que tiene 96,5 % de
    caracteres sanos, que es el caso real de Sanguinetti.
    """

    __slots__ = ("tablas", "lexico", "paginas", "informe")

    def __init__(self, tablas: dict, lexico: set, paginas: set, informe: list):
        self.tablas = tablas
        self.lexico = lexico
        self.paginas = paginas          #: numeros de pagina (0-based) con algun span roto
        self.informe = informe          #: una fila por fuente, para la bitacora y los tests

    def decodificar_span(self, fuente: str, texto: str):
        """El span decodificado, o None si no hay que tocarlo.

        La puerta: la version nueva tiene que reconocer algo Y mejorar a la vieja por
        `MARGEN_SPAN`. Un span sano decodificado da basura y no pasa; un span roto que la tabla
        no alcanza a resolver tampoco, y se queda como estaba.
        """
        tabla = self.tablas.get(fuente)
        if not tabla:
            return None
        nuevo = aplicar_tabla(texto, tabla)
        if nuevo == texto:
            return None
        punto_nuevo = puntaje(nuevo, self.lexico)
        if punto_nuevo <= 0:
            return None
        return nuevo if punto_nuevo > puntaje(texto, self.lexico) + MARGEN_SPAN else None

    def aplicar(self, page, texto: str) -> str:
        """Reemplaza en `texto` los spans rotos de `page` por su version decodificada.

        DE UNA SOLA PASADA, con una alternativa de regex ordenada de mas larga a mas corta: si
        se hiciera con `str.replace` encadenado, el resultado de un reemplazo podria contener el
        patron del siguiente y la cadena se comeria a si misma.
        """
        if page.number not in self.paginas:
            return texto
        cambios = {}
        for bloque in page.get_text("dict")["blocks"]:
            for linea in bloque.get("lines", []):
                for span in linea["spans"]:
                    crudo = span["text"]
                    if not crudo.strip() or crudo in cambios or not span_roto(crudo):
                        continue
                    nuevo = self.decodificar_span(span["font"], crudo)
                    if nuevo:
                        cambios[crudo] = nuevo
        if not cambios:
            return texto
        patron = re.compile("|".join(
            re.escape(c) for c in sorted(cambios, key=len, reverse=True)))
        return patron.sub(lambda m: cambios[m.group(0)], texto)


def resolver(doc, libro_id: str = None):
    """Analiza el documento entero y devuelve un `Decodificador`, o None si no hace falta.

    None significa "no hay nada roto que valga la pena", "no se pudo resolver" o "no llego a la
    confianza": en los tres casos el parseo sigue exactamente como antes.

    Cuesta UNA pasada de `get_text("dict")` sobre el documento (4,5 s en las 1.262 paginas de
    Castaño) y se hace una sola vez por libro, no por pagina.
    """
    # LA PASADA BARATA PRIMERO. `get_text()` es seis veces mas rapido que `get_text("dict")`, y
    # la enorme mayoria de los libros no tiene ni una marca: no hay por que pagar la pasada cara
    # para descubrirlo. Medido en PRONAP (162 pags sanas): 0,3 s contra 1,8 s.
    marcas = 0
    for numero in range(doc.page_count):
        marcas += marcas_rotas(doc[numero].get_text())
        if marcas >= MIN_MARCAS_ROTAS:
            break
    if marcas < MIN_MARCAS_ROTAS:
        return None

    rotos, sanos, paginas = defaultdict(list), [], set()
    for numero in range(doc.page_count):
        for bloque in doc[numero].get_text("dict")["blocks"]:
            for linea in bloque.get("lines", []):
                for span in linea["spans"]:
                    texto = span["text"]
                    if not texto.strip():
                        continue
                    if span_roto(texto):
                        rotos[span["font"]].append(texto)
                        paginas.add(numero)
                    else:
                        sanos.append(texto)

    total_roto = sum(len(t) for textos in rotos.values() for t in textos)
    if total_roto < MIN_CHARS_ROTOS:
        return None

    lexico = construir_lexico(sanos)
    indices = _indices_del_lexico(lexico)
    tablas, informe = {}, []
    porte = sorted(rotos.items(), key=lambda par: -sum(len(t) for t in par[1]))
    for fuente, textos in porte[:MAX_FUENTES]:
        chars = sum(len(t) for t in textos)
        if chars < MIN_CHARS_FUENTE:
            continue
        salida = resolver_fuente(textos, lexico, indices)
        aplicada = (salida["confianza"] >= UMBRAL_CONFIANZA
                    and salida["palabras"] >= MIN_PALABRAS_CONFIANZA)
        if aplicada:
            tablas[fuente] = salida["tabla"]
        fila = {"fuente": fuente, "metodo": salida["metodo"], "k": salida["k"],
                "codigos": len(salida["tabla"]), "confianza": salida["confianza"],
                "chars": chars, "estado": "aplicada" if aplicada else "descartada"}
        informe.append(fila)
        eventos.emitir(log, "decodificacion_fuente", libro_id=libro_id, **fila)

    if not tablas:
        return None
    return Decodificador(tablas, lexico, paginas, informe)
