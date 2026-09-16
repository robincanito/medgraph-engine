"""pipeline/parseo.py — el parseo y el chunking canonicos del engine.

QUE ES ESTE ARCHIVO. `pipeline/` es un ESPEJO byte a byte del pipeline privado de MedGraph
(ver README, "The pipeline is a mirror"): el codigo no se edita aca. Lo que si vive aca son
los tests de REGRESION de las curas que se replicaron, para que un espejo desactualizado o
una copia mal pegada se note en CI del lado publico tambien.

Cuatro capas:
  1. funciones puras (limpieza, normalizacion, clasificacion, corte por oracion);
  2. el chunker sobre paginas sinteticas: tamanos, solape, padres, metadata;
  3. las dos curas del 13-sep-2026 que llegaron al engine: P11 (`_clean_spaced_text` y la
     "y" espanola) y la decision D (el titulo se asigna por POSICION, no por pagina);
  4. un PDF generado en el test (PyMuPDF, texto inventado, cero copyright) parseado de punta
     a punta hasta el texto canonico que se embebe.

Cero red, cero grafo, cero llamadas pagas.
"""
import ast
import re
from pathlib import Path

import pytest

from pipeline import decodificacion, embeddings, parseo

RAIZ = Path(__file__).resolve().parent.parent

PARRAFO = (
    "La insuficiencia cardiaca es un sindrome clinico en el que el corazon no puede "
    "bombear la sangre con la eficacia necesaria. Los pacientes presentan disnea, edemas "
    "y fatiga. El tratamiento se basa en diureticos, inhibidores de la enzima convertidora "
    "y betabloqueantes, con control periodico de la funcion renal y del potasio. "
)


def _paginas(n_paginas=6, palabras_por_pagina=400):
    """Paginas estructuradas sinteticas con capitulo y seccion (como las produce
    `detect_structure`, que es lo que el chunker recibe de verdad)."""
    texto = (PARRAFO * 20).split()
    out = []
    for p in range(1, n_paginas + 1):
        cuerpo = " ".join(texto[:palabras_por_pagina])
        out.append({"page": p, "text": cuerpo,
                    "titulo_capitulo": f"Capitulo {1 + (p - 1) // 3}",
                    "titulo_seccion": f"Seccion {p}"})
    return out


class TestFuncionesPuras:
    def test_normalize_for_search_quita_acentos_y_baja(self):
        assert parseo.normalize_for_search("Corazón Ñandú") == "corazon nandu"
        assert parseo.normalize_for_search("") == ""

    def test_clean_text_colapsa_espacios_y_no_pierde_palabras(self):
        limpio = parseo.clean_text("hola    mundo\n\n\n\nfin")
        assert "hola" in limpio and "mundo" in limpio and "fin" in limpio
        assert "    " not in limpio

    def test_classify_content_type_devuelve_un_tipo_conocido(self):
        assert parseo.classify_content_type(PARRAFO) in {"body", "tabla", "lista", "definicion"}
        lista = "\n".join(f"- item {i}" for i in range(12))
        assert parseo.classify_content_type(lista) in {"lista", "body", "tabla", "definicion"}

    def test_find_sentence_boundary_corta_despues_de_un_punto(self):
        words = "uno dos tres. cuatro cinco seis. siete ocho".split()
        idx = parseo._find_sentence_boundary(words, 4)
        assert words[idx - 1].endswith(".")

    def test_fusionar_continuaciones_no_pierde_celdas(self):
        filas = [["Farmaco", "Dosis"], ["Amoxicilina", "500 mg"], ["", "cada 8 h"]]
        fusion = parseo._fusionar_continuaciones(filas)
        plano = " ".join(" ".join(str(c) for c in f) for f in fusion)
        for celda in ("Farmaco", "Dosis", "Amoxicilina", "500 mg", "cada 8 h"):
            assert celda in plano

    def test_parece_encabezado_distingue_titulos_de_datos(self):
        assert parseo._parece_encabezado(["Farmaco", "Dosis", "Via"]) is True
        assert parseo._parece_encabezado(["12,5", "3.4", "0.9"]) is False


class TestChunker:
    def test_tamanos_dentro_de_los_limites(self):
        children, _parents = parseo.generate_chunks_v2(_paginas(), "libro-test")
        assert children, "tiene que producir chunks"
        for c in children[:-1]:  # el ultimo puede fusionarse o quedar corto
            assert c["word_count"] <= parseo.MAX_SIZE
            assert c["word_count"] >= parseo.MIN_SIZE
        assert all(c["libro_id"] == "libro-test" and c["version"] == 2 for c in children)

    def test_overlap_entre_chunks_consecutivos(self):
        children, _ = parseo.generate_chunks_v2(_paginas(), "libro-test")
        a, b = children[0]["text"].split(), children[1]["text"].split()
        cola = a[-parseo.OVERLAP_SIZE:]
        assert b[:len(cola)] == cola, \
            "el segundo chunk arranca con las ultimas OVERLAP_SIZE palabras del primero"

    def test_parents_agrupan_de_a_tres_y_respetan_el_techo(self):
        children, parents = parseo.generate_chunks_v2(_paginas(), "libro-test")
        assert len(parents) == -(-len(children) // parseo.PARENT_WINDOW)
        assert all(p["word_count"] <= parseo.MAX_PARENT_WORDS for p in parents)
        assert all(c["parent_id"] for c in children), "todo child apunta a un parent"
        assert {c["parent_id"] for c in children} == {p["id"] for p in parents}

    def test_metadata_de_pagina_y_titulos_viaja_al_chunk(self):
        children, _ = parseo.generate_chunks_v2(_paginas(), "libro-test")
        assert children[0]["page_start"] == 1
        assert children[-1]["page_end"] == 6
        assert children[0]["titulo_capitulo"] == "Capitulo 1"
        assert children[-1]["titulo_capitulo"] == "Capitulo 2"

    def test_normalize_chunks_agrega_los_campos_que_usa_el_uploader(self):
        children, _ = parseo.generate_chunks_v2(_paginas(2), "libro-test")
        parseo.normalize_chunks(children)
        for c in children:
            for k in ("text_busqueda", "titulo_seccion_busqueda", "titulo_capitulo_busqueda",
                      "keywords"):
                assert k in c
            assert c["text_busqueda"] == parseo.normalize_for_search(c["text"])

    def test_paginas_vacias_no_rompen(self):
        assert parseo.generate_chunks_v2([], "x") == ([], [])

    def test_ids_e_indices_siguen_el_patron(self):
        children, parents = parseo.generate_chunks_v2(_paginas(2), "libro-test")
        assert children[0]["id"] == "libro-test_v2_00000"
        assert parents[0]["id"] == "libro-test_v2_parent_00000"
        assert [c["chunk_index"] for c in children] == list(range(len(children)))


class TestDeteccionDeEstructura:
    def test_libro_desconocido_usa_los_patrones_default(self):
        paginas = [{"page": 1, "text": "Capitulo 3\ntexto del capitulo\n1.2 Subtema\nmas texto"}]
        out = parseo.detect_structure(paginas, "libro-que-no-existe")
        assert out[0]["page"] == 1 and "text" in out[0]
        assert "titulo_capitulo" in out[0] and "titulo_seccion" in out[0]


class TestLetrasEspaciadas:
    """P11 (13-sep-2026): una letra suelta entre palabras NO es texto espaciado.

    EL DEFECTO. `_clean_spaced_text` reconstruye los titulos que el PDF extrae con letras
    sueltas ("I N T R O D U C C I Ó N" -> "INTRODUCCIÓN"). Su patron —al menos dos pares
    "letra espacio" seguidos de una letra— tambien matcheaba la ultima letra de una palabra,
    una palabra espanola de UNA letra y la primera de la siguiente, asi que
    "Fisiopatología y mecanismos" salia "Fisiopatologíaymecanismos" en el titulo de capitulo
    de todo libro con una "y" en el titulo. Y el titulo viaja en el prefijo que se embebe
    (`pipeline/embeddings.build_embedding_text`): el error corria el vector.

    LA CURA (replicada byte a byte del privado). Una letra cuenta como espaciada solo si
    esta SUELTA, y hacen falta `MIN_LETRAS_SUELTAS` en la corrida. En "a y m" hay una sola;
    en "A R T E" (de "PA R T E") hay tres y se compacta.

    LO QUE NO CURA, y es una ambiguedad real: "Vitaminas A y D en el adulto" sigue
    colapsando ("VitaminasAyDen"). Mirando solo los caracteres, un humano tampoco decide.
    """

    TITULOS = (
        ("Capítulo 1. Definición y epidemiología del síndrome",
         "Capítulo 1. Definición y epidemiología del síndrome"),
        ("Capítulo 2. Fisiopatología y mecanismos de compensación",
         "Capítulo 2. Fisiopatología y mecanismos de compensación"),
        ("Capítulo 4. Tratamiento farmacológico y seguimiento",
         "Capítulo 4. Tratamiento farmacológico y seguimiento"),
        ("CAPÍTULO I - Del objeto y del ambito de aplicacion",
         "CAPÍTULO I - Del objeto y del ambito de aplicacion"),
        ("tejido u órgano", "tejido u órgano"),
    )

    ESPACIADOS = (
        ("I N T R O D U C C I Ó N", "INTRODUCCIÓN"),
        ("B Á S I C O", "BÁSICO"),
        ("PA R T E 3 : A M E T R O P Í A S", "PARTE 3 : AMETROPÍAS"),
        ("C A P I T U L O 5", "CAPITULO 5"),
    )

    @pytest.mark.parametrize("crudo,esperado", TITULOS,
                             ids=lambda v: v[:28] if isinstance(v, str) else v)
    def test_una_letra_suelta_entre_palabras_no_es_texto_espaciado(self, crudo, esperado):
        assert parseo._clean_spaced_text(crudo) == esperado

    @pytest.mark.parametrize("crudo,esperado", ESPACIADOS,
                             ids=lambda v: v[:28] if isinstance(v, str) else v)
    def test_el_texto_realmente_espaciado_se_sigue_compactando(self, crudo, esperado):
        assert parseo._clean_spaced_text(crudo) == esperado

    def test_el_umbral_esta_declarado_y_no_escondido_en_el_regex(self):
        assert parseo.MIN_LETRAS_SUELTAS == 2

    def test_el_titulo_de_capitulo_llega_entero_al_chunk(self):
        """El defecto se veia en el titulo del CHUNK, no en la funcion: aca esta el camino."""
        pagina = "Capítulo 2. Fisiopatología y mecanismos de compensación\n" + PARRAFO * 4
        estructura = parseo.detect_structure([{"page": 1, "text": pagina}], "qa-p11")
        hijos, _ = parseo.generate_chunks_v2(estructura, "qa-p11")
        assert hijos
        assert hijos[0]["titulo_capitulo"] == \
            "Capítulo 2. Fisiopatología y mecanismos de compensación"


class TestElTituloEsDeLaPosicion:
    """Decision D (13-sep-2026): el titulo se asigna por POSICION, no por pagina.

    QUE HACIA. `detect_structure` decidia UN `titulo_capitulo`/`titulo_seccion` por PAGINA y
    ganaba el ULTIMO encabezado que aparecia en ella; despues `generate_chunks_v2` le ponia a
    cada palabra la metadata de SU PAGINA. Con dos encabezados en una pagina —lo normal en un
    tratado maquetado a dos columnas— el chunk que ARRANCA en "Capítulo 1" salia etiquetado
    "Capítulo 2", y ese titulo viaja en el prefijo que se embebe: no era cosmetico.

    LA CURA. La pagina se parte en SEGMENTOS: cada encabezado abre uno, con el mismo `page`.
    El chunker no cambio —ve "paginas" consecutivas con el mismo numero— asi que
    `page_start`/`page_end` y los ids siguen siendo los de antes; lo unico que cambia son los
    titulos.
    """

    def test_la_pagina_con_dos_encabezados_se_parte_en_segmentos(self):
        """El unitario de la cura, sin PDF: dos entradas con el MISMO `page`, cada una con
        su titulo, y el texto conservado palabra por palabra."""
        pagina = ("Capítulo 1. Definición\ncuerpo del primero\n"
                  "Capítulo 2. Fisiopatología\ncuerpo del segundo")
        salida = parseo.detect_structure([{"page": 7, "text": pagina}], "qa-d")
        assert [s["page"] for s in salida] == [7, 7], salida
        assert [s["titulo_capitulo"] for s in salida] == \
            ["Capítulo 1. Definición", "Capítulo 2. Fisiopatología"]
        assert " ".join(s["text"] for s in salida).split() == pagina.split()

    def test_una_pagina_sin_encabezados_sigue_siendo_una_sola_entrada(self):
        """La cura no fragmenta lo que no hace falta: sin encabezados hay UN segmento, con el
        titulo arrastrado de la pagina anterior."""
        paginas = [{"page": 1, "text": "Capítulo 1. Definición\ncuerpo"},
                   {"page": 2, "text": "sigue el cuerpo sin ningún encabezado nuevo"}]
        salida = parseo.detect_structure(paginas, "qa-d")
        assert len(salida) == 2, salida
        assert salida[1]["page"] == 2
        assert salida[1]["titulo_capitulo"] == "Capítulo 1. Definición"

    def test_el_chunk_que_abre_el_capitulo_1_lleva_el_capitulo_1(self):
        """El escenario del hallazgo, en positivo y sobre chunks: una pagina con los dos
        encabezados y cuerpo suficiente para que cada uno de un chunk propio."""
        pagina = ("Capítulo 1. Definición y epidemiología\n" + PARRAFO * 4
                  + "\nCapítulo 2. Fisiopatología y mecanismos\n" + PARRAFO * 4)
        estructura = parseo.detect_structure([{"page": 1, "text": pagina}], "qa-d")
        hijos, _ = parseo.generate_chunks_v2(estructura, "qa-d")
        abre = [c for c in hijos if "Capítulo 1. Definición" in c["text"]]
        assert abre, "ningun chunk contiene el encabezado del capitulo 1"
        assert abre[0]["titulo_capitulo"].startswith("Capítulo 1."), abre[0]["titulo_capitulo"]
        # Y los dos capitulos etiquetan a alguien: con el titulo por PAGINA, el 1 no
        # etiquetaba ni un chunk (ganaba el ultimo encabezado de la pagina).
        numeros = {int(m.group(1)) for c in hijos
                   if (m := re.match(r"Capítulo (\d+)\.", c["titulo_capitulo"]))}
        assert numeros == {1, 2}, sorted(numeros)

    def test_el_chunker_tolera_paginas_repetidas(self):
        """La premisa de la cura: `generate_chunks_v2` acepta varias entradas con el MISMO
        `page`, y `page_start`/`page_end` siguen saliendo del numero de pagina."""
        palabras = " ".join(["palabra"] * 400)
        paginas = [
            {"page": 1, "text": palabras, "titulo_capitulo": "Capítulo 1", "titulo_seccion": ""},
            {"page": 1, "text": palabras, "titulo_capitulo": "Capítulo 2", "titulo_seccion": ""},
            {"page": 2, "text": palabras, "titulo_capitulo": "Capítulo 2", "titulo_seccion": ""},
        ]
        hijos, _ = parseo.generate_chunks_v2(paginas, "qa-d")
        assert hijos
        assert hijos[0]["page_start"] == 1 and hijos[0]["page_end"] == 1
        assert max(c["page_end"] for c in hijos) == 2
        assert all(c["page_start"] <= c["page_end"] for c in hijos)


class TestLoQueNoEsContenido:
    """Las listas de referencias y los indices de codigos no son contenido, y el retrieval
    los excluye (`parseo.NO_CONTENIDO`). Lo que estos tests fijan es la FRONTERA del
    detector: que marque la lista pura y NO el chunk mixto, porque marcar el mixto borra
    prosa real (se midieron ~466 chunks-equivalentes del corpus privado el 12-sep-2026)."""

    LISTA = " ".join(
        f"Autor{i} A, Otro B. Titulo del trabajo {i}. J Med. 201{i % 10};12({i}):100-110."
        for i in range(10))
    PROSA = ("El paciente con insuficiencia cardiaca debe manejarse con precaucion por el riesgo "
             "de provocar una fibrilacion ventricular. ") * 12

    def test_una_lista_pura_de_citas_es_referencias(self):
        assert parseo.classify_content_type(self.LISTA) == "referencias"

    def test_prosa_que_termina_en_la_bibliografia_del_capitulo_sigue_siendo_body(self):
        mixto = self.PROSA + " ".join(f"Autor{i} A. Titulo. Lancet. 2019;39{i}:1-9."
                                      for i in range(9))
        assert parseo.classify_content_type(mixto) == "body"

    def test_pocas_citas_no_alcanzan(self):
        tres = " ".join(f"Autor{i} A. Titulo. Lancet. 2019;39{i}:1-9."
                        for i in range(3)) + " " + self.PROSA
        assert parseo.classify_content_type(tres) == "body"

    def test_la_cita_con_fasciculo_cuenta(self):
        assert parseo.CITA_BIBLIOGRAFICA.search("Lancet. 2018;391(10125):1023-1075.")
        assert parseo.CITA_BIBLIOGRAFICA.search("Hum Genet 1998;102:170-177.")
        assert not parseo.CITA_BIBLIOGRAFICA.search("en 2018; 391 pacientes fueron incluidos")

    def test_una_tabla_de_codigos_es_indice(self):
        codigos = " ".join(f"J{10 + i}.{i % 10} Gripe debida a virus identificado tipo {i}"
                           for i in range(12))
        assert parseo.classify_content_type(codigos) == "indice"

    def test_los_falsos_positivos_medidos_siguen_siendo_body(self):
        b12 = ("La deficiencia de B12 produce anemia megaloblastica. La B12 se absorbe en el ileon. "
               "El factor intrinseco une B12. Sin B12 hay neuropatia. La B12 serica baja. " * 2)
        genes = ("P53 es un supresor tumoral. P16 inhibe CDK4. P21 detiene el ciclo. P53 se muta en "
                 "la mitad de los tumores. P16 se silencia. P21 depende de P53. P53 y P16 juntos. ")
        assert parseo.classify_content_type(b12) == "body"
        assert parseo.classify_content_type(genes) == "body"

    def test_los_tipos_excluidos_son_exactamente_los_que_el_parser_produce(self):
        assert set(parseo.NO_CONTENIDO) == {"referencias", "indice"}
        assert parseo.classify_content_type(self.LISTA) in parseo.NO_CONTENIDO


# ══════════════════════════════════════════════════════════════════════════════════
# El PDF de punta a punta: archivo -> chunks -> texto canonico que se embebe
# ══════════════════════════════════════════════════════════════════════════════════

#: Las paginas del PDF sintetico. La PRIMERA lleva DOS encabezados de capitulo a proposito:
#: es el escenario de la decision D, y el segundo titulo trae la "y" de P11.
PAGINAS_PDF = (
    ("Capítulo 1. Definición y epidemiología del síndrome", PARRAFO * 3,
     "Capítulo 2. Fisiopatología y mecanismos de compensación", PARRAFO * 3),
    ("2.1 Mecanismos De Compensación", PARRAFO * 5),
    ("Capítulo 3. Tratamiento farmacológico y seguimiento", PARRAFO * 5),
)


@pytest.fixture(scope="module")
def pdf_sintetico(tmp_path_factory):
    """Un PDF de tres paginas hecho con PyMuPDF y texto INVENTADO (cero copyright).

    `insert_textbox` corta las lineas solo, que es lo que hace que los encabezados queden en
    una linea propia y `detect_structure` los vea. Si el texto no entrara en la caja el
    fixture falla: un documento truncado en silencio probaria otra cosa.
    """
    fitz = pytest.importorskip("fitz", reason="PyMuPDF es la unica dependencia del parseo")
    doc = fitz.open()
    for bloques in PAGINAS_PDF:
        pagina = doc.new_page()
        tope = 60.0
        for bloque in bloques:
            caja = fitz.Rect(50, tope, 545, 800)
            sobra = pagina.insert_textbox(caja, bloque, fontsize=9, fontname="helv")
            assert sobra > 0, "el bloque no entro en la pagina: el PDF quedaria truncado"
            tope = 800 - sobra + 12
    ruta = tmp_path_factory.mktemp("pdf") / "sintetico.pdf"
    doc.save(ruta)
    doc.close()
    return str(ruta)


@pytest.fixture(scope="module")
def parseado(pdf_sintetico):
    return parseo.parse_pdf_v2(pdf_sintetico, "qa-sintetico")


class TestPdfDePuntaAPunta:
    def test_produce_chunks_con_todos_los_campos_del_contrato(self, parseado):
        hijos, padres = parseado
        assert hijos and padres
        campos = {"id", "libro_id", "page_start", "page_end", "text", "word_count",
                  "titulo_capitulo", "titulo_seccion", "tipo_contenido", "parent_id",
                  "chunk_index", "version"}
        assert campos <= set(hijos[0])
        assert all(c["libro_id"] == "qa-sintetico" for c in hijos)

    def test_las_paginas_son_las_del_documento(self, parseado):
        hijos, _ = parseado
        assert hijos[0]["page_start"] == 1
        assert max(c["page_end"] for c in hijos) == len(PAGINAS_PDF)
        assert all(1 <= c["page_start"] <= c["page_end"] <= len(PAGINAS_PDF) for c in hijos)

    def test_el_capitulo_y_la_seccion_llegan_desde_el_papel(self, parseado):
        hijos, _ = parseado
        assert hijos[0]["titulo_capitulo"].startswith("Capítulo 1."), hijos[0]["titulo_capitulo"]
        numeros = [int(m.group(1)) for c in hijos
                   if (m := re.match(r"Capítulo (\d+)\.", c["titulo_capitulo"]))]
        assert set(numeros) == {1, 2, 3}, sorted(set(numeros))
        assert numeros == sorted(numeros), "el capitulo de los chunks tiene que ser monotono"
        assert any(c["titulo_seccion"].startswith("2.1 ") for c in hijos)

    def test_ningun_titulo_quedo_pegado(self, parseado):
        """P11 de punta a punta: antes de la cura el capitulo 2 salia
        "Fisiopatologíaymecanismos"."""
        capitulos = {c["titulo_capitulo"] for c in parseado[0]}
        for titulo in capitulos:
            assert "íaymecanismos" not in titulo and "oyseguimiento" not in titulo, titulo
        assert any(" y mecanismos de compensación" in t for t in capitulos), capitulos

    def test_los_tamanos_caen_en_la_estrategia(self, parseado):
        hijos, _ = parseado
        for c in hijos[:-1]:  # el ultimo absorbe el resto del documento
            assert parseo.MIN_SIZE <= c["word_count"] <= parseo.MAX_SIZE, c["word_count"]

    def test_normalize_chunks_y_el_texto_canonico_que_se_embebe(self, parseado):
        """El final del camino: lo que se guarda y lo que se le manda al proveedor.

        `build_embedding_text` es LA politica del prefijo (una sola, la del privado): el
        titulo de capitulo con TILDE —la copia vieja de la API escribia "Capitulo:" sin
        tilde, o sea otro vector— y el texto del chunk al final, intacto.
        """
        hijos, _ = parseado
        copia = [dict(c) for c in hijos]
        parseo.normalize_chunks(copia)
        primero = copia[0]
        assert primero["text_busqueda"] == parseo.normalize_for_search(primero["text"])
        assert primero["keywords"].startswith("capitulo 1. definicion y epidemiologia")

        canonico = embeddings.build_embedding_text(primero)
        assert canonico.startswith(f"Capítulo: {primero['titulo_capitulo']}.")
        assert canonico.endswith(primero["text"])
        assert "Capitulo:" not in canonico, "el prefijo sin tilde es el de la API vieja"

    def test_el_parseo_es_una_funcion(self, pdf_sintetico):
        """Determinismo: mismo PDF, mismos ids, mismos cortes, mismo texto."""
        una, padres_una = parseo.parse_pdf_v2(pdf_sintetico, "qa-sintetico")
        otra, padres_otra = parseo.parse_pdf_v2(pdf_sintetico, "qa-sintetico")
        assert [(c["id"], c["chunk_index"], c["text"]) for c in una] == \
               [(c["id"], c["chunk_index"], c["text"]) for c in otra]
        assert [(p["id"], p["text"]) for p in padres_una] == \
               [(p["id"], p["text"]) for p in padres_otra]


class TestUnaSolaFuenteDeVerdad:
    """La doctrina como garantia tecnica: `pipeline/parseo.py` es la unica copia del parseo,
    y `parser_v2.py` es un SHIM que la re-exporta (no una segunda implementacion)."""

    CANONICAS = {"normalize_for_search", "clean_text", "_area", "_dentro_de_alguna",
                 "_celda_limpia", "_fusionar_continuaciones", "_parece_encabezado",
                 "render_tabla", "_tokens", "_pierde_contenido", "extraer_texto_pagina",
                 "classify_content_type", "_find_sentence_boundary", "generate_chunks_v2",
                 "detect_structure", "parse_pdf_v2",
                 "_clave_de_linea", "lineas_repetidas", "quitar_lineas",
                 # 16-sep-2026 (tanda 2 del diseño del uso real): el indice analitico y la
                 # calidad por chunk. Entran a la lista el mismo dia que nacen.
                 "_cerca_de_un_borde", "_es_indice", "calidad_de_texto", "firmas_de_calidad",
                 "ratio_palabras_sin_vocales", "ratio_letras_sueltas", "ratio_no_alfabetico",
                 "_escalar"}

    def _defs(self, rel):
        arbol = ast.parse((RAIZ / rel).read_text(encoding="utf-8"))
        return {n.name for n in arbol.body if isinstance(n, ast.FunctionDef)}

    def test_el_paquete_las_define_todas(self):
        assert self.CANONICAS <= self._defs("pipeline/parseo.py")

    def test_el_shim_del_cli_no_redefine_ninguna(self):
        assert self._defs("parser_v2.py") & self.CANONICAS == set()
        assert self._defs("migrate_chunks.py") & {"normalize_for_search",
                                                  "normalize_chunks"} == set()

    def test_parser_v2_reexporta_los_MISMOS_objetos(self):
        """Identidad, no igualdad: si alguien copiara el cuerpo en el shim, esto se rompe."""
        import parser_v2

        assert parser_v2.parse_pdf_v2 is parseo.parse_pdf_v2
        assert parser_v2.generate_chunks_v2 is parseo.generate_chunks_v2
        assert parser_v2.detect_structure is parseo.detect_structure
        assert parser_v2.STRUCTURE_PATTERNS is parseo.STRUCTURE_PATTERNS

    def test_el_paquete_no_importa_fitz_al_cargar(self):
        arbol = ast.parse((RAIZ / "pipeline/parseo.py").read_text(encoding="utf-8"))
        top = {a.name for n in arbol.body if isinstance(n, ast.Import) for a in n.names}
        assert "fitz" not in top, \
            "PyMuPDF se importa perezoso (10 s de arranque en frio, 1-sep-2026)"


class TestLaMaquetaNoEsContenido:
    """El encabezado/pie de pagina se va por su FORMA, no por el nombre de nadie.

    LA CURA QUE ESTE TEST PROTEGE (13-sep-2026, items C-1/C-2 de la auditoria de exposicion).
    `clean_text` borraba lineas con regex que nombraban al titular de los derechos de un libro y
    al sitio del que habia salido un PDF. En un espejo publico eso no es limpieza: es un recibo de
    la procedencia del corpus, y ademas obliga a editar el codigo por cada documento nuevo. Ahora
    son dos reglas de forma:

      · REPETICION -- `lineas_repetidas` mira el DOCUMENTO (no la pagina) y marca la linea que
        aparece en el borde de >= MIN_PAGINAS_REPETIDAS paginas, corta en absoluto y corta contra
        el ancho del cuerpo. `parse_pdf_v2` la aplica antes de `detect_structure`.
      · AVISO DE DERECHOS -- `AVISO_DE_DERECHOS`: el simbolo con año, o la palabra con la que se
        escribe un aviso. Sin un solo nombre propio.

    Las tres condiciones de la primera regla salieron de falsos positivos MEDIDOS; los tests de
    abajo son uno por condicion, y son lo que impide que el filtro se coma contenido.
    El pie de los fixtures es inventado: un test que usara el pie real de un libro reintroduciria
    en el repo justo lo que se saco. `tests/test_repo.py` lo verifica sobre todo el arbol.
    """

    PIE = "Editorial Ejemplo S.A. -- prohibida su reproduccion"
    UNICA = "Nota del editor a esta primera tirada, que aparece una sola vez"

    @classmethod
    def _paginas(cls, n=6, con_pie=4, unica_en=1):
        """`n` paginas de prosa; las primeras `con_pie` llevan el pie al final."""
        out = []
        for p in range(1, n + 1):
            lineas = [f"Pagina {p}. " + " ".join(PARRAFO.split())]
            if p == unica_en:
                lineas.append(cls.UNICA)
            if p <= con_pie:
                lineas.append(cls.PIE)
            out.append("\n".join(lineas))
        return out

    def test_el_pie_repetido_en_4_de_6_paginas_es_maqueta(self):
        assert self.PIE in parseo.lineas_repetidas(self._paginas())

    def test_la_linea_que_aparece_una_vez_se_conserva(self):
        repetidas = parseo.lineas_repetidas(self._paginas())
        assert self.UNICA not in repetidas
        limpia = parseo.quitar_lineas(self._paginas()[0], repetidas)
        assert self.UNICA in limpia and self.PIE not in limpia

    def test_dos_paginas_no_alcanzan(self):
        """MIN_PAGINAS_REPETIDAS = 3: con dos coincidencias todavia puede ser un parrafo."""
        assert parseo.lineas_repetidas(self._paginas(con_pie=2)) == set()

    def test_el_cuerpo_no_se_toca(self):
        repetidas = parseo.lineas_repetidas(self._paginas())
        for i, pagina in enumerate(self._paginas(), start=1):
            limpia = parseo.quitar_lineas(pagina, repetidas)
            assert f"Pagina {i}." in limpia and "insuficiencia cardiaca" in limpia

    def test_una_linea_repetida_en_EL_MEDIO_no_es_maqueta(self):
        """Condicion 1 (posicion): una frase que el libro repite en el CUERPO se queda. Un
        encabezado o un pie, ademas de repetirse, esta siempre en el mismo lugar."""
        estribillo = "Ante cualquier duda, consultar al especialista."
        cuerpo = " ".join(PARRAFO.split())
        paginas = [f"Titulo {p}\n{cuerpo}\n{estribillo}\n{cuerpo}\nfin de la pagina {p}"
                   for p in range(1, 6)]
        assert estribillo not in parseo.lineas_repetidas(paginas)

    def test_un_parrafo_LARGO_repetido_en_el_borde_no_es_maqueta(self):
        """Condicion 2 (largo absoluto): PyMuPDF no hace wrap, asi que un documento sintetico
        puede tener el mismo parrafo entero como unica linea de cada pagina."""
        largo = " ".join(PARRAFO.split())
        assert len(largo) > parseo.MAX_CHARS_MAQUETA
        assert parseo.lineas_repetidas([largo + "\n" + self.PIE for _ in range(5)]) == {self.PIE}

    def test_una_linea_de_CUERPO_a_ancho_completo_no_es_maqueta(self):
        """Condicion 3 (ancho relativo), y es el falso positivo mas fino: un texto armado con un
        vocabulario que CICLA produce paginas periodicas, y la primera y la ultima linea de una
        pagina salen identicas a las de otra. Son lineas de cuerpo justificadas -pasan el techo
        absoluto- y aun asi no son maqueta: un pie no llena la caja de texto."""
        vocabulario = ("caudal", "impulsor", "rodamiento", "sello", "brida", "aspiracion",
                       "descarga", "cebado", "valvula", "tablero", "conexion", "carcasa")
        palabras = [vocabulario[i % len(vocabulario)] for i in range(600)]
        lineas = [" ".join(palabras[i:i + 10]) for i in range(0, len(palabras), 10)]
        paginas = ["\n".join(lineas[i:i + 12]) for i in range(0, len(lineas), 12)]
        assert len(paginas) >= 3 and max(len(l) for l in lineas) < parseo.MAX_CHARS_MAQUETA
        assert parseo.lineas_repetidas(paginas) == set()

    def test_la_repeticion_se_mide_con_los_espacios_normalizados(self):
        """El mismo pie extraido de dos paginas difiere en un espacio doble: el espaciado lo pone
        la maqueta, no el texto."""
        cuerpo = " ".join(PARRAFO.split())
        paginas = [f"{cuerpo} uno\n" + self.PIE,
                   f"{cuerpo} dos\n" + self.PIE.replace(" -- ", "   --  "),
                   f"{cuerpo} tres\n" + self.PIE + " "]
        assert parseo.lineas_repetidas(paginas) == {self.PIE}

    def test_el_aviso_de_derechos_se_va_por_su_forma(self):
        """El simbolo con año o la palabra. El MISMO patron saca el aviso de una editorial
        inventada y el de otra, sin conocer ninguna: es lo que hace que no haya que editar el
        codigo cuando entra un libro nuevo. El simbolo se escribe con su escape para que el
        guard de `tests/test_repo.py` no se tropiece con este archivo."""
        for aviso in ("\u00a9 2024 Casa Editora", "(c) 2024 Casa Editora",
                      "Todos los derechos reservados", "Copyright de la presente edicion",
                      "Prohibido fotocopiar esta obra"):
            assert parseo.clean_text(aviso) == "", aviso

    def test_una_linea_de_cuerpo_larga_con_la_palabra_se_conserva(self):
        """La palabra sola no alcanza: un aviso legal es una linea CORTA. Una linea de cuerpo que
        use "derechos reservados" o "fotocopiar" -plausible en un tratado de derecho o en un
        manual- es larga, y borrarla seria borrar contenido."""
        cuerpo = ("Las provincias conservan todo el poder no delegado y los derechos reservados "
                  "por pactos especiales al tiempo de su incorporacion, segun el articulo 121 "
                  "de la Constitucion Nacional, que la doctrina lee como regla de reparto.")
        assert len(cuerpo) > parseo.MAX_CHARS_MAQUETA
        assert parseo.clean_text(cuerpo) == cuerpo
        assert parseo.clean_text("Todos los derechos reservados. Prohibida su reproduccion.") == ""

    def test_el_simbolo_solo_no_alcanza(self):
        """Sin año ni palabra no es un aviso: el simbolo suelto aparece en notas al pie."""
        assert parseo.clean_text("el signo \u00a9 se usa para marcar la obra") != ""

    def test_el_texto_de_cuerpo_pasa_intacto(self):
        assert parseo.clean_text(PARRAFO.strip()) == " ".join(PARRAFO.split())


class TestElPdfConPieRepetido:
    """El filtro de punta a punta, sobre un PDF de verdad y no sobre strings."""

    PIE = TestLaMaquetaNoEsContenido.PIE

    @pytest.fixture
    def pdf_con_pie(self, tmp_path):
        fitz = pytest.importorskip("fitz")
        doc = fitz.open()
        for p in range(1, 7):
            pagina = doc.new_page()
            pagina.insert_text((72, 72), f"Capitulo {p}\n" + PARRAFO * 6, fontsize=9)
            if p <= 4:                     # el pie, en 4 de las 6 paginas
                pagina.insert_text((72, 800), self.PIE, fontsize=8)
            if p == 5:                     # una linea de UNA pagina, en el mismo lugar
                pagina.insert_text((72, 800), "Errata de la pagina cinco", fontsize=8)
        ruta = tmp_path / "con_pie.pdf"
        doc.save(ruta)
        doc.close()
        return str(ruta)

    def test_el_pie_no_llega_a_ningun_chunk(self, pdf_con_pie):
        hijos, padres = parseo.parse_pdf_v2(pdf_con_pie, "libro-con-pie")
        assert hijos and padres
        assert not any("Editorial Ejemplo" in c["text"] for c in hijos)
        assert not any("Editorial Ejemplo" in p["text"] for p in padres)

    def test_el_cuerpo_y_la_linea_unica_sobreviven(self, pdf_con_pie):
        hijos, _ = parseo.parse_pdf_v2(pdf_con_pie, "libro-con-pie")
        todo = " ".join(c["text"] for c in hijos)
        assert "insuficiencia cardiaca" in todo
        assert "Errata de la pagina cinco" in todo
        assert hijos[0]["page_start"] == 1 and hijos[-1]["page_end"] == 6

    def test_el_pie_no_se_hace_pasar_por_titulo(self, pdf_con_pie):
        """Por eso la pasada corre ANTES de `detect_structure`."""
        hijos, _ = parseo.parse_pdf_v2(pdf_con_pie, "libro-con-pie")
        titulos = {c["titulo_capitulo"] for c in hijos} | {c["titulo_seccion"] for c in hijos}
        assert not any("Editorial" in t for t in titulos), titulos


class TestElIndiceYLaCalidadDelEspejo:
    """Las dos reglas que llegaron al espejo el 16-sep-2026 (tanda 2 del diseño del uso real).

    QUE FIJAN. `pipeline/` no se edita aca: es un espejo byte a byte del pipeline privado. Lo
    que vive aca es la REGRESION, para que un espejo desactualizado o una copia mal pegada se
    note en el CI publico. Estas dos son las de esta tanda:

      · `indice` reconoce el INDICE ANALITICO de un tratado —pares "termino, numero de pagina"
        densos, sin oraciones ni verbos— y no solo la tabla de codigos de clasificacion. La
        posicion de la pagina llega al clasificador como REFUERZO y nunca como condicion.
      · Cada chunk sale con `calidad` y `calidad_valor`: la calidad del TEXTO, medida por
        forma, con las firmas que ven los bytes de control y los literales `(cid:NN)` que un
        detector por letras sueltas no puede ver.

    Los textos son inventados, como todo el resto de esta suite.
    """

    INDICE = ("de shigatoxina, 723 difusamente adherente, 723 enteroagregativa, 723 "
              "enterohemorragica, 723 sindrome uremico hemolitico, 724 enteroinvasiva, 723 "
              "enteropatogena, 722 enterotoxigenica, 722 ") * 4
    TABLA_DOSIS = ("Ceftriaxona 50-75 Ceftazidima 150 Tobramicina 5-7 Ampicilina 100-200 "
                   "Gentamicina 5-7 Vancomicina 40-60 Meropenem 60-120 Cefotaxima 150-200 ") * 3
    CORRUPTO = "Por HVWH\x03PRWLYR\x0f\x03HO\x03GLDJQyVWLFR\x03\\\x03OD\x03GH\x03ODV\x03DQH- PLDV"

    def test_el_indice_analitico_es_indice(self):
        assert parseo.classify_content_type(self.INDICE) == "indice"

    def test_una_columna_de_dosis_no_lo_es(self):
        assert parseo.classify_content_type(self.TABLA_DOSIS) == "body"

    def test_la_prosa_no_lo_es_ni_al_final_del_libro(self):
        assert parseo.classify_content_type(PARRAFO * 4) == "body"
        assert parseo.classify_content_type(PARRAFO * 4, 998, 1000) == "body"

    def test_la_posicion_es_un_refuerzo_opcional(self):
        """La firma nueva tiene que aceptar los llamadores viejos de un solo argumento."""
        assert parseo.classify_content_type(self.INDICE) == \
            parseo.classify_content_type(self.INDICE, None, None)

    def test_la_rama_lista_ya_no_existe(self):
        vinetas = "\n".join(f"- item numero {i}" for i in range(12))
        assert parseo.classify_content_type(vinetas) != "lista"

    def test_la_calidad_ve_los_bytes_de_control(self):
        assert parseo.calidad_de_texto(self.CORRUPTO)[0] == "corrupta"
        assert parseo.calidad_de_texto(PARRAFO * 3)[0] == "ok"

    def test_cada_chunk_sale_con_su_calidad(self):
        hijos, _ = parseo.generate_chunks_v2(_paginas(), "libro-test")
        assert hijos
        assert all(c["calidad"] == "ok" and 0.0 <= c["calidad_valor"] <= 1.0 for c in hijos)

    def test_la_carga_declara_los_dos_campos(self):
        from pipeline import carga

        assert carga.CAMPOS_CHUNK["calidad"] == "ok"
        assert carga.CAMPOS_CHUNK["calidad_valor"] == 0.0


class TestElContratoChunkV1:
    """Todo chunk que este parser produce valida contra `chunk/v1` (copia en tests/contracts/).

    POR QUE ESTA ACA. El engine ya vendorizaba tres schemas de `nomos-contracts` con su test de
    paridad; `chunk/v1` faltaba, y es el que describe lo que ESTE paquete produce. Sin el, los
    campos nuevos de un chunk podian salir del parseo sin que nada del lado publico dijera si
    el contrato los admite — que es justo lo que paso el 13-sep-2026 con `page_start` en null.
    """

    @staticmethod
    def _validador():
        import json

        from jsonschema import Draft202012Validator

        schema = json.loads(
            (RAIZ / "tests" / "contracts" / "chunk.schema.json").read_text(encoding="utf-8"))
        return Draft202012Validator(schema)

    def test_los_chunks_del_chunker_validan(self):
        hijos, _ = parseo.generate_chunks_v2(_paginas(), "libro-test")
        parseo.normalize_chunks(hijos)
        validador = self._validador()
        assert hijos
        for chunk in hijos:
            errores = sorted(validador.iter_errors(chunk), key=lambda e: list(e.path))
            assert errores == [], (chunk["id"], [e.message for e in errores])

    def test_la_calidad_y_su_valor_estan_declarados(self):
        schema = self._validador().schema
        assert schema["properties"]["calidad"]["enum"] == ["ok", "dudosa", "corrupta"]
        assert schema["properties"]["calidad_valor"]["maximum"] == 1
        assert "calidad" not in schema["required"], "es aditivo: el corpus viejo no la trae"

    def test_el_schema_vendorizado_es_el_de_nomos_contracts(self):
        """Paridad byte a byte, igual que los otros tres schemas (tests/test_admin_v1.py)."""
        fuente = RAIZ.parent.parent / "nomos" / "nomos-contracts" / "schemas" / "chunk.schema.json"
        if not fuente.exists():
            pytest.skip("nomos-contracts no esta en este disco")
        assert (RAIZ / "tests" / "contracts" / "chunk.schema.json").read_text(encoding="utf-8") == \
            fuente.read_text(encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════════
# TANDA 6 (16-sep-2026): el folio acotado al borde, la valvula por tabla, la firma de
# letras espaciadas y el decodificador de mapeos rotos.
#
# El texto de estos tests es INVENTADO, como el del resto de la suite: ningun libro entra
# a este repo. La FORMA del cifrado si es la medida en el corpus real —el corrimiento
# reproduce el orden de glifos estandar (el espacio en \x03, la "a" en "D") y la
# sustitucion, el orden de subset de un CFF (codigos de control por primer uso)—.
# ══════════════════════════════════════════════════════════════════════════════════

PROSA_INVENTADA = (
    "La revision periodica del equipo exige una lista de control escrita y firmada por el "
    "responsable del turno. Cada punto de la lista describe una condicion observable y el "
    "valor esperado, de modo que dos personas distintas lleguen al mismo resultado. Cuando "
    "una condicion no se cumple se anota la desviacion, se estima el riesgo y se decide si "
    "el equipo puede seguir en servicio hasta la proxima parada programada. El registro se "
    "conserva durante cinco anos y se revisa en cada auditoria interna de la planta. "
    "La formacion del personal es la segunda mitad del sistema: sin una practica anual "
    "documentada, la lista se llena por costumbre y deja de describir el estado real. "
    "El procedimiento define un examen breve y una demostracion practica para cada tarea. "
)


class TestElFolioEsDelBorde:
    """La regla del numero de pagina dejo de comerse las celdas de las tablas.

    La regla vieja borraba CUALQUIER linea de 1 a 4 digitos, y PyMuPDF emite cada celda de
    una tabla en su propio renglon: toda celda numerica corta sin separador desaparecia.
    Ahora sale a lo sumo UNA linea de digitos por borde (`LINEAS_DE_BORDE` lineas no vacias
    arriba y abajo).
    """

    TABLA_POR_RENGLONES = ("Repuesto\nCantidad\nIntervalo\nUnidad\n"
                           "Reten\n15\n12-24\nmeses\n"
                           "Rodamiento\n40\n8\nmeses\n")

    def test_una_celda_numerica_del_medio_sobrevive(self):
        limpio = parseo.clean_text("Tabla de repuestos\n" + self.TABLA_POR_RENGLONES + "Fin.")
        renglones = limpio.split("\n")
        assert "15" in renglones and "40" in renglones, limpio

    def test_el_folio_detras_del_titulo_corrido_se_va(self):
        pagina = "Manual de la planta - capitulo 3\n12\nEl cuerpo de la pagina empieza aca.\n"
        assert "12" not in parseo.clean_text(pagina).split("\n")

    def test_el_folio_del_pie_tambien(self):
        assert "45" not in parseo.clean_text("Cuerpo.\nPie del manual\n45\n").split("\n")

    def test_a_lo_sumo_uno_por_borde(self):
        assert "30" in parseo.clean_text("15\n30\nCuerpo de la pagina.\n").split("\n")

    def test_quitar_folio_es_puro(self):
        assert "7" in parseo.quitar_folio("Titulo\n7\nCuerpo", borde=1).split("\n")
        assert "7" not in parseo.quitar_folio("Titulo\n7\nCuerpo", borde=2).split("\n")


class _TablaDoble:
    def __init__(self, bbox, filas):
        self.bbox = bbox
        self._filas = filas

    def extract(self):
        return self._filas


class _PaginaDoble:
    def __init__(self, plano, bloques, tablas):
        self.number = 0
        self._plano = plano
        self._bloques = bloques
        self._tablas = tablas

    def get_text(self, modo="text"):
        if modo == "blocks":
            return [(*bloque, i, 0) for i, bloque in enumerate(self._bloques)]
        return self._plano

    def find_tables(self):
        return type("Resultado", (), {"tables": self._tablas})()


class TestLaValvulaPorTabla:
    """La perdida se evalua POR TABLA y no por pagina: una region de prosa que `find_tables`
    cree grilla ya no arrastra al piso a las tablas buenas de la misma pagina."""

    ESPURIO = ("El retraso en el cumplimiento de una tarea puede deberse a la variacion normal "
               "del proceso, y por eso conviene repetir la medicion antes de intervenir el "
               "equipo o de cambiar el punto de ajuste del controlador principal.")

    def _pagina(self):
        buena = _TablaDoble((0, 100, 200, 200),
                            [["Repuesto", "Cantidad"], ["Reten", "15 u"], ["Rodamiento", "40 u"]])
        espuria = _TablaDoble((0, 300, 200, 400),
                              [[self.ESPURIO[:40], None], [None, ""], ["", None]])
        bloques = [(0, 100, 200, 200, "Repuesto Cantidad\nReten 15 u\nRodamiento 40 u"),
                   (0, 300, 200, 400, self.ESPURIO)]
        plano = "Repuesto Cantidad\nReten 15 u\nRodamiento 40 u\n" + self.ESPURIO
        return _PaginaDoble(plano, bloques, [buena, espuria])

    def test_la_buena_se_reconstruye_y_la_que_pierde_vuelve_a_plano(self):
        salida = parseo.extraer_texto_pagina(self._pagina())
        assert "Reten (Repuesto)" in salida, salida
        assert self.ESPURIO in salida, salida

    def test_el_texto_sobreimpreso_no_es_perdida(self):
        """El PDF escribe el mismo parrafo DOS veces y la grilla lo rinde una: contando
        ocurrencias eso era un 50 % de perdida; contando palabras distintas es cero."""
        doble = self.ESPURIO + "\n" + self.ESPURIO
        assert parseo._palabras_perdidas(doble, self.ESPURIO) == 0.0
        assert not parseo._pierde_la_tabla(doble, self.ESPURIO)

    def test_una_tabla_que_se_come_el_parrafo_si_pierde(self):
        assert parseo._pierde_la_tabla(self.ESPURIO, self.ESPURIO[:40])

    def test_los_dos_umbrales_son_distintos(self):
        assert parseo.MAX_PERDIDA_TOKENS == 0.02 and parseo.MAX_PERDIDA_TABLA == 0.10


class TestCorruptaPideUnaFirmaDura:
    """Las dos firmas BLANDAS —letras sueltas y proporcion no alfabetica— describen igual de
    bien una tabla numerica que un texto roto, asi que solas techan en `dudosa`."""

    TABLA_DE_CRUCES = ("x x FVA13 Si x x x x x x x MenFIC Si x x x x x x x HPX Si x x x x "
                       "x x DTPz Si x x x x x x x VPX Si x x x x")
    EPIGRAFE = ("RSS Formacion de horquilla de extremos y eliminacion del segmento intercalado, "
                "incluyendo las dos RSS Escision a nivel de las RSS A B C D V D J C A B C D")
    ESPACIADO = " ".join("l a r e v i s i o n d e l e q u i p o e s a n u a l".split()) * 8

    def test_la_tabla_de_cruces_no_es_corrupta(self):
        firmas = parseo.firmas_de_calidad(self.TABLA_DE_CRUCES)
        assert max(firmas[f] for f in parseo.FIRMAS_DURAS) == 0.0
        assert parseo.calidad_de_texto(self.TABLA_DE_CRUCES)[0] == "dudosa"

    def test_el_epigrafe_de_figura_tampoco(self):
        assert parseo.calidad_de_texto(self.EPIGRAFE)[0] in ("ok", "dudosa")
        assert parseo.ratio_letras_espaciadas(self.EPIGRAFE) == 0.0

    def test_el_texto_espaciado_si(self):
        """La mitad que no se puede perder: el PDF extraido caracter a caracter."""
        assert parseo.firmas_de_calidad(self.ESPACIADO)["espaciadas"] > 0
        assert parseo.calidad_de_texto(self.ESPACIADO)[0] == "corrupta"

    def test_las_dos_listas_cubren_todas_las_firmas(self):
        firmas = set(parseo.firmas_de_calidad("cualquier texto de prueba"))
        assert firmas == set(parseo.FIRMAS_DURAS) | set(parseo.FIRMAS_BLANDAS)


class TestElDecodificadorDeMapeosRotos:
    """`pipeline/decodificacion.py`: el texto de un PDF cuyos glifos no se pueden traducir se
    recupera con la clave deducida del PROPIO documento. Cifrado sintetico, texto inventado."""

    CORRIMIENTO = 29
    ACENTOS = {"a": chr(105), "e": chr(112), "i": chr(116), "o": chr(121), "u": chr(126)}

    @staticmethod
    def _corrimiento(texto):
        return "".join(chr(ord(c) - TestElDecodificadorDeMapeosRotos.CORRIMIENTO)
                       if 32 <= ord(c) <= 126 else c for c in texto)

    @staticmethod
    def _sustitucion(texto):
        tabla, siguiente = {}, 2
        for c in texto:
            if c not in tabla:
                tabla[c] = chr(siguiente)
                siguiente += 1
        return tabla, "".join(tabla[c] for c in texto)

    @staticmethod
    def _spans(texto, largo=90):
        return [texto[i:i + largo] for i in range(0, len(texto), largo)]

    @property
    def lexico(self):
        return decodificacion.construir_lexico([PROSA_INVENTADA])

    def test_recupera_un_corrimiento_constante(self):
        cifrado = self._corrimiento(PROSA_INVENTADA * 6)
        salida = decodificacion.resolver_fuente(self._spans(cifrado), self.lexico)
        assert salida["metodo"] == "corrimiento" and salida["k"] == self.CORRIMIENTO
        assert salida["confianza"] >= decodificacion.UMBRAL_CONFIANZA, salida
        decodificado = decodificacion.aplicar_tabla(cifrado, salida["tabla"])
        assert "La revision periodica del equipo" in decodificado

    def test_recupera_una_sustitucion_sin_corrimiento_posible(self):
        _, cifrado = self._sustitucion(PROSA_INVENTADA * 6)
        salida = decodificacion.resolver_fuente(self._spans(cifrado), self.lexico)
        assert salida["metodo"] == "sustitucion" and salida["k"] is None
        assert salida["confianza"] >= decodificacion.UMBRAL_CONFIANZA, salida
        assert "revision" in decodificacion.aplicar_tabla(cifrado, salida["tabla"])

    def test_la_prosa_sana_no_se_toca(self):
        """La puerta por span: una misma fuente puede traer spans rotos y spans sanos."""
        cifrado = self._corrimiento(PROSA_INVENTADA * 6)
        salida = decodificacion.resolver_fuente(self._spans(cifrado), self.lexico)
        deco = decodificacion.Decodificador({"F": salida["tabla"]}, self.lexico, {0}, [])
        for sano in self._spans(PROSA_INVENTADA)[:8]:
            assert deco.decodificar_span("F", sano) is None, repr(sano[:50])

    def test_lo_que_no_se_puede_resolver_queda_como_esta_y_la_calidad_lo_marca(self):
        ruido = " ".join("qxzkvw" + str(i % 7) for i in range(4000))
        _, cifrado = self._sustitucion(ruido)
        salida = decodificacion.resolver_fuente(self._spans(cifrado), self.lexico)
        assert salida["confianza"] < decodificacion.UMBRAL_CONFIANZA
        assert parseo.calidad_de_texto(cifrado[:2000])[0] == "corrupta"

    def test_el_espacio_que_python_llama_espacio_no_lo_es(self):
        """`str.isspace()` dice True para \\x0b, \\x0c y \\x1c-\\x1f, y en estos PDF esos codigos
        son glifos (el parentesis, los dos puntos). Al reves, \\t y \\n pueden ser glifos."""
        for codigo in ("\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x1f", "\t", "\n"):
            assert not decodificacion.es_espacio(codigo)
        assert decodificacion.es_espacio(" ")

    def test_un_documento_sano_no_paga_la_pasada_cara(self):
        class DocDoble:
            page_count = 3

            def __init__(self):
                self.modos = []

            def __getitem__(self, i):
                doc = self

                class P:
                    number = i

                    def get_text(self, modo="text"):
                        doc.modos.append(modo)
                        return PROSA_INVENTADA

                return P()

        doc = DocDoble()
        assert decodificacion.resolver(doc, "qa-sano") is None
        assert set(doc.modos) == {"text"}

    def test_la_copia_del_modulo_es_la_del_repo_privado_si_esta(self):
        privado = RAIZ.parent / "medgraph" / "pipeline" / "decodificacion.py"
        if not privado.exists():
            pytest.skip("el repo privado no esta en este disco")
        assert (RAIZ / "pipeline" / "decodificacion.py").read_text(encoding="utf-8") == \
            privado.read_text(encoding="utf-8")
