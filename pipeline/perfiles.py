"""Donde viven los perfiles de dominio y como se leen. Un solo lugar, un solo cargador.

POR QUE (9-sep-2026). El perfil (`nomos-contracts/profiles/<dominio>.yaml`) lo necesitan dos
partes que hasta ahora no se hablaban:

  - la API, para armar el descriptor de admin/v1 (entidades, relaciones, `medicina@1`);
  - el pipeline, para saber COMO procesar el material (pipeline/estrategia.py).

Estaba vendorizado en `api/profiles/`, o sea adentro de una de las dos. El pipeline no puede
importar de la API (el engine OSS no la tiene), asi que el directorio subio a la raiz y este
modulo es el unico que sabe la ruta. La copia sigue siendo copia: un test la compara byte a byte
con la de nomos-contracts, porque el contrato es de nomos-contracts y esto es un vendor.

Se resuelve relativo a este archivo, que queda igual en el repo (`medgraph/profiles/`) y en la
imagen (`/app/profiles/`, ver Dockerfile).
"""
from functools import lru_cache
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DIRECTORIO = RAIZ / "profiles"

# Dominio de esta instancia. Cuando una instancia sirva mas de uno, sale de config, no de aca.
POR_DEFECTO = "medicina"


def ruta_de(dominio: str = POR_DEFECTO) -> Path:
    """Ruta del YAML de un dominio. No verifica que exista: eso lo dice `cargar`."""
    return DIRECTORIO / f"{dominio}.yaml"


@lru_cache(maxsize=4)
def cargar(dominio: str = POR_DEFECTO) -> dict:
    """El perfil como dict. Falla cerrado: sin perfil no se sabe que se esta procesando."""
    import yaml  # perezoso: el pipeline corre en contextos donde no se necesita

    ruta = ruta_de(dominio)
    if not ruta.exists():
        raise FileNotFoundError(
            f"no hay perfil para el dominio '{dominio}' en {DIRECTORIO}; "
            "copialo de nomos-contracts/profiles/"
        )
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def estrategia(dominio: str = POR_DEFECTO):
    """La estrategia de procesamiento que declara el perfil de un dominio.

    Sin secciones `parse:`/`chunk:` devuelve el default historico, asi que llamar a esto en un
    dominio que todavia no las declaro no cambia nada.
    """
    from pipeline.estrategia import desde_perfil

    return desde_perfil(cargar(dominio))
