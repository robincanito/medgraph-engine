"""Errores de admin/v1 con los codigos estables del contrato (nomos-contracts, admin/v1).

POR QUE UN MODULO PARA UNA EXCEPCION. El contrato pide que el cuerpo de error sea
`{detail, code}` en la RAIZ: la consola enruta por `code` y muestra `detail`. Un `HTTPException`
de FastAPI envuelve todo en `{"detail": ...}` y el `code` se pierde, asi que admin/v1 usa su
propia excepcion con su handler en `main.py`.

LO QUE ESTE MODULO NO TIENE, Y ES LA DIFERENCIA CON LA INSTANCIA PRIVADA: roles. Alla este
archivo decide quien puede ESCRIBIR (key completa, key de solo lectura, JWT de Clerk con rol de
organizacion). Aca hay una sola credencial —la API key del middleware de `main.py`— y admin/v1 es
de solo lectura, asi que no hay nada que autorizar: o entraste o no entraste. El descriptor lo
publica tal cual (`auth.schemes = ["api_key"]`), para que la consola no espere un modelo de roles
que este servicio no tiene.
"""


class AdminError(Exception):
    """Error de admin/v1 con codigo estable del contrato.

    `code` es uno de los codigos que la consola conoce: not_found, conflict, forbidden_role,
    not_editable, warming_up, capability_disabled, validation_error, unauthorized. Se agrega
    `graph_unavailable` para el grafo que no contesta, que es el 503 de una instancia sin
    wake-on-request (el contrato solo define `warming_up`, que es otra cosa: un grafo que
    esta despertando a proposito).
    """

    def __init__(self, status: int, code: str, detail: str):
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail
