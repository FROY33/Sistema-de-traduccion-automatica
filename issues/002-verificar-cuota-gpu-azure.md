# Verificar la cuota de GPU en las 4 cuentas de Azure y decidir si sustituye a Colab

**Tipo:** HITL (decisión con las cuentas del equipo)

## PRD de origen

`issues/prd.md`

## Qué construir

El equipo dispone de 4 cuentas de Azure con 100 USD cada una. Si permiten máquinas con GPU (por ejemplo, una T4 o instancias Spot), ofrecen muchas más horas de cómputo que Colab y sin desconexiones. Las suscripciones de este tipo suelen tener cuota de GPU en 0, así que hay que comprobarlo antes de contar con ellas.

Cada integrante revisa la cuota de GPU de su suscripción en la región que convenga, solicita aumento si es posible, y el equipo decide: si hay GPU disponible, Azure sustituye a Colab en `issues/006-ejecutor-en-colab-o-azure.md`, `issues/010-noche-1-busquedas.md` e `issues/012-modelos-de-griego.md`; si no, el plan sigue con Colab sin cambios. El plan nunca debe depender de esta issue.

## Criterios de aceptación

- [ ] Para cada una de las 4 cuentas se registra la cuota de GPU disponible (familia, región y número de núcleos) o el rechazo de la solicitud.
- [ ] Si hay cuota: se arranca una VM con GPU, se comprueba `nvidia-smi` y se estima el costo por hora frente al saldo.
- [ ] Si hay cuota: se configura un apagado automático o presupuesto para no agotar el saldo por olvido.
- [ ] La decisión (Azure o Colab) queda anotada en esta issue y comunicada al equipo.
- [ ] Si Azure funciona, se propone por separado una issue para entrenar griego con el corpus completo; no se añade al plan sin esa decisión.

## Resultado (2026-11-25, cuenta del dueño de la GPU)

- Oferta: **Azure for Students** (tenant Instituto Politécnico Nacional). Las 4 cuentas del equipo tienen la misma oferta.
- El proveedor `Microsoft.Compute` estaba sin registrar (por eso `az vm list-usage` devolvía vacío); se registró.
- La política del tenant solo permite desplegar en: `westus`, `northcentralus`, `southcentralus`, `mexicocentral` y `eastus2`.
- Cuota en `southcentralus`, `westus` y `northcentralus` (idéntica en las tres):
  - `Standard NCASv3_T4 Family vCPUs` (T4): **límite 0**.
  - `Standard NVADSA10v5` (A10): límite 0.
  - `Standard NC` / `NC Promo` y `NV` / `NV Promo`: límite 6, pero son series antiguas (K80 / M60) retiradas por Azure; no se pueden crear.
  - `Standard NVSv4`: límite 2, GPU AMD sin CUDA; no sirve para PyTorch.
  - `Total Regional vCPUs`: límite 6.
- **Decisión: Azure no aporta GPU utilizable. El plan sigue con Colab sin cambios** en `issues/006-ejecutor-en-colab-o-azure.md`, `issues/010-noche-1-busquedas.md` e `issues/012-modelos-de-griego.md`. No hace falta revisar las otras 3 cuentas: misma oferta y mismo tenant.

## Bloqueado por

Ninguno - se puede empezar de inmediato.

## Historias de usuario cubiertas

Por número, del PRD de origen:

- Ninguna directamente (infraestructura); habilita el plan de cómputo del PRD.
