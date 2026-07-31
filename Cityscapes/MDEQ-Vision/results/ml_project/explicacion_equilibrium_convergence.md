# Convergencia hacia el equilibrio z\* en MDEQ: explicación completa

Este documento explica, de principio a fin, el análisis de convergencia del equilibrio `z*` en MDEQ: qué hace cada notebook, por qué se hace así, y qué significan los resultados. Cubre dos notebooks:

- **`equilibrium_convergence.ipynb`** — análisis a profundidad de **una** imagen.
- **`convergence_population_analysis.ipynb`** — la misma pregunta, extendida a las **500 imágenes** del conjunto de validación de Cityscapes.

Los detalles de problemas técnicos que surgieron en el camino (y cómo se resolvieron) están reunidos al final, en la sección [Problemas encontrados y sus soluciones](#problemas-encontrados-y-sus-soluciones), para no interrumpir la explicación principal.

## Contexto: ¿qué es z\* y por qué nos importa?

MDEQ no tiene un número fijo de capas. En vez de eso, define su representación interna como la solución de una ecuación de punto fijo: busca un estado `z*` tal que `f(z*) = z*` (aplicarle la red a `z*` te devuelve el mismo `z*`). Ese estado se encuentra iterativamente con un algoritmo llamado **Broyden**, que parte de un estado inicial `z⁽⁰⁾` y lo va corrigiendo paso a paso hasta que casi no cambia.

Todo el proyecto de "qué escala importa más" (el barrido principal, `results_master.csv`) se basa en manipular ese `z⁽⁰⁾` — es decir, en el modo `stream`, donde se reutiliza el equilibrio del frame anterior como punto de partida. Este análisis pregunta algo distinto y complementario, siempre en modo `baseline` (sin streaming, cada imagen arranca de cero): **una vez que el solver llega al equilibrio `z*`, ¿todas las 4 escalas convergen a la misma velocidad? ¿Todos los canales (features) dentro de una escala convergen igual, o hay unos que tardan mucho más que otros?**

Si quieres el mapa visual de todo esto antes de leer el código, están los 3 diagramas en:
https://claude.ai/code/artifact/a807994a-8f94-483e-bbcc-a9c8dd981627 (también en el Appendix de `analisis_resultados.ipynb`).

---

# Parte 1 — Una sola imagen (`equilibrium_convergence.ipynb`)

Todo lo que sigue en esta parte usa **una** imagen de validación (`frankfurt_000000_000294_leftImg8bit`) para poder mirar el problema de cerca antes de escalarlo a las 500.

## Sección 1 — Setup: GPU, rutas e imports

Fija qué GPU se va a usar (`CUDA_VISIBLE_DEVICES`, antes de `import torch`, porque una vez que PyTorch inicializa CUDA ya no se puede cambiar), y ajusta el directorio de trabajo.

El detalle importante es `os.chdir(...)`: el código del modelo (`lib/models/mdeq.py`) usa rutas relativas al directorio desde el que se ejecuta Python (`sys.path.append("lib/models")`, etc.), no relativas a dónde está el archivo. Todo el proyecto se corre normalmente desde `Cityscapes/MDEQ-Vision/`, pero el notebook vive una carpeta más adentro (`results/ml_project/`). Por eso, antes de importar el modelo, el notebook cambia su directorio de trabajo a `MDEQ-Vision/` — así el resto del código (rutas a `pretrained_models/...`, `data/...`, `experiments/...`) se comporta exactamente igual que al correr `python tools/seg_test.py` normalmente.

## Sección 2 — Cargar el config, construir el modelo, cargar el checkpoint

Lee el YAML `experiments/cityscapes/seg_mdeq_XL_sf_27i_gpu0.yaml` (usando el patrón YACS `defrost() → merge_from_file() → freeze()` que usa todo el proyecto). Sus valores clave:
- `DEQ.MODE: baseline` — sin streaming, cada llamada arranca fría. Es lo que necesitamos: un z\* que dependa solo de la imagen, no de ningún historial.
- `DEQ.F_THRES: 27` — 27 iteraciones de Broyden, el presupuesto que el proyecto original ya definió como "suficiente para converger bien" (más que las 1/2/4/8 del barrido principal).

Construye el modelo con `models.mdeq.get_seg_net(config)` (igual que `seg_test.py`) y carga el checkpoint (`pretrained_models/MDEQ_XL_Seg.pkl`), quitando el prefijo `"model."` de cada nombre de peso para que coincida con el modelo recién construido. Confirma **131 de 131 tensores** cargados. A diferencia de `seg_test.py`, no se envuelve el modelo en `nn.DataParallel`, para tener acceso directo a métodos internos (`_forward`, `segment`) sin el prefijo `.module.`.

## Sección 3 — Cargar una imagen de validación

Usa la misma clase `Cityscapes` (dataset) y un `DataLoader` con `batch_size=1` para tomar la primera imagen de `list/cityscapes/val.lst` (500 imágenes, lista plana, sin frames de calentamiento — no hacen falta en modo `baseline`).

Resultado: la imagen `frankfurt_000000_000294_leftImg8bit`, tensor de forma `(1, 3, 1024, 2048)`.

## Sección 4 — Correr el forward completo para obtener z\*

```python
z_star, jac_loss, sradius = model._forward(
    [image_gpu, None], train_step=-1, compute_jac_loss=False, f_thres=config.DEQ.F_THRES,
)
```

Se llama a `_forward()` directamente en vez de `model(...)` porque el `forward()` normal termina en `self.segment(y)` — devuelve la predicción de segmentación, no el estado interno. El segundo argumento (`None`) es el estado inicial: `None` significa "arranca en cero", el comportamiento de `mode='baseline'`.

**Resultado:** `z_star` es una lista de 4 tensores, uno por escala:

| Escala | Shape | Resolución |
|---|---|---|
| 0 | `(1, 88, 256, 512)` | más fina, menos canales |
| 1 | `(1, 176, 128, 256)` | media-alta |
| 2 | `(1, 352, 64, 128)` | media-baja |
| 3 | `(1, 704, 32, 64)` | más gruesa, más canales |

En total, 21,626,880 números (~86.5 MB en float32).

## Sección 5 — Guardar z\* y verificar el round-trip

`torch.save`/`torch.load` funcionan directo sobre una lista de tensores. Antes de guardar se hace `.detach().cpu()` (mover a CPU evita que el archivo dependa de en qué GPU física se calculó). `torch.equal(a, b)` confirma que el archivo recargado es **idéntico bit a bit** al original.

Se guarda en `results/ml_project/z_star_cache/frankfurt_000000_000294_leftImg8bit_f27_z_star.pt`.

## Sección 6 — ¿Cuánto influye cada feature en la segmentación final?

Una medida **estática**, complementaria a la velocidad de convergencia: cuánto *pesa* cada canal en la predicción final, usando solo los pesos ya entrenados (sin iterar nada).

`segment()` concatena las 4 escalas en 1320 canales (88+176+352+704) y las pasa por una convolución 1x1 (`Conv2d(1320, 1320)`), que matemáticamente es una combinación lineal de canales. La norma del peso de cada canal de entrada (sumada sobre las salidas) mide directamente cuánto puede mover ese canal el resultado final.

**Resultado:**

| Escala | # features | Peso promedio | Contribución promedio | % del total |
|---|---|---|---|---|
| 0 | 88 | 0.1332 | 0.00823 | 12.5% |
| 1 | 176 | 0.1251 | 0.00551 | 16.7% |
| 2 | 352 | 0.1239 | 0.00469 | 28.4% |
| 3 | 704 | 0.1163 | 0.00350 | 42.4% |

Por canal, la escala 0 tiene el mayor peso promedio; pero como la escala 3 tiene 8 veces más canales, su contribución **total** es la más grande (42%). Ninguna de las dos lecturas pone a la escala 1 arriba — y eso es esperable: es una medida estática y lineal, distinta de la importancia dinámica que ya conocemos del barrido principal (cuánto ayuda *reutilizar* una escala en el warm-start). No hay razón para que coincidan.

## Sección 7 — Radio espectral en el equilibrio

Mide qué tan rápido *debería* converger el solver cerca de z\*, linealizando la función del modelo alrededor de z\* y estimando su autovalor más grande con el método de la potencia (`power_method`, 150 iteraciones, ya implementado en `lib/models/mdeq_core.py` vía `spectral_radius_mode=True`).

**Resultado: radio espectral ≈ 2.37 — mayor a 1**, es decir, la zona alrededor de z\* es *localmente inestable* (no contractiva). Esto no invalida al solver de Broyden (usa una actualización cuasi-Newton, no aplicación repetida de la función, así que no necesita que el mapa contraiga), pero sí es relevante: significa que pequeñas diferencias numéricas cerca de z\* se amplifican en vez de atenuarse.

Un detalle de código a tener en cuenta si se reutiliza esta llamada: con `spectral_radius_mode=True`, `_forward()` retorna `func(z*)` (una aplicación extra de la función sobre z\*, necesaria para construir el grafo que usa `power_method`), no el `z*` crudo del solver. Son dos cantidades distintas por diseño — útil saberlo si se comparan resultados de ambos modos.

## Sección 8 — Mapa espacial de residuo

Una probada visual: en vez de resumir el error en un número, se muestra como una imagen 2D. Se compara el estado a `f_thres=2` (claramente no convergido) contra z\*, canal por canal, para el canal de mayor influencia de la escala 1 (Sección 6).

**Resultado:** el error no está disperso uniformemente — se concentra en estructuras verticales (postes, siluetas de peatones) y bordes de objetos. Las zonas de detalle fino son las que más le cuesta resolver al solver en pocas iteraciones.

## Sección 9 — MSE por canal a lo largo de las iteraciones de Broyden

La tarea central de este notebook: medir, canal por canal, qué tan rápido converge cada uno. El solver de Broyden (`lib/solvers.py::broyden()`) no guarda su estado interno en cada iteración — solo un residuo escalar — así que la forma de medir esto, consistente con cómo el resto del proyecto ya trata `f_thres`/`broyden_iterations` como un parámetro entero cualquiera (igual que el barrido principal con 1/2/4/8), es volver a llamar `_forward()` con `f_thres = 1, 2, ..., 27` y comparar cada resultado contra el z\* de la Sección 5.

```python
F_THRES_VALUES = list(range(1, config.DEQ.F_THRES + 1))  # 1..27
mse_per_scale = [[] for _ in range(4)]

with torch.no_grad():
    for ft in F_THRES_VALUES:
        z_ft, _, _ = model._forward([image_gpu, None], train_step=-1, compute_jac_loss=False, f_thres=ft)
        for i in range(4):
            per_channel_mse = (z_ft[i] - z_star[i]).pow(2).mean(dim=(0, 2, 3)).cpu()
            mse_per_scale[i].append(per_channel_mse)
```

`.mean(dim=(0, 2, 3))` promedia sobre el batch y las dos dimensiones espaciales, dejando un número por canal. Al final, por escala, hay una matriz `(27, num_canales_de_esa_escala)`.

Un detalle importante para las gráficas siguientes: Broyden tiene su propio criterio de parada temprana (`if new_objective < eps: break`, en `solvers.py`). Para esta imagen, el solver ya converge (bit a bit) desde `f_thres=26` — llamarlo con 26 o con 27 da exactamente el mismo resultado.

## Sección 10 — Velocidad de convergencia promedio por escala

Promedia el MSE por canal dentro de cada escala, dando una curva por escala.

**Resultado:** la escala 0 (resolución más fina) converge más rápido en promedio; la escala 3 (más gruesa) es la más lenta; las escalas 1 y 2 quedan en el medio — una curva descendente casi recta en escala logarítmica.

## Sección 11 — Variación de convergencia dentro de cada escala

Un promedio puede esconder mucho — una escala podría converger rápido en promedio pero tener canales atípicos muy lentos. Se grafica, un panel por escala (mismo eje Y para comparar), la mediana de sus canales más una banda del percentil 10 al 90.

**Resultado:** la escala 0 tiene la mayor dispersión canal a canal — algunos de sus canales se estabilizan casi de inmediato, otros se quedan bastante atrás del promedio de su propia escala.

## Sección 12 — ¿Los canales lentos importan más o menos?

Cruza la velocidad de convergencia con la influencia de cada canal (Sección 6). Se usa el MSE en `f_thres=2` como indicador de "qué tan lento es este canal", comparado contra `contribution` de la Sección 6, con dos correlaciones (Pearson y Spearman, ya que los valores abarcan ~12 órdenes de magnitud y Spearman es más robusta a esa dispersión):

```python
pearson_r = np.corrcoef(early_mse, contribution)[0, 1]      # 0.755
spearman_r, _ = spearmanr(early_mse, contribution)           # 0.844
```

**Resultado:** los canales que todavía están lejos de converger en `f_thres=2` tienden a ser también los de mayor contribución a la segmentación final — los canales más lentos en asentarse **no** son los que la cabeza de segmentación ignora.

## Sección 13 — Las 1320 features de una sola vez

Un punto por canal por `f_thres` (1320 canales × 27 iteraciones), un panel por escala (la escala 3 tiene 8 veces más canales que la escala 0, así que separarlas en paneles evita que una domine visualmente a las demás):

```python
x = np.repeat(F_THRES_VALUES, n_channels) + jitter  # ruido horizontal para que se vea como nube
y = scale_mse.flatten()
ax.scatter(x, y, s=3, alpha=0.06, color=colors[i])  # alpha bajo: más denso donde hay más canales
```

**Resultado:** una nube densa que desciende de forma bastante uniforme (la mayoría de los canales siguen una curva de convergencia parecida), más una cola dispersa de canales que caen hasta 1e-9 – 1e-11 muy temprano — coincide con los "canales casi instantáneos" que ya veíamos en la Sección 12.

## Sección 14 — Distribución de velocidad de convergencia por escala

Reduce la trayectoria completa de cada canal a un solo número: la menor `f_thres` en la que su MSE relativo (contra su propio valor en `f_thres=1`) cae por debajo del **10%**. Ese número se grafica como un punto por canal, agrupado por escala.

```python
CONVERGE_THRESHOLD = 0.1
iters_to_converge = [primera f_thres donde MSE(t)/MSE(1) < 0.1, por canal]
```

**Resultado:** medianas de 10, 11, 12 y 16 iteraciones para las escalas 0, 1, 2 y 3 — coincide con el orden de la Sección 10. Pero lo más informativo es la dispersión: los puntos de la escala 0 están agrupados (7-14), mientras que los de la escala 3 se extienden desde un dígito hasta 26 — la escala 3 no solo es más lenta en promedio, ahí viven los canales genuinamente atípicos.

## Resumen de la Parte 1

- Escala 0 converge más rápido, escala 3 más lento; escalas 1 y 2 en el medio.
- Dentro de cada escala hay muchísima heterogeneidad — no todos los canales de una escala convergen igual.
- Los canales lentos tienden a ser también los más influyentes en la segmentación (correlación 0.76-0.84).
- **La escala 1 (la más importante para el warm-start, según el barrido principal) no es la más lenta en converger por sí sola** — esa es la escala 3. Esto sugiere que el valor de reutilizar la escala 1 no viene de que tarde más en resolverse sola, sino probablemente de cómo su error se propaga a las demás escalas al fusionarse (`fullstage`) — una buena pregunta para explorar después, fuera del alcance de este notebook.
- Todo lo anterior es sobre **una sola imagen** — el siguiente paso natural es ver si se sostiene en todo el conjunto de validación.

---

# Parte 2 — Las 500 imágenes (`convergence_population_analysis.ipynb`)

## El script de extracción: `scripts/ml_project/extract_convergence_mse.py`

Repetir el análisis de la Parte 1 para las 500 imágenes de validación es costoso (cada imagen necesita ~28 pasadas del solver: una para z\*, 27 más para el barrido de MSE), así que se separó en dos piezas:

1. **Un script** (`extract_convergence_mse.py`, corrido una vez en una sesión `tmux`, ~6 horas) que, por cada imagen: calcula (o reutiliza si ya existe) su z\* — guardado en `z_star_cache/`, igual que en la Sección 5 —, luego corre el barrido `f_thres=1..27` comparando contra ese z\*, y guarda el MSE resultante en `results/ml_project/convergence_mse_all_images.pt`. Es **reanudable**: si se interrumpe, al volver a correrlo salta las imágenes ya procesadas.
2. **Este notebook**, que solo lee ese archivo ya calculado — no toca el modelo ni la GPU — y construye las gráficas. Puede volver a correrse en segundos cada vez que el script avance.

Se decidió guardar tanto z\* (86.5 MB × 500 ≈ 43 GB) como el MSE derivado (mucho más chico, ~71 MB): el costo real está en *calcular* z\*, no en guardarlo, así que tenerlo cacheado evita tener que recalcularlo si hace falta para algo más adelante.

## Sección 1 — Cargar los resultados

```python
results = torch.load(RESULTS_PATH, map_location="cpu", weights_only=False)
mse_stack = torch.stack(results["mse"])  # (num_images, 27, 1320)
```

`SCALE_CHANNELS = [88, 176, 352, 704]` se deja fijo (es una constante de la arquitectura), ya que este notebook no carga el modelo. El chequeo de cordura (MSE promedio en `f_thres=27` debe ser ~0) se confirma con las 500 imágenes.

## Sección 2 — Convergencia por escala, con variabilidad entre imágenes

La versión poblacional de la Sección 10: para cada imagen, se promedia el MSE sobre los canales de una escala, y se grafica la **mediana entre imágenes** más una banda de percentil 10-90.

**Resultado:** el orden se mantiene igual de estable que con una imagen (escala 0 más rápida, escala 3 más lenta), y la escala 3 se separa claramente del resto desde `f_thres≈8` en adelante.

## Sección 3 — Dispersión por canal, promediando sobre imágenes

La versión poblacional de la Sección 11: se promedia primero cada canal sobre las 500 imágenes, y luego se muestra la mediana + percentil 10-90 **entre canales**, un panel por escala.

**Resultado:** la heterogeneidad "hay canales rápidos y lentos en cada escala" no se diluye al promediar sobre todo el dataset — es una propiedad estructural del modelo, no ruido de una imagen particular.

## Sección 4 — Distribución robusta de velocidad de convergencia

La versión poblacional de la Sección 14: mismo criterio (10% del valor en `f_thres=1`), aplicado a la trayectoria ya promediada sobre las 500 imágenes.

**Resultado:** medianas de **10, 11, 11 y 21** iteraciones para las escalas 0-3 — prácticamente iguales a las de una sola imagen, señal de que el patrón se estabiliza rápido y no era casualidad.

## Sección 5 — ¿Es la velocidad de convergencia una propiedad fija del canal, o depende de la imagen?

Una pregunta que una sola imagen no puede responder: para cada canal, se calcula "iteraciones para llegar al 10%" **por separado en cada una de las 500 imágenes**, y se mide qué tan dispersa es esa cifra entre imágenes (su desviación estándar).

**Resultado:** desviaciones estándar de **1.14, 1.45, 2.18 y 4.16** iteraciones para las escalas 0-3. Los canales de la escala 0 son rápidos **y** predecibles (casi no importa qué haya en la imagen); los de la escala 3 son lentos **y**, además, su velocidad depende mucho del contenido de cada imagen. La relación entre "velocidad promedio" y "qué tan predecible es" es clara y casi monótona: a mayor velocidad, mayor también su variabilidad de imagen a imagen.

## Sección 6 — ¿Cuándo para cada imagen su propio solver?

Broyden resuelve las 4 escalas como **un solo vector concatenado**, así que su criterio de parada temprana se activa para las 4 escalas **al mismo tiempo**, dentro de una misma imagen — pero el punto exacto varía de imagen a imagen. Este histograma muestra esa distribución directamente: para cada imagen, la primera `f_thres` en la que su MSE (para cualquier escala) es exactamente `0.0`.

**Resultado:** el mínimo fue 14, el máximo 27, con mediana en 27 y media en 26.1. Alrededor de **302 de las 500 imágenes (60%)** nunca activan el criterio interno de parada del solver dentro del presupuesto de 27 iteraciones — usan el presupuesto completo sin que el solver se declare "convergido" según su propia regla. El resto (~200) sí lo logra, en algún punto entre 14 y 26.

## Sección 7 — Tasa de convergencia por canal, sin depender de un umbral

Un umbral (como el 10% de las secciones anteriores) da solo una foto en un punto. Aquí se mide, para cada canal, la **pendiente** de `log10(MSE)` vs `f_thres` (cuántos órdenes de magnitud pierde el error por iteración) — una tasa continua, no una foto. El ajuste usa solo `f_thres=1` a `10`, un rango seguro antes de que empiece a intervenir la parada temprana de ninguna imagen (el mínimo encontrado en la Sección 6 fue 14).

**Resultado:** tasas medianas de **0.118, 0.114, 0.107 y 0.075** unidades log10 por iteración para las escalas 0-3 — mismo orden que todo lo anterior, ahora expresado como una velocidad continua en vez de un punto de cruce.

## Sección 8 — Ver un canal individual

Una función (`plot_channel(c)`) para inspeccionar cualquiera de los 1320 canales por su índice global: `channel_to_scale(c)` traduce ese índice a `(escala, índice local)`, usando el mismo orden de concatenación de todo el proyecto (`boundaries = [0, 88, 264, 616, 1320]` — canal 0 es el primero de la escala 0, canal 88 el primero de la escala 1, etc.). El gráfico muestra las 500 trayectorias individuales (líneas finas y transparentes) más la curva promedio en negro.

Dos ejemplos de referencia: el canal más lento de todos (canal 1081, escala 3, tasa 0.034) tiene una curva mucho más irregular y se mantiene "ancho" por más tiempo; el más rápido (canal 606 — de la escala 2, no de la 0) decae de forma suave y agresiva hasta valores mucho más bajos.

## Sección 9 — Todos los canales de una escala, superpuestos

Una sola gráfica con las N curvas de una escala (una por canal), coloreadas según su tasa de convergencia (Sección 7, de más lenta=morado a más rápida=amarillo). Cada curva se normaliza contra su propio valor en `f_thres=1` (`relative_mse`, la misma normalización de la Sección 4/14) — sin esto, la comparación se dominaba por la escala de activación propia de cada canal en vez de por la forma de su decaimiento; normalizando, todas arrancan en 1.0 y se abren en abanico hacia abajo según su velocidad, y el color se alinea mucho mejor con la altura de cada curva.

## Sección 10 — La misma vista, solo para las imágenes "difíciles"

Usando el resultado de la Sección 6, se repite la Sección 9 pero promediando solo sobre las ~302 imágenes que nunca activaron la parada temprana del solver. Los colores se mantienen fijos (de la Sección 7, calculada sobre las 500 imágenes) para que un mismo canal se vea del mismo color en ambas vistas.

**Resultado:** la historia general se mantiene (mismo abanico, mismo orden de velocidad), pero aparece un pequeño rebote hacia arriba cerca de `f_thres≈22-25` en varias curvas lentas, más marcado que en el promedio de las 500 imágenes completas — ver la nota técnica sobre esto en la sección de problemas y soluciones, al final.

## Sección 11 — Repetir para las escalas 1, 2 y 3

Mismo par de vistas (500 imágenes / 302 "difíciles") para las tres escalas restantes, reusando las funciones ya definidas.

**Resultado:** las escalas 1 y 2 se parecen a la 0. La **escala 3 es notablemente distinta**: el grupo más lento apenas cae por debajo de 0.15-0.2 en toda la corrida (mucho más alto que en las demás escalas), y el "rebote" inicial es mucho más marcado — varias curvas llegan a superar su propio valor de `f_thres=1` (hasta 2-3 veces) en las primeras iteraciones antes de empezar a decaer en serio.

## Resumen de la Parte 2

Con las 500 imágenes, la historia de la Parte 1 se confirma y se refuerza:
- La escala 0 es la más rápida, la más consistente entre imágenes, y la de menor variabilidad interna.
- La escala 3 es la más lenta, la más impredecible de imagen a imagen, y la que tiene el "rebote" inicial más pronunciado.
- Tres métricas independientes (iteraciones al 10%, tasa de decaimiento, consistencia entre imágenes) apuntan todas a la misma conclusión.
- **La escala 1 sigue sin ser la más lenta en converger**, pese a ser la más importante para el warm-start — la pregunta de *por qué* importa sigue abierta, y probablemente pertenece a un análisis en modo `stream` (fuera del alcance de este par de notebooks, que se mantienen deliberadamente en modo `baseline`).

---

# Problemas encontrados y sus soluciones

Registro breve de los tropiezos técnicos durante este trabajo, para tenerlos en cuenta si algo similar vuelve a aparecer.

| Problema | Cómo se detectó | Solución |
|---|---|---|
| Se pensó que z\* no era reproducible entre corridas (posible no-determinismo de GPU) | Al comparar z\* (Sección 4) contra el resultado de `spectral_radius_mode=True` (Sección 7), los valores no coincidían | Se confirmó que `spectral_radius_mode=True` devuelve `func(z*)`, no `z*` — son cantidades distintas por diseño, no un problema de reproducibilidad. Se verificó corriendo el mismo cómputo en procesos separados: sin ningún ajuste especial de cuDNN, los resultados coincidían bit a bit. |
| Un ajuste de determinismo de cuDNN (`cudnn.deterministic=True`) se agregó por precaución, pensando que hacía falta | Al confirmar lo anterior, se probó explícitamente si ese ajuste cambiaba algo | No hacía falta — se quitó del notebook. |
| Un heatmap (canal × iteración, color = MSE) para ver las 1320 features a la vez salía comprimido, sin importar el ajuste de escala de color (piso arbitrario, percentiles, orden por velocidad) | Al inspeccionar visualmente la gráfica en cada iteración de diseño | Se reemplazó por un scatter/nube de puntos (donde el valor es una posición en el eje Y, no un color) — funciona porque no necesita un único rango de color para todo el gráfico. |
| Al graficar el MSE promedio por escala en escala logarítmica, las curvas parecían "cortarse" antes de la iteración 27 | El corte coincidía con el hallazgo de que Broyden converge exactamente (bit a bit) antes de la iteración 27 | Se determinó que no era un error: `log(0)` no existe, así que un MSE exactamente `0.0` no se puede graficar en escala log — el "corte" es ese mismo hallazgo haciéndose visible. Se dejó así (sin agregar un piso artificial), documentando la causa en el texto. |
| Al correr el barrido de 500 imágenes, la memoria de GPU reportada por `nvidia-smi` se veía alta y estable | Se agregó un diagnóstico de memoria (`torch.cuda.memory_allocated()` vs `.memory_reserved()`) por imagen | Confirmado como comportamiento normal de PyTorch: la memoria "reservada" no se libera de vuelta al sistema entre imágenes (para evitar pedirla de nuevo cada vez), pero la memoria realmente en uso (`allocated`) se mantenía constante — no había fuga real. |
| En la Sección 10 (imágenes "difíciles"), aparece un pequeño aumento del MSE cerca de `f_thres≈23` antes de la caída final | Visible al graficar la superposición de canales | El solver de Broyden en este proyecto corre sin búsqueda de línea (`ls=False` por defecto en `lib/solvers.py::broyden()`, y `mdeq_core.py` nunca pasa `ls=True`), así que cada paso se aplica completo sin verificar que reduzca el error — un comportamiento esperado de un método cuasi-Newton sin esa salvaguarda, más visible en las imágenes donde el solver todavía está corrigiendo activamente. |
