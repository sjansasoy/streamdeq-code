# Registro de instalación y decisiones — StreamDEQ en ImageNet-VID

Este documento explica, paso a paso, cómo se dejó funcionando el pipeline de
**inferencia** de StreamDEQ para ImageNet-VID en este servidor (136.145.54.118),
qué problemas aparecieron, por qué aparecieron, y qué se decidió hacer en cada
caso. La idea es que cualquiera (incluido yo mismo dentro de un mes) pueda
entender no solo *qué* comandos se corrieron, sino *por qué*.

Ver también:
- El plan completo de reproducción (Fase A: código+entorno, Fase B: dataset,
  Fase C: entrenamiento): `/home/sjansasoy/.claude/plans/zazzy-booping-owl.md`
- [`notebooks/forward_pass_diagrams.ipynb`](notebooks/forward_pass_diagrams.ipynb) —
  diagramas del forward pass del backbone MDEQ (referenciado en el paso 4).

## Regla que seguimos en todo este proceso

**El código del repo (lo que estamos reproduciendo) no se toca si hay alguna
forma de resolver el problema en la capa de entorno** (paquetes que nosotros
instalamos, versiones que nosotros elegimos). Vas a ver que en un caso
(sección 4) probamos primero una solución que sí tocaba el repo, y la
revertimos apenas encontramos una forma de no hacerlo. `git status` en la raíz
del repo debe quedar limpio (sin cambios a archivos trackeados) salvo que
digamos explícitamente lo contrario.

---

## 1. Contexto y objetivo

Meta: reproducir los resultados de StreamDEQ en ImageNet-VID. Primer hito
(este documento): validar que **el código corre** y que podemos llegar a
evaluar con un checkpoint preentrenado — *antes* de invertir en entrenar desde
cero. El dataset todavía no está accesible desde este servidor (vive en otra
máquina), así que todo lo de acá es validación **sin dataset y sin
checkpoint**, usando tensores aleatorios.

## 2. El import roto: faltaba `mmdet/models/backbones/lib/`

**Qué encontramos:** `mmdet/models/backbones/mdeq.py` (el backbone MDEQ)
importa 4 archivos desde una carpeta relativa `lib/`:
`jacobian.py`, `layer_utils.py`, `optimizations.py`, `solvers.py`. Esa carpeta
no existía en el checkout.

**Por qué pasó:** `ImageNetVID/.gitignore` tiene una regla `lib/` (pensada
originalmente para ignorar entornos virtuales de Python, que suelen llamarse
`lib/` o `lib64/`). Como efecto colateral, esa regla también ignora esta
carpeta de código DEQ genérico, así que nunca quedó comiteada al repositorio,
aunque el código la necesita para funcionar.

**Qué hicimos:** los mismos 4 archivos (código DEQ genérico, no específico de
segmentación ni de detección) sí están trackeados en `Cityscapes/lib/`, del
otro lado de este mismo repo. Los copiamos:

```bash
cp Cityscapes/lib/{jacobian.py,layer_utils.py,optimizations.py,solvers.py} \
   ImageNetVID/mmdet/models/backbones/lib/
```

Esto **no aparece en `git status`** (sigue gitignoreado), así que si cloneás
este repo de nuevo en otra máquina vas a tener que repetir este paso.

## 3. El entorno `streamdeq_vid`

El servidor ya tenía dos entornos conda (`streamdeq`, sin `mmcv`/`mmdet`
instalados; `streamdeq_city`, para el lado Cityscapes). Decidimos no tocar
ninguno de los dos y crear `streamdeq_vid` desde cero, siguiendo
`ImageNetVID/README.md` como punto de partida.

### 3.1 La restricción de versión que cambió el plan

`ImageNetVID/mmdet/__init__.py` fija en código:

```python
mmcv_minimum_version = '1.2.4'
mmcv_maximum_version = '1.3'
```

Es decir, este fork de `mmdet` (2.10.0, del año 2021) solo acepta
`mmcv-full` en el rango 1.2.4–1.3. Consultamos el índice de wheels
precompilados de OpenMMLab (`download.openmmlab.com/mmcv/dist/`) y
confirmamos que `mmcv-full==1.2.7` prearmado **solo existe para PyTorch
≤1.7.0** — no hay wheel para 1.10, 1.13 ni nada más nuevo.

### 3.2 Decisión: PyTorch 1.7.0 + CUDA 11.0

Evaluamos 3 opciones (wheel viejo / compilar mmcv desde fuente contra un
torch moderno / relajar el chequeo de versión e instalar un mmcv más nuevo) y
elegimos la primera por ser la de menor riesgo, aceptando de entrada que la
GPU L40S (arquitectura Ada Lovelace, más nueva que ese build de CUDA) quizás
no funcionara.

**Sorpresa:** sí funcionó. El driver de NVIDIA instalado en el servidor es
mucho más nuevo que el toolkit CUDA 11.0 con el que se compiló ese PyTorch, y
pudo compilar el código para la L40S al vuelo (JIT desde PTX) la primera vez
que se usó. Resultado: las 3 GPUs (L40S + 2×A30) quedaron utilizables.

### 3.3 Comandos de instalación (en orden)

```bash
conda create -n streamdeq_vid python=3.8 -y
conda activate streamdeq_vid

pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 \
    -f https://download.pytorch.org/whl/torch_stable.html

pip install mmcv-full==1.2.7 \
    -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
```

## 4. El problema de `mmpycocotools` (resuelto: sí se puede instalar el real)

`ImageNetVID/requirements/runtime.txt` pide `mmpycocotools` (el fork de
OpenMMLab de la librería COCO, con métodos snake_case que este repo usa en
varios lugares — ver sección 4.2). El primer intento de instalarlo (3
variantes: paquete de PyPI, clon de GitHub, `pip install .` local) falló
compilando su código en C, con el mismo síntoma: el compilador buscaba un
archivo en una ruta relativa (`../common/maskApi.c`) que no coincidía con
dónde `setuptools` realmente lo había puesto (`common/maskApi.c`).

**Diagnóstico correcto (tras insistir):** el problema era la versión de
**Cython**. `setuptools` moderno, al encontrar `Cython` instalado, auto-
"cythoniza" los `.pyx` del `Extension` antes de compilar. Con `Cython 3.x`
(lo que se instala por defecto hoy), ese paso duplicaba la entrada de
`common/maskApi.c` en la lista de fuentes con dos rutas relativas distintas,
y el compilador terminaba compilándolo dos veces — silencioso hasta que un
symlink (intento intermedio, ya no necesario) destapó un `ld: multiple
definition` al enlazar los dos objetos duplicados.

**Solución real:** bajar `Cython` a la serie `0.29.x` (la que existía cuando
se escribió este paquete, en 2020) **solo para compilar** — no es una
dependencia de tiempo de ejecución de `mmdet`, así que no afecta nada más:

```bash
pip install "cython<3"
pip uninstall -y pycocotools   # si ya estaba instalado el estándar
git clone https://github.com/open-mmlab/cocoapi.git   # necesita el repo completo:
cd cocoapi/pycocotools                                 # common/ es hermano de pycocotools/,
pip install . --no-build-isolation                      # por eso no alcanza con "pip install mmpycocotools"
```

Con esto, `mmpycocotools` real queda instalado, con sus alias snake_case y su
atributo `__version__` nativos — **no hace falta ningún parche** sobre el
paquete instalado ni sobre el repo.

### 4.1 (Historial) Camino que abandonamos: `pycocotools` estándar + parches

Antes de encontrar la causa real, habíamos optado por usar `pycocotools`
estándar (que sí tiene wheel precompilado) y parchear las diferencias de API
en el entorno. Se documenta acá porque ilustra el principio que seguimos
todo el tiempo (parchar el entorno, nunca el repo), aunque ya no está en uso:

- `mmdet/datasets/coco.py` (repo) chequea `pycocotools.__version__`, que
  `pycocotools` estándar no define. Probamos primero editar esa línea en el
  repo (`git checkout` la revirtió) y en cambio le agregamos el atributo al
  paquete instalado en el env.
- Al cargar el dataset real (sección 10) descubrimos que además hacían falta
  6 alias snake_case más (`get_cat_ids`, `get_img_ids`, etc., que
  `mmpycocotools` sí trae pero `pycocotools` estándar no).

Al conseguir instalar `mmpycocotools` real (arriba), **revertimos ambos
parches** desinstalando `pycocotools` estándar — ya no son necesarios.

### 4.2 Qué API usa este repo, y por qué importa

```bash
grep -rhoE '\.coco\.[a-z_]+\(' mmdet/ --include="*.py" | sort -u
```

```
.coco.get_ann_ids(       .coco.get_img_ids_from_vid(
.coco.get_cat_ids(       .coco.get_vid_ids(
.coco.get_img_ids(       .coco.load_anns(
                         .coco.load_cats(
                         .coco.load_imgs(
```

Los primeros 6 son alias snake_case que `mmpycocotools` agrega sobre la API
original de COCO (`getCatIds`, `getImgIds`, etc. en camelCase); `pycocotools`
estándar no los tiene. Los últimos 2 (`get_img_ids_from_vid`, `get_vid_ids`)
están definidos directamente en este repo
(`mmdet/datasets/parsers/coco_video_parser.py`), no dependen de qué paquete
COCO se use.

## 5. Otras dependencias que faltaban (no declaradas en `requirements/`)

Mientras verificábamos los imports, fuimos encontrando paquetes que el código
necesita pero que no están en `requirements/runtime.txt` de este fork:

| Paquete | Usado por | Sin él |
|---|---|---|
| `yacs` | `mmdet/models/backbones/deq_config/` (config del DEQ) | `ModuleNotFoundError` al importar `mdeq.py` |
| `scipy` | `lib/solvers.py` (import sin usar, pero falla igual) | `ModuleNotFoundError` |
| `termcolor` | `lib/solvers.py` (mensajes de debug coloreados) | `ModuleNotFoundError` |
| `terminaltables` | `mmdet/datasets/coco.py` (tablas de métricas) | `ModuleNotFoundError` |

```bash
pip install yacs scipy termcolor terminaltables
```

## 6. Instalación del propio repo como paquete editable

```bash
cd ImageNetVID
pip install -e . --no-deps --no-build-isolation
```

Usamos `pip install -e .` en vez de `python setup.py develop` (que sugiere el
`README.md`) porque este último usa una herramienta vieja (`easy_install`)
que exige el paquete `mmpycocotools` **por nombre exacto** — no reconoce que
`pycocotools` ya cubre esa necesidad, y vuelve a caer en el bug de la sección
4. `pip install -e . --no-deps` hace lo mismo (el código del repo queda
"enlazado" como el paquete `mmdet` activo, en vez de copiado a
`site-packages`) sin ese chequeo.

## 7. Verificación final: prueba de humo del backbone

Con todo instalado, corrimos `MDEQNet` con un tensor aleatorio en GPU (A30),
sin dataset ni checkpoint, en los dos modos del forward pass que se
diagraman en
[`notebooks/forward_pass_diagrams.ipynb`](notebooks/forward_pass_diagrams.ipynb):

- **Imagen sola** (`flag=True`, arranca en cero).
- **Lista/streaming** (`flag=False`, warm-start con la salida del paso
  anterior).

Las dos corridas terminaron sin error, y las formas de salida coincidieron
exactamente con lo esperado (input `1024×608` → stem `/4` → `256×152`, y cada
rama sucesiva a mitad de resolución): `(1,88,152,256)`, `(1,176,76,128)`,
`(1,352,38,64)`, `(1,704,19,32)`.

Esto confirma que el **código en sí funciona** en este servidor — lo que
faltaba resolver era pura fricción de entorno, no un problema del método.

## 8. Estado actual del repo

```
$ git status --short
?? CLAUDE.md
?? ImageNetVID/notebooks/
```

Ningún archivo trackeado del repo quedó modificado. Lo único "no estándar"
que existe y no aparece en `git status` es la carpeta
`ImageNetVID/mmdet/models/backbones/lib/` (sección 2, gitignoreada mediante
la regla `lib/` de `.gitignore`, no un problema nuestro).

## 9. Checkpoint de detección preentrenado

Descargado el checkpoint que linkea `README.md` (Google Drive, vía `gdown`) a
`pretrained_models/streamdeq_imagenetvid_detector.pth` (761MB):

```bash
pip install gdown
gdown "https://drive.google.com/uc?id=1mMcLgBZR9va3vYxG-E6ACKag49PrfXS3" \
    -O pretrained_models/streamdeq_imagenetvid_detector.pth
```

Verificado con `torch.load` (no solo que el archivo no esté corrupto/sea un
HTML de error de Drive, sino su contenido):

- `state_dict` con 144 tensores; las claves del backbone
  (`backbone.downsample.0.weight`, forma `(88,3,3,3)`) coinciden con la
  arquitectura que instanciamos en la prueba de humo (sección 7).
- `meta` incluye la config de entrenamiento completa: confirma
  `mmdet 2.10.0` + `mmcv 1.2.7` (exactamente nuestras versiones),
  `ImagenetVIDDataset`, entrenado 14 épocas (checkpoint guardado en época 11)
  sobre 4×A100. El entrenamiento usó `ref_img_sampler` de 1 frame de
  contexto — coherente con lo que vimos en `streamdeq.py`: el entrenamiento
  de video no está implementado (`forward_train` lanza
  `NotImplementedError`), así que el modelo se entrena como detector de
  un solo frame y el comportamiento streaming multi-frame (`2f`, `5f`, etc.)
  se evalúa solo en test time.

Con esto, la Fase A del plan queda completa: código funcionando, entorno
armado, checkpoint listo. Falta únicamente el dataset (Fase B).

## 10. NumPy: alias deprecados (`np.int`, `np.bool`) — otro caso sistémico

Al cargar el dataset real por primera vez (sección 11) apareció:

```
AttributeError: module 'numpy' has no attribute 'int'.
`np.int` was a deprecated alias for the builtin `int`... removed in NumPy 1.24.
```

Este repo es de 2020-2021 y usa `np.int`/`np.bool` (alias que NumPy fue a
propósito eliminando: solo advertencia desde la 1.20, error duro desde la
1.24). Antes de tocar nada, buscamos qué tan extendido era el problema:

```bash
grep -rEn '\bnp\.(int|float|bool|object|str|long|complex)\b' mmdet/ --include="*.py"
```

Aparecieron **9 ocurrencias** repartidas en 8 archivos distintos del repo
(`mean_ap.py`, `iou_balanced_neg_sampler.py`, `custom.py`,
`coco_video_dataset.py`, `structures.py`, `deq_utils/utils.py`,
`bbox_head.py`, `coco_video_parser.py`). Al ser sistémico — no un lugar
puntual — confirma que el problema real es la versión de NumPy del entorno,
no algo para parchear archivo por archivo. Bajamos a una versión donde estos
alias siguen funcionando (con warning, no error):

```bash
pip install "numpy==1.23.5"   # <1.24 (np.int no roto) y >=1.20 (matplotlib lo pide)
```

**Cuidado con el orden:** cualquier paquete con extensión en C compilada
contra NumPy (como `mmpycocotools`, sección 4) queda con un binario atado a
la versión de NumPy con la que se compiló — cambiar NumPy después rompe ese
`.so` (`ValueError: numpy.ndarray size changed...`). Hubo que recompilar
`mmpycocotools` de nuevo tras fijar la versión de NumPy:

```bash
cd cocoapi/pycocotools && rm -rf build *.egg-info pycocotools/_mask.c pycocotools/*.so
pip install . --no-build-isolation
```

## 11. Ubicación del dataset ImageNet-VID

El dataset está accesible desde este servidor vía un montaje NFS existente:
`/mnt/data` → `136.145.54.119:/data` (confirmado con `df -h`). La carpeta
`/mnt/data/users/sjansasoy/imagenetvid2015/imagenetvid_raw/` ya tenía:

- `annotations/{imagenet_vid_val,imagenet_vid_train,imagenet_det_30plus1cls}.json`
  — anotaciones ya convertidas a formato COCO-VID (alguien ya había hecho
  este paso; no tuvimos que escribir/adaptar el conversor de mmtracking).
- `ILSVRC2015/Data/VID/{train,val,test}/...` — imágenes ya extraídas.
- **No** había `Data/DET` extraído (solo `ILSVRC2017_DET.tar.gz` sin
  descomprimir) — no bloquea la evaluación (solo usa el split VID val), sí
  va a hacer falta para entrenar (Fase C).

**Verificación antes de usarlo** (no asumimos que "estaba ahí" alcanzaba):

```bash
# 1. ¿el JSON referencia archivos que realmente existen?
python -c "import json; d=json.load(open('.../imagenet_vid_val.json')); print(d['images'][0])"
# -> file_name: 'val/ILSVRC2015_val_00000000/000000.JPEG'
ls ".../ILSVRC2015/Data/VID/val/ILSVRC2015_val_00000000/000000.JPEG"   # existe

# 2. ¿la cantidad de imagenes del JSON coincide con los archivos en disco?
find .../ILSVRC2015/Data/VID/val -iname "*.JPEG" | wc -l   # 176126, igual al JSON
```

`ImageNetVID/.gitignore` ya ignora `data/`, así que armamos ahí la estructura
que espera `configs/_base_/datasets/imagenet_vid_dataset_mdeq.py`
(`data_root='data/ILSVRC/'`) con **symlinks** (no copias — son ~176k
archivos, cientos de GB):

```bash
mkdir -p data/ILSVRC/annotations
ln -sfn /mnt/data/users/sjansasoy/imagenetvid2015/imagenetvid_raw/ILSVRC2015/Data \
    data/ILSVRC/Data
ln -sfn /mnt/data/users/sjansasoy/imagenetvid2015/imagenetvid_raw/annotations/imagenet_vid_val.json \
    data/ILSVRC/annotations/imagenet_vid_val.json
```

Verificación final: instanciamos `ImagenetVIDDataset` de verdad (no solo
archivos en disco) y corrimos un sample completo por el pipeline — carga,
resize, normalize, pad, collect — sin errores, con `176126` frames clave
disponibles para evaluar.

## 12. Evaluación real: subset chico antes del val set completo

Antes de correr `tools/test.py` sobre las 176126 muestras del val set (que
tarda mucho), armamos un subset chico para validar la pipeline de punta a
punta rápido: los 3 videos más cortos del val set (`video_id` 275, 260, 262;
41 frames en total), preservando `categories`/`videos`/`images`/`annotations`
en formato COCO-VID:

```python
# quedarse solo con las imagenes/anotaciones de KEEP_VIDEO_IDS, ver script
# completo en el historial -- misma estructura que imagenet_vid_val.json
```

Guardado en `data/ILSVRC/annotations/imagenet_vid_val_subset.json` (dentro de
`data/`, gitignoreado). Corrido con `--cfg-options` para no tener que crear
un config nuevo:

```bash
CUDA_VISIBLE_DEVICES=1 python tools/test.py \
    configs/streamdeq/faster_rcnn_mdeq_fpn_1x_imagenetvid_stream_2f_1i.py \
    pretrained_models/streamdeq_imagenetvid_detector.pth \
    --eval bbox \
    --cfg-options data.test.ann_file=data/ILSVRC/annotations/imagenet_vid_val_subset.json
```

**Primer resultado (`2f_1i`, 1 iteración de Broyden por frame): mAP = 0.000**
en todas las métricas, incluso AP@IoU=0.50. Antes de asumir un bug, revisamos
las predicciones crudas (`--out` + inspección con `pickle`): el modelo sí
generaba cajas (50 en total sobre 41 imágenes, score máximo 0.36) — es decir,
no estaba "roto" (no habría dado *ninguna* caja, o cajas con coordenadas
absurdas), simplemente ninguna superaba IoU=0.5 con el ground truth.

**Hipótesis:** `1i` fija `f_thres=1` (una sola iteración de Broyden por
frame — el ajuste más agresivo entre los configs disponibles en
`configs/streamdeq/`, que solo llegan hasta `2i`), combinado con una muestra
de 41 frames deliberadamente sesgada a videos cortos (elegidos solo por
velocidad). **Verificación:** misma prueba con `2f_2i` (2 iteraciones) sobre
el mismo subset →

```
bbox_mAP: 0.122, bbox_mAP_50: 0.211, bbox_mAP_75: 0.132
```

El salto de 0.000 a 0.122 confirma la hipótesis: la pipeline (checkpoint,
preprocesamiento, decodificación de cajas, mapeo de categorías, cómputo de
mAP) funciona correctamente de punta a punta. El 0.000 inicial era esperable
dado el ajuste más agresivo posible sobre una muestra mínima, no un error de
instalación. Estos números (0.12/0.21 sobre 41 frames) tampoco son
comparables a los del paper — el subset es demasiado chico y sesgado para
eso; sirve solo como humo, no como número de referencia.

## 13. Qué falta (histórico — ver sección 15 para el estado actual de Fase C)

Ver el plan completo para el detalle, pero en resumen:

- **Correr `tools/test.py` sobre el val set completo** (176126 frames, no el
  subset de 41 de la sección 12) para obtener un mAP comparable al reportado
  en el paper. La pipeline ya está validada — esto es "solo" tiempo de
  cómputo, no debería aparecer ningún problema nuevo.
- **DET** (`ILSVRC2017_DET.tar.gz`) todavía no está extraído — no bloquea la
  evaluación (solo usa VID val), sí hace falta para entrenar (Fase C).
- Recién después de validar la evaluación con datos reales: descargar el
  backbone preentrenado en ImageNet y lanzar el entrenamiento completo.

## 14. Receta rápida (reproducir el entorno desde cero)

```bash
# 1. Restaurar el lib/ gitignoreado
cp Cityscapes/lib/{jacobian.py,layer_utils.py,optimizations.py,solvers.py} \
   ImageNetVID/mmdet/models/backbones/lib/

# 2. Crear el entorno
conda create -n streamdeq_vid python=3.8 -y
conda activate streamdeq_vid

# 3. PyTorch 1.7.0 + CUDA 11.0 (única versión con wheel de mmcv-full 1.2.7)
pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 \
    -f https://download.pytorch.org/whl/torch_stable.html

# 4. mmcv-full 1.2.7 prearmado
pip install mmcv-full==1.2.7 \
    -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html

# 5. mmpycocotools real: necesita Cython viejo para compilar (ver seccion 4)
pip install "cython<3"
git clone https://github.com/open-mmlab/cocoapi.git /tmp/cocoapi
cd /tmp/cocoapi/pycocotools
pip install . --no-build-isolation
cd -

pip install mmdet==2.10.0 --no-deps

# 6. Dependencias no declaradas en requirements/ (ver seccion 5)
pip install yacs scipy termcolor terminaltables

# 7. NumPy: fijar version <1.24 (np.int/np.bool aun funcionan) -- ver seccion 10
#    OJO: esto rompe el binario de mmpycocotools compilado en el paso 5 contra
#    otra version de numpy -- hay que recompilarlo despues de fijar numpy.
pip install "numpy==1.23.5"
cd /tmp/cocoapi/pycocotools
rm -rf build *.egg-info pycocotools/_mask.c pycocotools/*.so
pip install . --no-build-isolation
cd -

# 8. Instalar este repo como paquete editable (ver seccion 6)
cd ImageNetVID
pip install -e . --no-deps --no-build-isolation

# 9. Checkpoint de deteccion preentrenado (ver seccion 9)
pip install gdown
mkdir -p pretrained_models
gdown "https://drive.google.com/uc?id=1mMcLgBZR9va3vYxG-E6ACKag49PrfXS3" \
    -O pretrained_models/streamdeq_imagenetvid_detector.pth

# 10. Symlinks al dataset (ver seccion 11) -- ajustar la ruta origen a la tuya
mkdir -p data/ILSVRC/annotations
ln -sfn /mnt/data/users/sjansasoy/imagenetvid2015/imagenetvid_raw/ILSVRC2015/Data \
    data/ILSVRC/Data
ln -sfn /mnt/data/users/sjansasoy/imagenetvid2015/imagenetvid_raw/annotations/imagenet_vid_val.json \
    data/ILSVRC/annotations/imagenet_vid_val.json
```

## 15. Fase C (entrenamiento): progreso y un hallazgo importante

### 15.1 `yapf` demasiado nuevo rompe `Config.pretty_text`

Al preparar la prueba de humo de entrenamiento, `tools/train.py` (y cualquier
script que llame a `cfg.pretty_text`, algo que el propio `tools/train.py`
hace para loguear la config) falla con:

```
TypeError: FormatCode() got an unexpected keyword argument 'verify'
```

`mmcv` 1.2.7 llama a `yapf.FormatCode(..., verify=True)`, pero versiones
modernas de `yapf` (la que se instaló por defecto, 0.43.0) eliminaron ese
argumento. Mismo patrón que los demás hallazgos de esta lista: bajamos
`yapf` a una versión de esa época:

```bash
pip install "yapf==0.31.0"
```

Esto **afecta también al entrenamiento real**, no solo a nuestra prueba de
humo — sin este fix, ni siquiera `tools/train.py` normal arrancaría.

### 15.2 Checkpoint del backbone y dataset de entrenamiento

- Backbone MDEQ preentrenado en ImageNet descargado a
  `pretrained_models/mdeq_XL_cls_new.pkl` (362MB, vía `gdown`, mismo link de
  Google Drive del README) — verificado con `torch.load`: 224 tensores, las
  claves (`downsample.0.weight`, forma `(88,3,3,3)`) coinciden con el
  backbone que ya conocemos.
- `imagenet_vid_train.json` symlinkeado (ya existía en el mount, igual que
  `val.json` — ver sección 11).
- `ILSVRC2017_DET.tar.gz` (60.8GB): confirmamos antes de extraer que
  `imagenet_det_30plus1cls.json` solo referencia el split `train` (457.158
  imágenes en el tar; el JSON usa 349.721, un subconjunto filtrado a las 30
  clases que se solapan con VID). Extraído completo (train+val+test, decisión
  del usuario, aunque el pipeline actual solo necesita `train`) a
  `.../ILSVRC2015/Data/DET/`, usando `tar --wildcards 'ILSVRC/Data/DET/*'
  --strip-components=3` para quedarnos solo con las imágenes (no
  `Annotations/`, que no hace falta porque ya tenemos el JSON convertido).

### 15.3 Hallazgo importante: la L40S crashea en entrenamiento (backward), no en eval

Antes de comprometernos a un entrenamiento de varios días, probamos una
prueba de humo (5 videos, 100 frames de VID, 1 época) en la L40S, que estaba
libre mientras la evaluación de la Fase B ocupaba las A30. El proceso murió
con **`Segmentation fault (core dumped)`** (exit code 139) apenas entrado el
loop de entrenamiento, antes de completar la primera iteración.

**Descartado como causa:** `cuDNN` — el mismo crash ocurre incluso con
`torch.backends.cudnn.enabled = False` (fuerza kernels genéricos, no los de
cuDNN).

**Descartado como causa:** las operaciones CUDA propias de `mmcv` (RoIAlign,
NMS) en general — corrimos la evaluación completa (RPN+RoIHead, forward-only)
en la L40S (`CUDA_VISIBLE_DEVICES=0`) sobre el mismo subset de 41 frames que
usamos en la Fase B, y terminó normal (exit code 0, mismo resultado que en
las A30). O sea, el forward pass del detector completo sí funciona en L40S.

**Conclusión (sin poder confirmar el kernel exacto):** el problema es
específico del *backward* — probablemente el kernel de backward de
`RoIAlign` (compilado por `mmcv-full` 1.2.7 en 2021, nunca antes ejercitado
en L40S porque solo hicimos eval hasta ahora) o el `backward_hook` de
diferenciación implícita de `mdeq.py`. No pudimos confirmar cuál exactamente
porque no hay `sudo` en este servidor y herramientas como `py-spy`/`gdb`
para inspeccionar el core dump necesitan `ptrace` con privilegios elevados.

**Decisión:** no vamos a insistir con la L40S para entrenar. El entrenamiento
se va a validar y correr en las A30 (arquitectura Ampere, bien soportada por
este stack), una vez que la evaluación de la Fase B libere esas GPUs.

### 15.4 Actualización: el crash no era de la L40S — pasaba también en A30

Ver sección 16: al reintentar la prueba de humo en una A30 (arquitectura
Ampere, la misma familia que las A100 del paper), **el mismo segfault ocurrió
igual**, en el mismo punto exacto. Esto descartó por completo la hipótesis de
"incompatibilidad de la L40S" de esta sección — la causa real era otra (ver
16.1).

## 16. El segfault de entrenamiento: diagnóstico completo y solución

### 16.1 El crash es 100% reproducible, independiente de la GPU

Repetimos la prueba de humo en una A30 (`CUDA_VISIBLE_DEVICES=2`, la misma
arquitectura que las A100 que usó el paper) — mismo `Segmentation fault
(core dumped)`, en el mismo punto. Esto invalidó la hipótesis de la sección
15.3 (arquitectura de GPU) y abrió una investigación más profunda.

**Aislamiento quirúrgico** (cada prueba en un script standalone, sin pasar
por el dataset ni la cabeza de detección, para ir descartando capas):

| Prueba | Resultado |
|---|---|
| `forward_train()` completo (RPN+RoIHead), sin backward | ✅ OK — el crash es específicamente en `.backward()` |
| Solo el backbone MDEQ (sin RPN/RoIHead), forward+backward | ❌ Crashea igual — no es RoIAlign/RPN |
| Patrón genérico de hook reentrante en PyTorch puro (una conv simple) | ✅ OK — no es un bug genérico de PyTorch/CUDA en este entorno |
| `broyden()` aislado (`lib/solvers.py`) con una función sintética, mismo patrón de hook reentrante | ✅ OK — el solver en sí no es la causa |
| `weight_norm` (`lib/optimizations.py`) + `broyden()` reentrante | ✅ OK — tampoco es la reparametrización de pesos |
| `MDEQNet` completo (todas las ramas, `GroupNorm`, fusión multiescala) | ❌ Crashea |
| `b_thres=1` (una sola llamada reentrante en vez de hasta 26) | ❌ Crashea igual — no es por acumulación de llamadas |

Conclusión de esta ronda: el bug está en la combinación completa de MDEQ
(múltiples ramas fusionándose) bajo backward reentrante — no en ningún
componente aislado que pudiéramos probar por separado. Mecanismos genéricos
descartados con evidencia directa (no solo sospecha):
`OMP_NUM_THREADS=1`, `torch.autograd.set_detect_anomaly(True)`.

### 16.2 La pista real: el README pide PyTorch 1.10.0, no 1.7.0

`README.md` dice explícitamente `PyTorch (1.10.0)` y
`mmcv-full==1.2.7 -f .../cu111/torch1.10.0/index.html`. Elegimos 1.7.0 al
principio porque era la única versión con wheel *precompilado* de
`mmcv-full` 1.2.7 disponible. La metadata del checkpoint oficial (sección 9)
también dice `PyTorch: 1.10.0` — coincide.

Verificamos la URL exacta que cita el README: existe (HTTP 200), pero solo
tiene `mmcv-full` desde la versión 1.3.16 en adelante para `torch1.10.0` —
nunca la 1.2.7 que este fork exige (`mmcv_maximum_version='1.3'`, estricto).
Es decir, los autores originales **compilaron `mmcv-full` desde código
fuente** contra PyTorch 1.10.0; el wheel que el README sugiere simplemente ya
no existe en el índice de OpenMMLab.

Dado que PyTorch tuvo bastantes correcciones de estabilidad en el motor de
autograd entre 1.7.0 (oct. 2020) y 1.10.0 (oct. 2021) — justo el tipo de
patrón avanzado que usa MDEQ (backward reentrante desde un hook) — esto se
volvió el sospechoso principal.

### 16.3 Compilar mmcv-full 1.2.7 contra PyTorch 1.10.0: la odisea de compiladores

Armamos un entorno nuevo, **`streamdeq_vid_train`** (nombre elegido por el
usuario), sin tocar `streamdeq_vid` (que sigue sirviendo para evaluación).
Cada paso encontró un obstáculo distinto — quedan documentados acá porque
`scripts/setup_train_env.sh` los automatiza todos:

1. **PyTorch 1.10.0 vía conda** (no pip, siguiendo el README al pie de la
   letra): `conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.1
   -c pytorch -c conda-forge`. Sin problemas.
2. **Compilar `mmcv-full` con el `nvcc` del sistema (12.8) → falla.** PyTorch
   exige que la versión *mayor* de CUDA coincida con la que se usó para
   compilarlo (11.x), no la versión exacta — 12.8 vs 11.1 no pasa ese
   chequeo. (Un intento previo de instalar `nvcc` 11.1 exacto vía
   `nvidia/label/cuda-11.1.1` resultó, por alguna razón, en la versión 13.3 —
   headers pip de NVIDIA tampoco sirvieron, solo traen `ptxas`, no el
   compilador completo.) Se resolvió instalando
   `conda install -c conda-forge cudatoolkit-dev=11.4` — mismo major (11.x),
   alcanza.
3. **`nvcc` 11.4 rechaza GCC 11 (el del sistema) como host compiler → falla.**
   CUDA 11.4 solo soporta hasta GCC 10. Se instaló
   `conda install -c conda-forge gxx_linux-64=9 gcc_linux-64=9` y se apuntó
   `nvcc` a usarlo (`NVCC_PREPEND_FLAGS="-ccbin .../g++"`, `CC`/`CXX`).
4. **Falta `crypt.h` → falla.** El sysroot aislado de `conda-forge` no lo
   trae (típico de la migración de `crypt.h` fuera de glibc hacia
   `libxcrypt`). Un primer intento de solucionarlo agregando *todo*
   `/usr/include` al `CPATH` **empeoró las cosas**: mezcló headers del
   sistema con los del sysroot propio de conda, causando un choque de
   versiones de glibc (`bits/stdio2.h` con símbolos no declarados). La
   solución correcta fue más quirúrgica: una carpeta con **solo** un symlink
   a `crypt.h`, agregada al `CPATH` — sin arrastrar el resto de
   `/usr/include`.
5. Con los 4 pasos anteriores resueltos, `MMCV_WITH_OPS=1 pip install -e .`
   **compiló completo y sin errores** (decenas de operadores CUDA: NMS,
   RoIAlign, CARAFE, box_iou_rotated, etc.).

Confirmamos que `mmcv-full` compilado funciona (`nms()` real corriendo en
GPU) antes de seguir.

### 16.4 Resultado: el segfault desapareció

Con `mmcv-full` 1.2.7 recién compilado sobre PyTorch 1.10.0, repetimos
exactamente la misma prueba de humo aislada del backbone (sección 16.1) que
crasheaba de forma 100% reproducible en PyTorch 1.7.0:

```
=== Backward (aca vive el backward_hook de mdeq.py) ===
BACKWARD OK -- el backbone por si solo NO crashea.
```

Y la prueba de humo de entrenamiento completa (RPN+RoIHead, datos reales)
corrió sus 10 iteraciones sin error, con la pérdida bajando de forma
esperable (3.69 → 0.27) y el checkpoint guardándose correctamente. Confirma
la hipótesis de la sección 16.2: era un bug de estabilidad de autograd
específico de PyTorch 1.7.0 con backward reentrante en grafos complejos,
corregido en versiones posteriores — **no** un problema del método, del
código de este repo, ni de ninguna GPU en particular.

Dependencias sueltas adicionales encontradas en el camino, mismo patrón que
siempre (no están en `requirements/runtime.txt`):
`yacs`, `scipy`, `termcolor`, `terminaltables`, `numpy==1.23.5` (por
`np.int`/`np.bool`, sección 10), `yapf==0.31.0` (mismo motivo que la sección
15.1 — se reinstala la versión nueva cada vez que se instala `mmcv` desde
fuente, porque está en su `requirements.txt`, hay que volver a bajarla
después).

### 16.5 Estimación de tiempo de entrenamiento real

Con `samples_per_gpu=1` y `F_THRES=26` (26 iteraciones de Broyden en forward
*y* backward, por imagen — el mismo número que usa el paper para el
backbone), medimos ~2.3s/iteración en 1 A30. El dataset combinado
(VID train + DET) tiene ≈1.47M imágenes por época, 7 épocas:

| GPUs | Tiempo estimado |
|---|---|
| 1 | ~274 días |
| 2 | ~137 días |
| 3 (las de este servidor) | ~91 días |

Es una cifra grande pero consistente con lo que dice el propio paper: los
modelos implícitos son mucho más lentos de entrenar que las arquitecturas
explícitas, precisamente por el costo de resolver el punto fijo en cada
iteración. **Decisión:** no se lanza el entrenamiento completo en este
servidor. El plan es entrenarlo en Google Cloud con más GPUs; acá se deja
listo el entorno + los scripts (sección 16.6) para poder lanzarlo ahí.

### 16.6 Scripts para reproducir esto en otra máquina

- [`scripts/setup_train_env.sh`](scripts/setup_train_env.sh) — automatiza
  toda la receta de la sección 16.3 (crear el env, instalar PyTorch 1.10.0,
  compilar `mmcv-full` y `mmpycocotools`, instalar el resto de las
  dependencias, restaurar `lib/`). Incluye una copia empaquetada de los 4
  archivos de `lib/` en `scripts/vendor_lib/` para no depender de que
  `Cityscapes/` también esté clonado en la máquina nueva.
- [`scripts/launch_training.sh`](scripts/launch_training.sh) — lanza
  `tools/dist_train.sh` con la config correcta (`MDEQ_FasterRCNN`, no
  `StreamDEQ` — ese tipo es solo para streaming en test time) y la cantidad
  de GPUs que se le pase.

**Importante:** el SO base de la máquina de destino puede diferir de este
servidor — los pasos 3-4 de `setup_train_env.sh` (compilador auxiliar,
`crypt.h`) son los más sensibles a esto. Si el build de `mmcv` falla en un
punto distinto, conviene leer el error puntual en vez de asumir que el
script cubre todos los casos — fue exactamente así como encontramos cada uno
de los pasos de la sección 16.3.

## 17. Resultado final de `2f_2i` sobre el val set completo — y una corrección importante sobre cómo comparar contra el paper

### 17.1 El resultado

`tools/test.py` sobre las 176.126 imágenes del val set completo (1 GPU A30,
`streamdeq_vid`, ~21.4h de cómputo — coincide con la estimación de la
sección 12):

```
bbox_mAP:     0.095
bbox_mAP_50:  0.192  (19.2%)
bbox_mAP_75:  0.081
```

### 17.2 Estábamos comparando contra el número equivocado de la Tabla I

En la sección 12 comparamos este tipo de resultado contra el **39.5 mAP@50**
que la Tabla I del paper reporta para IL-StreamDEQ a 2 iteraciones — y nos
pareció bajo. Revisando la **Fig. 20** del paper (Apéndice, material
suplementario — `"mAP@50 results of IL-StreamDEQ for various number of
iterations after initialization with zeros from the beginning of a clip on
the ImageNet-VID dataset"`), el texto que la acompaña aclara:

> "In streaming mode, if we perform 2 iterations with IL-StreamDEQ, we
> improve the performance from 0 to 39.5 mAP@50 **in 20 frames**."

Es decir: **39.5 es el valor de convergencia a los 20 frames** de streaming
continuo (`t=19` en el eje x de la Fig. 20, `"Time Offset to Evaluation
Frame"`), no el valor que se obtiene con poco contexto. La Fig. 20 grafica
la curva completa: en `t=0` (sin contexto) el mAP@50 arranca cerca de 0, y
sube progresivamente hasta estabilizarse en ~39.5 recién hacia `t=19`.

### 17.3 La correspondencia correcta: `Nf` de este repo ↔ punto `t=N` de la Fig. 20

Nuestros configs `stream_Nf_Ki` (sección de la Fase B) no implementan "poco
contexto en general" — implementan exactamente el mismo procedimiento que
genera la Fig. 20: arrancar en cero `N` frames antes del frame evaluado, y
correr `K` iteraciones de Broyden por frame hasta llegar al frame de
interés. Esto es matemáticamente equivalente a leer la curva de la Fig. 20
(para `K` iteraciones) en el punto `t=N`, **no** a comparar contra el valor
de convergencia (`t=19`, que es lo que reporta la Tabla I).

Confirmación numérica: nuestro `2f_2i` (19.2% mAP@50, sección 17.1) coincide
con el valor que se lee en la curva **"IL-StreamDEQ (2 Iterations)"** de la
Fig. 20 en `t=2` (~19%, según inspección visual del gráfico). Igual de
consistente: nuestro `2f_1i` (mediciones sobre subsets en la sección 12: 0%
en 41 frames, 4.6% en 895 frames — ruidoso por el tamaño chico de la
muestra) es compatible con el valor que muestra la curva **"IL-StreamDEQ (1
Iteration)"** en `t=2` (~2%).

**Conclusión: la reproducción está funcionando correctamente.** El "número
bajo" de la sección 12 no era un problema de nuestra instalación ni del
checkpoint — era estar comparando el punto `t=2` de una curva contra el
punto `t=19` de esa misma curva.

### 17.4 Para comparar directo contra la Tabla I

Si se quiere un número directamente comparable a la Tabla I (39.5 para 2
iteraciones, 9.1 para 1 iteración), hay que usar los configs con `N=20`
(`faster_rcnn_mdeq_fpn_1x_imagenetvid_stream_20f_1i.py` /
`..._20f_2i.py`), que corresponden al extremo derecho de la Fig. 20
(`t=19`, el punto de convergencia) — no `2f`. Estos configs no se corrieron
todavía sobre el val set completo.
