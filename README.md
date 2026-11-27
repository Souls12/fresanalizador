# fresanalizador: control de calidad de fresas con segmentación por instancia

Proyecto del curso Aprendizaje Computacional, Ingeniería de Sistemas, profesor Oswaldo Vélez Langs.
Grupo: Ricardo Andrés Ramírez Durango, Michell Carvajal Gómez y Luisa Fernanda Zúñiga Fernández.

El programa recibe una foto con varias fresas, separa cada fresa con su propia máscara, dice si está sana o dañada con su probabilidad y muestra cuántas hay de cada una.

**Página publicada:** https://souls12--control-calidad-fresas-pagina.us-east.modal.run
(si nadie la ha usado en los últimos minutos, la primera foto tarda cerca de un minuto mientras arranca)

## Cómo funciona

1. **Grounding DINO tiny** (`IDEA-Research/grounding-dino-tiny`) busca las fresas con el texto `a strawberry.` y umbral 0.3. Después se quitan los cuadros repetidos, los que encierran varias fresas y los pedazos dentro de otra fresa.
2. **SAM base** (`facebook/sam-vit-base`) saca la máscara de cada fresa a partir de su cuadro.
3. Cada fresa se recorta con su cuadro agrandado un 10 % y pasa por un **EfficientNet-B0** de ImageNet ajustado con fresas. Si la probabilidad de dañada es 0.1 o más, o si el 10 % o más de la máscara se ve blanca o gris como el moho, la fresa se marca dañada.

## Resultados en las 20 fotos de prueba (286 fresas marcadas a mano)

| | a red strawberry | a strawberry |
|---|---|---|
| Fresas encontradas (IoU de 0.5 o más) | 225 | 240 |
| Precisión / recall / F1 de separación | 0.95 / 0.79 / 0.86 | 0.95 / 0.84 / 0.89 |
| IoU promedio de las encontradas | 0.89 | 0.89 |
| Dañadas encontradas de 23 | 8 | 8 |

El detalle, los experimentos y el análisis de errores están en el informe.

## Estructura

| Carpeta | Qué tiene |
|---|---|
| `1_...`, `2_...`, `2b_...`, `3_...ipynb` | Los cuadernos de Colab, en el orden en que se corren |
| `space/` | La página: `app.py`, el clasificador entrenado, ejemplos y `modal_app.py` para publicarla |
| `propias/` | Nuestras fotos, separadas en prueba, entrenamiento y no usadas, con `anotaciones_prueba.json` (COCO) y un `LEEME.txt` |
| `datos/` | `etiquetas_propias.csv` con la etiqueta de cada recorte de nuestras fotos y los recortes de Mendeley. Los zips grandes no se suben |
| `resultados/` | Resultados de los experimentos, de la evaluación y las figuras de errores |
| `docs/` | Informe, ficha de datos, bitácora, anexo de IA y el avance |

## Cómo reproducirlo

Los datasets públicos no están en el repositorio por su tamaño. Se descargan de Mendeley Data con los enlaces de la ficha de datos y se arma `fresas_datos.zip` como se explica ahí. `fotos_entrenamiento.zip` es la carpeta `propias/entrenamiento` comprimida.

1. En Google Drive, dentro de *Mi unidad*, crear la carpeta `fresas` y subir `fresas_datos.zip` y `fotos_entrenamiento.zip`.
2. En Colab con GPU T4 correr `1_recortar_fresas.ipynb`. Deja `recortes.zip` en Drive y guarda el avance cada 200 fotos.
3. Subir `datos/etiquetas_propias.csv` a la misma carpeta y correr `2_entrenar_clasificador.ipynb`. Entrena los cuatro experimentos y deja `clasificador.pt` y `clasificador.json`.
4. Subir `maduracion_recortes.zip` (la carpeta `datos/maduracion_recortes` comprimida) y correr `2b_ajustar_clasificador.ipynb`, que ajusta el clasificador con las fresas podridas de Mendeley.
5. Subir `fotos_prueba.zip` (la carpeta `propias/roboflow_prueba` comprimida) y correr `3_evaluar_programa.ipynb`, que escoge el umbral con la validación y mide el programa en las 20 fotos de prueba.

## Correr la página

En un computador con Python 3.12 o más nuevo:

```
pip install torch torchvision transformers==5.23.0 gradio==6.32.0 numpy pillow
python space/app.py
```

y abrir http://127.0.0.1:7860. La primera vez descarga Grounding DINO y SAM. Sin GPU cada foto tarda entre 30 y 60 segundos.

Para publicarla en Modal con una cuenta propia:

```
pip install modal
modal setup
modal deploy space/modal_app.py
```

## Versiones usadas

Python 3.14 en local y 3.12 en Colab y Modal, torch 2.15.0, torchvision 0.29.2, transformers 5.23.0, gradio 6.32.0.

## Licencias

Grounding DINO y SAM con Apache 2.0, EfficientNet-B0 de torchvision con BSD 3, y los tres datasets de Mendeley Data con CC BY 4.0. Las fotos y anotaciones propias son del grupo. El detalle está en el anexo de IA y en la ficha de datos.
