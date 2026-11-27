---
title: Control de calidad de fresas
emoji: 🍓
colorFrom: red
colorTo: green
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12.12"
app_file: app.py
pinned: false
license: cc-by-4.0
---

# Control de calidad de fresas

Proyecto del curso Aprendizaje Computacional. La página recibe una foto con varias fresas, separa cada fresa con su contorno y dice si está sana o dañada, junto con el conteo de cada una.

## Cómo funciona

1. **Grounding DINO** (IDEA-Research/grounding-dino-tiny) busca las fresas con el texto "a strawberry." y umbral 0.3.
2. **SAM** (facebook/sam-vit-base) saca la máscara de cada fresa a partir de su caja.
3. **EfficientNet-B0**, preentrenado en ImageNet y ajustado con recortes de fresas, dice si cada recorte está sano o dañado. El umbral de 0.1 se escogió con la validación de nuestras fotos.

## Datos de entrenamiento del clasificador

- Spoiled and fresh fruit inspection dataset, Universidad Militar Nueva Granada, CC BY 4.0, DOI 10.17632/6ps7gtp2wg.1
- Fresh and Rotten Fruits Dataset, Universidad Jahangirnagar, CC BY 4.0, DOI 10.17632/bdd69gyhv8.1
- Comprehensive Dataset on Ripening Stages of Strawberries and Avocados, CC BY 4.0, DOI 10.17632/zysvgmxcyz.1
- Fotos propias tomadas por el grupo

## Resultados en las 20 fotos de prueba (286 fresas marcadas a mano)

- Separar las fresas con "a strawberry.": precisión 0.95, recall 0.84 e IoU promedio de 0.89
- Diferenciar sana de dañada: encuentra 8 de 23 dañadas y marca varias sanas como dañadas, porque los daños de nuestras fresas son más pequeños que los de los datasets públicos
- Además del clasificador, una fresa se marca dañada si el 10 % o más de su máscara se ve blanca o gris, para atrapar el moho grande

## Autores

Ricardo Andrés Ramírez Durango, Michell Carvajal Gómez y Luisa Fernanda Zúñiga Fernández
