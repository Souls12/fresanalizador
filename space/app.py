import json
import os
import time

try:  # en Hugging Face con ZeroGPU hay que importar spaces antes que torch
    import spaces
    GPU = spaces.GPU(duration=90)
except ImportError:  # en un computador normal no existe y se corre sin GPU
    GPU = lambda f: f

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision import models, transforms
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor

TEXTO = "a strawberry."
UMBRAL_DINO = 0.3
MARGEN = 0.10
LADO = 256
LADO_MAX = 1600
COLOR = {"sana": (0, 200, 0), "danada": (230, 0, 230)}
MOHO_S, MOHO_V, MOHO_FRACCION = 0.25, 0.55, 0.10

disp = "cuda" if torch.cuda.is_available() else "cpu"
dino_proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
dino = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(disp).eval()
sam_proc = SamProcessor.from_pretrained("facebook/sam-vit-base")
sam = SamModel.from_pretrained("facebook/sam-vit-base").to(disp).eval()

os.chdir(os.path.dirname(os.path.abspath(__file__)))
cfg = json.load(open("clasificador.json"))
clasif = models.efficientnet_b0()
clasif.classifier[1] = nn.Linear(clasif.classifier[1].in_features, 2)
clasif.load_state_dict(torch.load("clasificador.pt", map_location=disp))
clasif = clasif.to(disp).eval()
tf = transforms.Compose([transforms.Resize((cfg["tamano"], cfg["tamano"])), transforms.ToTensor(),
                         transforms.Normalize(cfg["media"], cfg["desviacion"])])


def detectar(img, texto=TEXTO):
    ent = dino_proc(images=img, text=texto, return_tensors="pt").to(disp)
    with torch.no_grad():
        sal = dino(**ent)
    try:
        res = dino_proc.post_process_grounded_object_detection(
            sal, ent.input_ids, threshold=UMBRAL_DINO, text_threshold=UMBRAL_DINO, target_sizes=[img.size[::-1]])[0]
    except TypeError:
        res = dino_proc.post_process_grounded_object_detection(
            sal, ent.input_ids, box_threshold=UMBRAL_DINO, text_threshold=UMBRAL_DINO, target_sizes=[img.size[::-1]])[0]
    return res["boxes"].cpu().tolist(), res["scores"].cpu().tolist()


def segmentar(img, cajas):
    ent = sam_proc(img, input_boxes=[cajas], return_tensors="pt").to(disp)
    with torch.no_grad():
        sal = sam(**ent, multimask_output=False)
    return sam_proc.image_processor.post_process_masks(
        sal.pred_masks.cpu(), ent["original_sizes"].cpu(), ent["reshaped_input_sizes"].cpu())[0][:, 0].numpy()


def iou_cajas(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area = lambda c: (c[2] - c[0]) * (c[3] - c[1])
    return inter / (area(a) + area(b) - inter + 1e-9)


def limpiar_cajas(cajas, puntajes):
    orden = sorted(range(len(cajas)), key=lambda i: -puntajes[i])
    def dentro(i, j):
        a, b = cajas[i], cajas[j]
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        return ix * iy / ((a[2] - a[0]) * (a[3] - a[1]) + 1e-9)
    def centro_en(j, i):
        x0, y0, x1, y1 = cajas[i]
        return x0 <= (cajas[j][0] + cajas[j][2]) / 2 <= x1 and y0 <= (cajas[j][1] + cajas[j][3]) / 2 <= y1
    quedan = []
    for i in orden:
        if all(iou_cajas(cajas[i], cajas[j]) < 0.5 for j in quedan):
            quedan.append(i)
    quedan = [i for i in quedan if sum(centro_en(j, i) for j in quedan if j != i) < 2]
    return [i for k, i in enumerate(quedan) if all(dentro(i, j) < 0.8 for j in quedan[:k])]


def recorte_caja(img, caja):
    x0, y0, x1, y1 = caja
    mx, my = (x1 - x0) * MARGEN, (y1 - y0) * MARGEN
    r = img.crop((max(0, int(x0 - mx)), max(0, int(y0 - my)),
                  min(img.width, int(x1 + mx) + 1), min(img.height, int(y1 + my) + 1)))
    r.thumbnail((LADO, LADO))
    return r


def prob_danada(recortes):
    with torch.no_grad():
        x = torch.stack([tf(r) for r in recortes]).to(disp)
        return torch.softmax(clasif(x), 1)[:, 1].cpu().numpy()


def fraccion_moho(hsv, mascara):
    blanco = (hsv[..., 1] < MOHO_S) & (hsv[..., 2] > MOHO_V) & mascara
    return blanco.sum() / max(1, mascara.sum())


@GPU
def analizar(img, umbral, texto):
    if img is None:
        raise gr.Error("Sube una foto primero.")
    t0 = time.time()
    img = img.convert("RGB")
    img.thumbnail((LADO_MAX, LADO_MAX))
    texto = texto.strip() or TEXTO
    cajas, puntajes = detectar(img, texto if texto.endswith(".") else texto + ".")
    if not cajas:
        return img, "No se encontró ninguna fresa en la foto.", []
    cajas = [cajas[i] for i in limpiar_cajas(cajas, puntajes)]
    mascaras = segmentar(img, cajas)
    p = prob_danada([recorte_caja(img, c) for c in cajas])
    hsv = np.array(img.convert("HSV")).astype(float) / 255
    moho = [fraccion_moho(hsv, m) for m in mascaras]
    clases = ["danada" if x >= umbral or f >= MOHO_FRACCION else "sana" for x, f in zip(p, moho)]

    arr = np.array(img).astype(float)
    for m, c in zip(mascaras, clases):
        arr[m] = arr[m] * 0.5 + np.array(COLOR[c]) * 0.5
    salida = Image.fromarray(arr.astype(np.uint8))
    d = ImageDraw.Draw(salida)
    tam = max(18, img.width // 45)
    for j, (c, m) in enumerate(zip(cajas, mascaras), 1):
        ys, xs = np.nonzero(m)
        x, y = (xs.mean(), ys.mean()) if len(xs) else ((c[0] + c[2]) / 2, (c[1] + c[3]) / 2)
        d.text((x, y), str(j), fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0), font_size=tam, anchor="mm")

    sanas, danadas = clases.count("sana"), clases.count("danada")
    resumen = (f"### {len(clases)} fresas: {sanas} sanas y {danadas} dañadas\n"
               f"Verde es sana y morado es dañada. Tardó {time.time() - t0:.1f} segundos.")
    tabla = [[j, "dañada" if c == "danada" else "sana", round(float(x), 3), f"{100 * f:.0f} %"]
             for j, (c, x, f) in enumerate(zip(clases, p, moho), 1)]
    return salida, resumen, tabla


with gr.Blocks(title="Control de calidad de fresas") as demo:
    gr.Markdown("# Control de calidad de fresas\n"
                "Sube una foto con varias fresas. Grounding DINO encuentra cada fresa, SAM saca su contorno "
                "y un clasificador EfficientNet-B0 dice si está sana o dañada. Con GPU tarda unos segundos y sin GPU entre 30 y 60.")
    with gr.Row():
        with gr.Column():
            entrada = gr.Image(type="pil", label="Foto")
            umbral = gr.Slider(0.01, 0.9, value=cfg["umbral"], step=0.01,
                               label="Umbral para decir que está dañada (más bajo encuentra más dañadas pero se equivoca más con las sanas)")
            texto = gr.Textbox(value=TEXTO, label="Texto que se le da a Grounding DINO para buscar las fresas")
            boton = gr.Button("Analizar", variant="primary")
            gr.Examples([["ejemplos/ejemplo_1.jpeg"], ["ejemplos/ejemplo_2.jpeg"], ["ejemplos/ejemplo_3.jpeg"]], inputs=entrada)
        with gr.Column():
            resultado = gr.Image(label="Resultado")
            resumen = gr.Markdown()
            tabla = gr.Dataframe(headers=["fresa", "estado", "probabilidad de dañada", "parte blanca o gris"],
                                 label="Detalle por fresa (se marca dañada si la probabilidad pasa el umbral o si 10 % o más de la fresa se ve blanca o gris, como el moho)")
    boton.click(analizar, [entrada, umbral, texto], [resultado, resumen, tabla])

if __name__ == "__main__":
    # fuera de Hugging Face, share=True crea un link público temporal mientras el computador tenga la página corriendo
    demo.launch(share=not os.environ.get("SPACE_ID"))
