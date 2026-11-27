"""Publica la página de Gradio en Modal (modal.com) para que funcione con el computador apagado.
Uso, desde la carpeta frutas:  .venv\\Scripts\\modal deploy space\\modal_app.py
Modal apaga el contenedor cuando nadie lo usa, así que la primera foto después de un rato tarda más."""
from pathlib import Path

import modal

CARPETA = Path(__file__).parent
MODELOS = ("from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor; "
           "AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-tiny'); "
           "AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-tiny'); "
           "SamProcessor.from_pretrained('facebook/sam-vit-base'); SamModel.from_pretrained('facebook/sam-vit-base')")

imagen = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "torchvision", "transformers==5.23.0", "gradio==6.32.0", "fastapi[standard]", "numpy", "pillow")
    .run_commands(f'python -c "{MODELOS}"')  # los modelos quedan guardados dentro de la imagen
    .add_local_dir(CARPETA, "/pagina", ignore=["modal_app.py", "__pycache__"])
)

app = modal.App("control-calidad-fresas", image=imagen)


# shortcut: corre en CPU porque la GPU de Modal pide registrar una tarjeta, con GPU cada foto bajaría a pocos segundos
@app.function(cpu=8.0, memory=8192, max_containers=1, scaledown_window=300, timeout=900)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def pagina():
    import importlib.util
    import sys

    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app

    sys.path.insert(0, "/pagina")
    spec = importlib.util.spec_from_file_location("pagina_fresas", "/pagina/app.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return mount_gradio_app(app=FastAPI(), blocks=modulo.demo, path="/")
