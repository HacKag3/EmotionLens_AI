'''
Codice Principale del Pacchetto
'''
# pylint: disable=no-member
import os
import cv2
import gradio as gr
import numpy as np
import threading
import time
try:
    from AIs import FED2013
    import fer2013
except ImportError:
    from src.AIs import FED2013
    from src import fer2013

FER2013_MODELPATH : str = "./persistent_data/fer2013_model.pth"
FD_DNN_PATH : str = "./persistent_data"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30/1000
CSS = """
body { background-color: #121212; color: #eee; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
.gradio-container { max-width: 720px; margin: auto; padding: 20px; border-radius: 15px; background-color: #1e1e1e; box-shadow: 0 0 20px #222; }
h1, h2 { color: #FFA500; text-align: center; }
.gr-image { border-radius: 10px; box-shadow: 0 0 15px #FFA500; }
"""

def checkForModel(model_pah:str, epochs:int=5):
    if not os.path.exists(model_pah):
        fer2013.training_saving(epochs=epochs, save_path=model_pah)

def videoCapture_loop(fed: FED2013, frame_container: dict) -> None:
    """
    Acquisisce i frame dalla webcam, li elabora con FED2013 per il riconoscimento emozioni
    e aggiorna frame_container['frame'] con il frame processato.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Errore: Impossibile aprire la webcam.")
        return

    while True:
        ret, img = cap.read()
        if not ret:
            print("Frame non acquisito correttamente, riapro webcam...")
            cap.release()
            time.sleep(0.5)
            cap = cv2.VideoCapture(0)
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        processed_frame = fed.detect_and_classify(img_rgb, is_img_a_path=False)
        
        if processed_frame is not None:
            frame_container['frame'] = processed_frame
        else:
            frame_container['frame'] = img_rgb
        
        time.sleep(FPS)

def main():
    checkForModel(FER2013_MODELPATH)
    fed = FED2013(emotion_model=FER2013_MODELPATH, path=FD_DNN_PATH)
    # Contenitore condiviso per il frame attuale (per evitare problemi con variabili globali)
    frame_container = {'frame': np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)}
    # Avvio del thread per la cattura video e elaborazione
    capture_thread = threading.Thread(target=videoCapture_loop, args=(fed, frame_container), daemon=True)
    capture_thread.start()
    # Interfaccia Gradio con Blocks e aggiornamento automatico
    with gr.Blocks(css=CSS) as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("# Emotion Recognition 🎭")
                gr.Markdown("Webcam live stream con riconoscimento emozioni in tempo reale.")
        with gr.Row():
            img = gr.Image(label="Webcam Live", height=FRAME_HEIGHT, width=FRAME_WIDTH)
        
        timer = gr.Timer(FPS)  # 30 ms circa
        timer.tick(fn=lambda: frame_container["frame"], inputs=None, outputs=img)

    demo.launch()

if __name__=="__main__":
    main()
