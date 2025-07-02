import os
try:
    from AIs import FED2013
except ImportError:
    from src.AIs import FED2013

DATASET_FOLDER : str = "./data/fed2013"
RESULTS : str = "./results"
FER2013_MODELPATH : str = "./persistent_data/fer2013_model.pth"
FD_DNN_PATH : str = "./persistent_data"

def download():
    FED2013.download_dataset(DATASET_FOLDER)

def detectAndClassify(download_dataset:bool=False, emotion_model:str=FER2013_MODELPATH, fd_dnn_path:str=FD_DNN_PATH, fed_dataset:str=DATASET_FOLDER, to_show:bool=False):
    if download_dataset : download()
    fed = FED2013(emotion_model, fd_dnn_path)

    for filename in os.listdir(DATASET_FOLDER):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            image_path = os.path.join(fed_dataset, filename)  # Sostituisci con il percorso della tua immagine
            fed.detect_and_classify(image_path, i_show=to_show)

if __name__=="__main__":
    download()
    detectAndClassify(to_show=True)