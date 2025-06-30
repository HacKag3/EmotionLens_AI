import os
try:
    from AIs import FER2013
except ImportError:
    from src.AIs import FER2013

RESULTS : str = "./results"
DATASET_FOLDER : str = "./data/fer2013"
FER2013_MODELPATH : str = "./persistent_data/fer2013_model.pth"
FER2013_MODELPATH_SAVE : str = "./results/fer2013_model.pth"
IMAGE_PATH : str = './data/fer2013/inference'
INFERENCE_PATH : str = './results/inference'

def download():
    FER2013.download_dataset(DATASET_FOLDER)

def eda(download_dataset:bool=False):
    if download_dataset : download()
    fer = FER2013()
    fer.EDA("./data/fer2013")

def training_saving(download_dataset:bool=False, epochs:int = 2, save_path:str=FER2013_MODELPATH_SAVE, dataset_path:str=DATASET_FOLDER):
    if download_dataset : download()
    fer = FER2013()
    fer.train(path_data=dataset_path, epochs=epochs)
    fer.save_state_dict(path=save_path)
    return fer

def evaluating(download_dataset:bool=False, model_path:str=FER2013_MODELPATH, dataset_path:str=DATASET_FOLDER, path_ouput:str=RESULTS, verbose:bool=True):
    if download_dataset : download()
    if not os.path.exists(model_path):
        fer = training_saving(epochs=5)
    else:
        fer = FER2013(model_path)
    fer.evaluate(path_data=dataset_path, path_output=path_ouput, verbose=verbose)

def inferencing(image_path: str, download_dataset:bool=False, model_path:str=FER2013_MODELPATH, path_ouput: str=INFERENCE_PATH):
    if download_dataset : download()
    if not os.path.exists(model_path):
        raise FileNotFoundError("Necessario un modello pre-addestrato esistente")
    fer = FER2013(model_path)
    for filename in os.listdir(image_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            fer.inference_singleImg(path_img=os.path.join(image_path, filename), path_output=path_ouput)

if __name__ == "__main__":
    download()
    eda()
    evaluating()
    inferencing(IMAGE_PATH, model_path=FER2013_MODELPATH_SAVE)