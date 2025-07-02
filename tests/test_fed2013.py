import pytest
import os, shutil
import numpy as np
from PIL import Image
from src.fed2013 import detectAndClassify

FER2013_MODELPATH : str = "./persistent_data/fer2013_model.pth"
FD_DNN_PATH : str = "./persistent_data"
DATASET_FOLDER : str = "./data/fed2013"

def create_dummy_model(model_path:str=FER2013_MODELPATH):
    from src.AIs import FER2013
    model = FER2013(model_path)
    model.save_state_dict(model_path)

def create_dummy_dataset(base_path=DATASET_FOLDER):
    os.makedirs(base_path, exist_ok=True)
    img = Image.fromarray(np.zeros((50,50), dtype=np.uint8))
    img.save(os.path.join(base_path, "img1.jpg"))

@pytest.fixture(autouse=True)
def setup_and_cleanup():
    create_dummy_model()
    create_dummy_dataset()
    yield
    shutil.rmtree("./results/", ignore_errors=True)
    shutil.rmtree("./data/", ignore_errors=True)
    shutil.rmtree("./persistent_data/", ignore_errors=True)

def test_detectAndClassify():
    detectAndClassify(emotion_model=FER2013_MODELPATH, fd_dnn_path=FD_DNN_PATH, fed_dataset=DATASET_FOLDER)