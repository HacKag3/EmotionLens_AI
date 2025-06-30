import os, shutil, random, glob
import pytest
from unittest.mock import patch
from src.fer2013 import eda, training_saving, evaluating, inferencing
from PIL import Image
import numpy as np

RESULTS : str = "./results"
DATASET_FOLDER : str = "./data/fer2013"
FER2013_MODELPATH : str = "./persistent_data/fer2013_model.pth"
FER2013_MODELPATH_SAVE : str = "./results/fer2013_model.pth"
IMAGE_PATH : str = './data/fer2013/inference'
INFERENCE_PATH : str = './results/inference'


def create_dummy_dataset(base_path=DATASET_FOLDER):
    # Crea cartelle train e test con almeno una classe
    train_path = os.path.join(base_path, "train", "happy")
    test_path = os.path.join(base_path, "test", "happy")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # Crea un'immagine dummy nera 48x48
    img = Image.fromarray(np.zeros((48,48), dtype=np.uint8))
    img.save(os.path.join(train_path, "img1.jpg"))
    img.save(os.path.join(test_path, "img1.jpg"))

@pytest.fixture(autouse=True)
def setup_and_cleanup():
    create_dummy_dataset()
    yield
    shutil.rmtree("./results", ignore_errors=True)
    shutil.rmtree("./data", ignore_errors=True)
    shutil.rmtree("./persistent_data", ignore_errors=True)

@patch("src.fer2013.download")
def test_eda(mock_download):
    mock_download.return_value = None
    eda(download_dataset=True)
    
    assert os.path.exists(os.path.join(RESULTS, "class_distribution.png")), "Class distribution plot missing"
    assert os.path.exists(os.path.join(RESULTS, "samples_images.png")), "Sample images plot missing"

@patch("src.fer2013.download")
def test_training_saving(mock_download):
    mock_download.return_value = None
    training_saving(download_dataset=True, epochs=1, save_path=FER2013_MODELPATH_SAVE, dataset_path=DATASET_FOLDER)
    
    assert os.path.exists(FER2013_MODELPATH_SAVE), "Model file not found after training."
    assert os.path.getsize(FER2013_MODELPATH_SAVE) > 0, "Model file is empty."

@patch("src.fer2013.download")
def test_evaluation(mock_download):
    mock_download.return_value = None
    training_saving(download_dataset=True, epochs=1, save_path=FER2013_MODELPATH_SAVE, dataset_path=DATASET_FOLDER)
    verbose = True
    evaluating(model_path=FER2013_MODELPATH_SAVE, dataset_path=DATASET_FOLDER, path_ouput=RESULTS, verbose=verbose)
    
    assert os.path.exists(os.path.join(RESULTS, "confusion_matrix.png")), "Confusion matrix plot missing after evaluation."

def get_random_images(root_path: str, num:int=1) -> str:
    # Cerca tutte le immagini dentro alle sottocartelle (es. classi)
    image_files = glob.glob(os.path.join(root_path, '**', '*.jpg'), recursive=True)
    if not image_files:
        raise FileNotFoundError(f"No image files found in {root_path}")
    os.makedirs(IMAGE_PATH, exist_ok=True)
    for _ in range(num):
        shutil.copy2(random.choice(image_files), IMAGE_PATH)

@patch("src.fer2013.download")
def test_inferencing(mock_download):
    mock_download.return_value = None
    training_saving(download_dataset=True, epochs=1, save_path=FER2013_MODELPATH_SAVE, dataset_path=DATASET_FOLDER)
    get_random_images(os.path.join(DATASET_FOLDER, "test"))
    inferencing(model_path=FER2013_MODELPATH_SAVE, image_path=IMAGE_PATH, path_ouput=INFERENCE_PATH)

    assert os.path.exists(INFERENCE_PATH), "Inference output folder missing."
    pred_classes = os.listdir(INFERENCE_PATH)
    assert len(pred_classes) > 0, "Nessuna classe predetta generata"
    assert any(os.listdir(os.path.join(INFERENCE_PATH, c)) for c in pred_classes), "Nessuna immagine salvata"