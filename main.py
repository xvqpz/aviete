import os
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO
import torch
#parsisiust^

device = '0' if torch.cuda.is_available() else 'cpu'
load_dotenv()

def downloadDataset(path, version):
    #jei neturit dataset folderio projekte atsius
    try:
        if not os.path.exists(path):
            dataset = version.download("yolov8")
            path_to_yaml = os.path.join(dataset.location, "data.yaml")
        else:
            print("nuh uh")
            path_to_yaml = os.path.join(path, "data.yaml")

        return path_to_yaml
    except:
        print("ivyko klaida pasisiunciant dataset :[")
        return None

#1 epoch = viena kart perskaito dataset. 50 epochs - perskaito 50 kartu
    #ok? ok.
def trainModel(model, path, epochs=1, batch=-1):
    if path is None:
        print("ner yaml failo :(")
        return
    try:
        model.train(
            data=path, 
            epochs=epochs, 
            batch=batch, 
            imgsz=640, 
            patience=10,  #sustos jei modelis per 10 epochu nepageres
            plots=True    #generuos grafikus
        )
    except Exception as e:
        print(f"nepavyko treniruoti modelio :O {e}")

def evaluateModel(model):
    try:
        print("modelio tikslumas:")
        metrics = model.val()
        print(metrics.box.map)
    except Exception as e:
        print(f"nepavyko ivertint modelio {e}")

def exportModel(model):
    try:
        model.export(format='tfjs')     # tensorFlow.js
        model.export(format='tflite')   # tensorFlow Lite
        model.export(format='saved_model') # standart
    except Exception as e:
        print(f"nepavyko exportuoti modelio {e}")

def main():
    #info roboflow datasetui su gerimu fotkemis, galit ji redaguot, bet bukit tvarkingi :)
    api_key = os.getenv("ROBOFLOW_API_KEY")
    rf = Roboflow(api_key)
    project = rf.workspace("ievas-workspace").project("beverage-e0yac-yayji")
    version = project.version(1)

    #kintamieji veliau pabaigt komentara
    dataset_path = "./beverage-1"
    model = YOLO('yolov8n.pt')

    yaml_path = downloadDataset(dataset_path, version)
    trainModel(model, yaml_path)
    evaluateModel(model)
    exportModel(model)


if __name__ == "__main__":
    main()