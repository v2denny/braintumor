'''
Train YOLO model with the annotated images from the Roboflow's dataset.
Saves the used dataset into ClearML.
Initiates a task in ClearML to track all the experiment results.
'''

from clearml import Task, Dataset
from ultralytics import YOLO
import os

def main():
    data_path = os.path.join(os.path.dirname(__file__), 'BTds', 'data.yaml')
    dataset_path = os.path.join(os.path.dirname(__file__), 'BTds')
    save_path = os.path.join(os.path.dirname(__file__), 'runs')
    model_variant = "yolo11s-seg"
    model = YOLO(f"{model_variant}.pt")

    dataset = None
    try:
        dataset = Dataset.get(dataset_name='BTds', dataset_project='BrainTumor')
        print("Dataset 'BTds' already exists. Using existing dataset.")
    except ValueError:
        print("Creating new dataset: 'BTds'")
        dataset = Dataset.create(dataset_name='BTds', dataset_project='BrainTumor')
        dataset.add_files(dataset_path)
        dataset.upload()
        dataset.finalize()
        print("Dataset uploaded and finalized.")

    task = Task.init(project_name="BrainTumor", task_name="yolov11_segmentation_hyper1")
    task.set_parameter("model_variant", model_variant)

    args = dict(
        project=save_path,
        data=data_path,
        epochs=100,
        imgsz=640,
        batch=0.9,
        device='0',
        plots=True,
        lr0=1e-4,
        momentum=0.99,
        weight_decay=1e-5,
        optimizer='Adam'
    )

    task.connect(args)
    model.train(**args)

if __name__ == "__main__":
    main()

