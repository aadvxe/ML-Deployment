from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import json
from keras_preprocessing.text import Tokenizer
import tensorflow as tf
from keras._tf_keras.keras.utils import pad_sequences
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.python.keras.models import load_model
import pickle

app = FastAPI()

loaded_model = load_model("text_classify.h5")

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pkl', 'rb') as handle:
    label_encoder = pickle.load(handle)

max_length = 100

class TaskRequest(BaseModel):
    tasks: List[str]
    workers: List[str]

class TaskResponse(BaseModel):
    task_labels: List[str]
    tasks: dict
    dependencies: dict

def predict_task_labels(model, tokenizer, label_encoder, tasks):
    sequences = tokenizer.texts_to_sequences(tasks)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    predictions = model.predict(padded_sequences)
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_labels = label_encoder.inverse_transform(predicted_classes)
    return predicted_labels

def get_task_durations(task):
    data = pd.read_csv('dataset5.csv')
    data_choice = data[data['Label_Task'] == task]
    if data_choice.empty:
        raise ValueError(f"Task '{task}' not found in the dataset.")
    duration = data_choice['Estimated_hours'].iloc[0]
    return duration

def assign_workers_to_tasks(task_labels, task_workers):
    worker_assignments = {}
    for idx, (task, worker) in enumerate(zip(task_labels, task_workers)):
        if worker not in worker_assignments:
            worker_assignments[worker] = []
        duration = get_task_durations(task)
        worker_assignments[worker].append((task, duration))
    return worker_assignments

common_dependencies = {
    "Deployment": ["Frontend Development", "Backend Development", "Desain UI/UX"],
    "Frontend Development": ["Desain UI/UX"],
    "Backend Development": ["Frontend Development"]
}

def apply_common_dependencies(predicted_labels):
    predicted_labels = [str(label) for label in predicted_labels]
    unique_labels = [f"{label}_{idx}" for idx, label in enumerate(predicted_labels)]
    label_to_unique = dict(zip(predicted_labels, unique_labels))
    dependencies = {}
    for task, deps in common_dependencies.items():
        if task in label_to_unique:
            unique_task = label_to_unique[task]
            unique_deps = [label_to_unique[dep] for dep in deps if dep in label_to_unique]
            dependencies[unique_task] = unique_deps
    return dependencies

@app.post("/predict_tasks", response_model=TaskResponse)
def predict_tasks(task_request: TaskRequest):
    try:
        predicted_labels = predict_task_labels(loaded_model, tokenizer, label_encoder, task_request.tasks)
        tasks = assign_workers_to_tasks(predicted_labels, task_request.workers)
        dependencies = apply_common_dependencies(predicted_labels)
        return TaskResponse(task_labels=predicted_labels.tolist(), tasks=tasks, dependencies=dependencies)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
