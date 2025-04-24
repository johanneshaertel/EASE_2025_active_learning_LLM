# Code for training the next model.
import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import datetime

from transformers import AutoTokenizer, TFRobertaForSequenceClassification

path = os.path.dirname(os.path.abspath(__file__)) + "/../data/final.json"

df = pd.read_json(path, lines=True)

# print(df.info())
#  #   Column            Non-Null Count  Dtype  
# ---  ------            --------------  -----  
#  0   url               1298 non-null   object 
#  1   tag_pre_qa        1298 non-null   object 
#  2   claude_yes        1217 non-null   float64
#  3   claude_no         1217 non-null   float64
#  4   claude_unclear    1217 non-null   float64
#  5   resoning_yes      383 non-null    object 
#  6   resoning_no       897 non-null    object 
#  7   resoning_unclear  539 non-null    object 
#  8   body              1298 non-null   object 
#  9   iteration         1298 non-null   int64  
#  10  repo              1298 non-null   object 
#  11  tag_post_qa       1298 non-null   object 

# We use the post_qa tag as the final tag.
df["tag"] = df["tag_post_qa"]

tag_focus_index = {
    "no": 0,
    "unclear": 1,
    "language": 2,
    "yes": 3,
}

# Filter to only include tag_focus_index keys
df = df[df['tag'].isin(tag_focus_index.keys())]

# Debug count.
print(df.groupby('tag').size())

# Extract ys and xy as token
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
xs = tokenizer(list(df["body"]), return_tensors="np", padding=True, truncation=True, max_length=64)
ys = np.array(df["tag"].map(tag_focus_index))

# Start time as string.
start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# (OPTIONS) --------------------------------------------------------
split = 1.0 # Do not split
learning_rate = 1e-6 * 0.5
persist_epoch = 40 # persisted on epoch
run_epochs = 45 # run for 45 epochs
model_path = os.path.dirname(os.path.abspath(__file__)) + "/../data/model.h5"
log_path = os.path.dirname(os.path.abspath(__file__)) + "/../data/fit.json"

# --------------------------------------------------------
train_idx = np.random.choice(len(ys), int(split * len(ys)), replace=False)
valid_idx = np.setdiff1d(np.arange(len(ys)), train_idx)

inputs_train = {k: v[train_idx] for k, v in xs.items()}
labels_train = ys[train_idx]

inputs_valid = {k: v[valid_idx] for k, v in xs.items()}
labels_valid = ys[valid_idx]

model = TFRobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=len(tag_focus_index))

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss=loss)

for epoch in range(run_epochs):
    history = model.fit(inputs_train, labels_train, epochs=1, batch_size=8)
   
    if persist_epoch == epoch:
        print("Persisting model")
        model.save_pretrained(model_path)

    train_loss = history.history['loss'][0]

    record = {}
    record["epoch"] = epoch
    record["train_loss"] = float(train_loss)
    record["start_time"] = start_time

    # If validation set is not empty.
    if len(valid_idx) > 0:
        predicted = model.predict(inputs_valid)["logits"]
        valid_loss = loss(labels_valid, predicted).numpy()
        record["loss"] = float(valid_loss)

        for current in range(len(tag_focus_index)):
            numb_current = np.sum(labels_valid == current)
            sort_current = np.argsort(predicted[:, current])
            labels_valid_current = labels_valid[sort_current]
            top_current = labels_valid_current[-numb_current:]
            name = list(tag_focus_index.keys())[list(tag_focus_index.values()).index(current)]
            record["top_" + str(name)] = float(np.mean(top_current == current))

    print(record)

    # append line to json file fit.json.
    with open(log_path, 'a') as f:
        f.write(json.dumps(record) + "\n")
