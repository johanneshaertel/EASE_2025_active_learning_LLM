import os
import zipfile
import json

from transformers import AutoTokenizer, TFRobertaForSequenceClassification
import tensorflow as tf

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

tag_focus_index = {
    "no": 0,
    "unclear": 1,
    "language": 2,
    "yes": 3,
}

model = TFRobertaForSequenceClassification.from_pretrained(os.path.dirname(__file__) + "/../data/model.h5", 
                                                           num_labels=len(tag_focus_index))

def llm_classifications(jsons):
    texts = [json["body"] for json in jsons]
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="np")

    outputs = model(inputs)["logits"]
    outs = tf.nn.softmax(outputs).numpy()

    for i in range(len(jsons)):
        for key, value in tag_focus_index.items():
            jsons[i][key] = float(outs[i, value])

    
# This is an incredibly big file and not hosted on github.
zip_name = os.path.dirname(__file__) + "/../data/reviews.zip"

total_reviews_processed = 0

buffer = []

with zipfile.ZipFile(zip_name, 'r') as z:
        pullfiles = [name for name in z.namelist() if name.endswith(".json")]	

        print("Found " + str(len(pullfiles)) + " files.")

        for name in pullfiles:
            print("Processing file: " + name + " (" + str(total_reviews_processed) + ")")
            with z.open(name) as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        buffer.append(obj)
                        total_reviews_processed += 1

                        # TODO: Possible to filter for those with java path only.
                        if not obj["path"].endswith("java"):
                            continue

                        if len(buffer) > 64: # TODO: Can be increased if you have a good gpu.
                            llm_classifications(buffer)
                            # TODO: Add your specific candidate selection logic here. This is a dummy.
                            for obj in buffer:
                                if obj["yes"] > 0.8:
                                    print(obj["url"])
                                    print("Processed until now: " + str(total_reviews_processed))

                            buffer = []

                    except json.JSONDecodeError as e:
                        print("Error decoding JSON: ", e)
                        print("Line: ", line)


