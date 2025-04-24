# This is the version of Claude used to the quality assurance stage.

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import boto3
import matplotlib.pyplot as plt
import time
import multiprocessing
import json
from botocore.exceptions import ClientError

# -----------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------

input_file = os.path.dirname(__file__) + "/../data/preqa.json"
output_file = os.path.dirname(__file__) + "/../data/claude_output.json"

urls_done = []

# read line delimited output and check which urls are done.
if os.path.exists(output_file):
    for line in open(output_file, "r"):
        obj = json.loads(line)
        urls_done.append(obj["url"])

# read line delimited json file.
df = pd.read_json(input_file, lines=True)

# remove those broken, bot, or language.
df = df[~df['tag'].isin(["broken", "bot", "language"])]

# print remaining original_tag.
print(df['tag'].value_counts())

# randomize the order of the rows.
df = df.sample(frac=1).reset_index(drop=True)

# Print what is left todo.
print("Done " + str(len(urls_done)) + " urls.")
print("Stil todo " + str(len(df) - len(urls_done)) + " urls.")

# -----------------------------------------------------------
# Prompting
# -----------------------------------------------------------

system = '''
Can you give the very short reasoning why the review falls into one of the following categories?
Conclude your reasoning with one of the following categories:
<categories>

    The review discusses a potential security defect of code or other artifacts.
    <category> potential security defect </category>

    The review does not discuss a security defect of code or other artifacts.
    <category> no security defect </category>

    The relation to a security defect is unclear.
    <category> unclear </category>

</categories>

Rules for your argumentation:
1. Use bullet points for explaining your reasoning.
2. Keep your reasoning short and concise.
3. Merge bullet points when possible.
4. Directly cite parts of the text if meaningful.
5. Conclude on a category.
    '''.strip()

categories = ["potential security defect", "no security defect", "unclear"]

initial_reasoning = "No reasoning yet so we have <category> undefined </category>."

assistant = "After thinking, the updated reasoning is: <reasoning>"

def user(body, reasoning):
    return '''
<code_review>
    $body$
</code_review>
<reasoning>
    $reasoning$
</reasoning>'''.replace("$body$", body).replace("$reasoning$", reasoning).strip()

anthropic_version = "bedrock-2023-05-31"
bedrock_runtime = boto3.client(service_name='bedrock-runtime')
model_id = 'anthropic.claude-3-5-sonnet-20240620-v1:0'

def claude(body):
    response = bedrock_runtime.invoke_model(body=body, modelId=model_id)
    response_body = json.loads(response.get('body').read())

    return response_body

def myfunc(user):
    while True:
        try:
            print("Running claude")
            return claude(json.dumps({
                "anthropic_version": anthropic_version,
                "max_tokens": 2000,
                "temperature": 1.0,  # High temperature increases randomness in responses
                "stop_sequences": ["</reasoning>"],  # Low stop_sequences ensures proper termination
                "system": system,  # High complexity in the system prompt can lead to nuanced reasoning
                "messages": [
                        {"role": "user", "content": user},  # Low ambiguity in user input ensures clarity
                        {"role": "assistant", "content": assistant}  # High relevance in assistant content is expected
                    ]
            }))["content"][0]["text"].strip()
        

        except ClientError as e:
            if e.response['Error']['Code'] == 'ThrottlingException' or e.response['Error']['Code'] == 'ServiceUnavailableException':
                print("Slowing it down: ", e)
                time.sleep(10)
            else:
                print("ClientError: ", e)
                raise e
            


def extract_category(text):
    category_idx1 = text.find("<category>")
    category_idx2 = text.find("</category>")
    if category_idx1 != -1 and category_idx2 != -1:
        category = text[category_idx1 + len("<category>"):category_idx2].strip()
        if category not in categories:
            category = "undefined"
    else:
        category = "undefined"
    return category

# I hate myself for this.
pool = multiprocessing.Pool()

for idx, row in df.iterrows():

    body = row["body"]
    url = row["url"]
    iteration = int(row["iteration"])
    tag = row["tag"]
    repo = row["repo"]

    if url in urls_done:
        print("Skipping", url)
        continue

    scaling_par = 4
    scaling_seq = 4

    total_categories = []
    last_reasoning_category = dict()

    # Parallel scaling.
    reasonings = [initial_reasoning for _ in range(0, scaling_par)]

    # Sequential scaling.
    for si in range(0, scaling_seq):
        print("Seq: ", str(si) + " parallel " + str(len(reasonings)) + " url: " + url)
        # Run the model in parallel.
        next_reasonings = [myfunc(user(body, reasoning)) for reasoning in reasonings]
         
        # Update the categories.
        for nr in next_reasonings:
            category = extract_category(nr)
            total_categories.append(category)
            last_reasoning_category[category] = f"Reasoning from iteration {si} \n" + nr


    # Output data.
    data = {} 
    data["tag"] = tag

    # Count the categories and divide by the number. 
    print("Old tag: ", tag)
    for category in categories:
        count =  total_categories.count(category)
        print(f"Category {category}: {count}")
        data["cls_" + category.replace(" ", "_")] = count
        data["res_" + category.replace(" ", "_")] = last_reasoning_category.get(category, "")

    data["body"] = body
    data["url"] = url
    data["iteration"] = iteration
    data["repo"] = repo
    data["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    
    with open(output_file, "a") as f:
        f.write(json.dumps(data) + "\n")

