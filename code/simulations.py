import tensorflow as tf
import numpy as np
import keras as keras
import json
import boto3
import uuid

# Limit GPU memory to run this more than once.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=256 * 8)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)

# Simulation.
vars = 20

# Number of observations.
n = 1000000

# Balance.
ballance = 0.99

logit_stdev = np.random.choice([0.7, 1.0, 1.3])

# Correlation structure.
correlation = np.random.choice([True, False])

# Epochs.
epochs = np.random.choice([5, 10, 30])

# Increment and iterations.
options = [(200, 20), (40, 100)]

increment, iterations = options[np.random.choice(range(len(options)))]

# Random network structure (LOL).
st = [vars, 15, 15, 15, 1]
ws = [tf.random.normal(shape=(st[i-1], st[i]), mean=0, stddev=np.sqrt(2 / st[i-1])) for i in range(1, len(st))]
bs = [tf.random.normal(shape=(st[i],        ), mean=0, stddev=1                   ) for i in range(1, len(st))]

bs[-1] = tf.constant(0.0, shape=(1,))

def forward(input):
    current = input
    for i in range(len(ws)):
        current = tf.matmul(current, ws[i]) + bs[i]
        if i < len(ws) - 1:
            current = tf.nn.relu(current)

    return current

# Produce n observations with or without a random correlation structure.
xs = tf.random.normal(shape=(n, vars))

# forward correlation structure (optional).
if correlation:
    sigma = np.random.normal(0, 1, (vars, vars))
    sigma = np.dot(sigma, sigma.T)
    xs = tf.matmul(xs, tf.constant(sigma, dtype=tf.float32))

# Scale the last ws and bs and correct balance.
std = np.std(forward(xs).numpy())
ws[-1] = ws[-1] / std
ws[-1] = ws[-1] * logit_stdev
mean_adj = np.mean(forward(xs).numpy()) - np.log((1 - ballance) / ballance)
bs[-1] = tf.constant(float(-mean_adj), shape=(1,))

probability = tf.sigmoid(forward(xs))

ys = tf.random.stateless_binomial(shape=(n, 1), seed=[1, 2], counts=1, probs=probability)[:,0]

# We need it as numpy.
xs = xs.numpy()
ys = ys.numpy()

# Count.
n_pos = sum(ys == 1)
n_neg = sum(ys == 0)

print("n_pos: " + str(n_pos))
print("n_neg: " + str(n_neg))

# Select random samples to start process with.
initial_observations = np.random.choice(n, 200)

# Method to be tested in randomize order.
# "low"  = "majority"
# "high" = "rare class"
methods = ["random", "entropy", "high", "low"]
np.random.shuffle(methods)

# use simpe random uuid to identify the simulation.
id = str(uuid.uuid4())

# turns into tf

xs_tf = tf.constant(xs, dtype=tf.float32)
ys_tf = tf.constant(ys, dtype=tf.float32)

# Iterate over methods.
for sampling in methods:

    observations = initial_observations

    for iteration in range(0, iterations):
        print("iteration " + str(iteration) + " with " + str(len(observations)) + " (" + sampling + ")")
        
        #Print postive fraction in obversations and in population.
        # print("Positive fraction in observations: " + str(sum(ys[observations] == 1) / len(observations)))
        # print("Positive fraction in population: " + str(sum(ys == 1) / n))
        
        record = dict()
        record["id"] = id
        record["vars"] = vars
        record["n"] = n
        record["n_pos"] = n_pos
        record["n_neg"] = n_neg
        record["ballance"] = ballance
        record["sampling"] = sampling
        record["increment"] = increment
        record["iteration"] = iteration
        record["correlation"] = correlation
        record["logit_stdev"] = logit_stdev
        record["epochs"] = epochs
        
        model = keras.Sequential([
            keras.layers.Dense(15, activation='relu'),
            keras.layers.Dense(15, activation='relu'),
            keras.layers.Dense(15, activation='relu'),
            keras.layers.Dense(2, activation="linear"),
        ])

        # Fit the model.
        opitimizer = keras.optimizers.Adam(learning_rate=0.01)
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer=opitimizer, loss=loss)
        loss_hist = model.fit(xs[observations], ys[observations], batch_size = 128, epochs = epochs, verbose=0)

        # IMPORTANT: This is the prediction of labels in R.
        predict_before_soft = model.predict(xs_tf, batch_size = 1024, verbose = 0)
        ys_pred = tf.nn.softmax(predict_before_soft).numpy()
        
        full_dataset_loss = loss(ys_tf, predict_before_soft).numpy()

        record["loss_full_dataset"] = float(full_dataset_loss)

        # Record some statistics on these observations.
        n_obs = len(observations)
        record["n_obs"] = n_obs
        record["n_pos_obs"] = sum(ys[observations] == 1)
        record["n_neg_obs"] = sum(ys[observations] == 0)

        # Print first and last loss.
        record["loss_fit_fst"] = loss_hist.history['loss'][0]
        record["loss_fit_lst"] = loss_hist.history['loss'][-1]
      
        already_observed = set(observations)

        if sampling == "random":
            # Random sample observations that are not already in observations.
            new_observations = np.random.choice([x for x in range(n) if x not in already_observed], increment)

            observations = np.concatenate((observations, new_observations))
        
        if sampling == "entropy":
            # Compute entropy of each prediction and add those with the highest entropy to the observations.
            ys_pred_clip = np.clip(ys_pred, 1e-7, 1 - 1e-7)
            entropy = -np.sum(ys_pred_clip * np.log(ys_pred_clip), axis=1)
        
            priority = [x for x in np.argsort(entropy) if x not in already_observed] 
            new_observations = priority[-increment:]

            observations = np.concatenate((observations, new_observations))

        if sampling == "high":
            # Compute those with the highest (rare-label) probability of being positive and add those to the observations.
            priority = [x for x in np.argsort(ys_pred[:,1]) if x not in already_observed]
            new_observations = priority[-increment:]

            observations = np.concatenate((observations, new_observations))

        if sampling == "low":
            # Compute those with the lowest (majority-label) probability of being positive and add those to the observations.
            priority = [x for x in np.argsort(ys_pred[:,1]) if x not in already_observed]
            new_observations = priority[:increment]

            observations = np.concatenate((observations, new_observations))

        # Map record values to jsons serializable types.
        record = {k: str(v) for k, v in record.items()}

        with open(f"sim_{id}.json", 'a') as f:
            f.write(json.dumps(record) + "\n")


# s3://vua-data/simulation2/
aws_s3_bucket = "vua-data"
aws_s3_path = "simulation2/"

# Use boto to upload the file to S3.
s3 = boto3.client('s3')
s3.upload_file(f"sim_{id}.json", aws_s3_bucket, aws_s3_path + f"sim_{id}.json")
