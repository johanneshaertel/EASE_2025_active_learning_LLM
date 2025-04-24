# This code we use to infer the probability bounds of reviews beeing truely related to security defects.

# We can only do this on the random data. We only onserverd negatives on such sample, see lables_random. 
# According to the following model, we can be sure to 95% that the pobability
# of a reviews beeing related to a security defect is less than 2,88%.

import numpy as np
import matplotlib.pyplot as plt
import stan # TODO: Note: Add pystan to requirements.txt if you like to run this code.

# Our Real data: Repeat 100 zeros. This is what was observed when labeling 100 random reviews (see data/random.csv).
y = np.repeat(0, 100)

# Simulated from binomial when testing the model.
# y = np.random.binomial(1, 0.1, 100)

model_code = """
data {
    int<lower=0> N;
    array[N] int<lower=0, upper=1> y;
}

parameters {
    real<lower=0,upper=1> prob;
} 

model {
   // Prior for prob which is neutral.
   prob ~ beta(1, 1); 

   // Likelihood.
   y ~ binomial(1, prob);
}
"""

# Build and extact samples for paramter prob.
model = stan.build(model_code, data={"N": len(y), "y": y})
fit = model.sample(num_chains=1, num_samples=20000)
prop_samples = fit['prob'][0]

# Analyze the samples.
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

ax.hist(prop_samples, bins=100, density=True, alpha=0.6, color='g')

quantile_95 = np.quantile(prop_samples, 0.95)

print("95% quantile:", quantile_95)

# Plot quantile as vertical line.
ax.axvline(quantile_95, color='r', linestyle='dashed', linewidth=1)

# count to double-check
print(sum(prop_samples > quantile_95) / len(prop_samples))

plt.show()
