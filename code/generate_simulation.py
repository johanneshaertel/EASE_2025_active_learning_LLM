# Read and create plots for the simulation.
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

from sklearn.neighbors import KernelDensity

plt.rcParams['font.size'] = '14'

import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__)) + "/"

# Make paper generated dir if not exists.

os.makedirs(script_dir  + "../paper/generated" , exist_ok=True)

# Show all pandas columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

files = []

for file in os.listdir(script_dir + "../data/simulation/"):
    # Read line delimited json.
    files.append(pd.read_json(script_dir + "../data/simulation/" + file, lines=True))

df = pd.concat(files, ignore_index=True)

df["n_pos_frac_obs"] = df["n_pos_obs"] / df["n_obs"]
df["n_pos_frac"] = df["n_pos"] / df["n"]

# Cast correlation column as string "False"/"True" (pandas serialization issue).
df["correlation"] = df["correlation"].astype(str)

# Correlation vs non correlation, methods and different, sampling steps.
colors = ["red", "blue", "green", "black"]
markers = ["o", "x", "s", "^"]
line_styles = ["-", "--", "-.", ":"]

gray_colors = ["#999999", "#AAAAAA", "#BBBBBB", "#CCCCCC"]

default_figsize = (7, 5)

xlim = (100, 4570)

method_alias = {
    "random" : "Random",
    "entropy" : "Entropy",
    "low" : "Majority-Label",
    "high" : "Rare-Label"
}


################################################################################################

fig, ax = plt.subplots(1, 1, figsize=default_figsize,  dpi=100)

correlation = "False"
increment = 200

for i, sampling in enumerate(["random", "entropy", "low", "high"]):
    focus_df = df
    focus_df = focus_df[focus_df["increment"] == increment]
    focus_df = focus_df[focus_df['logit_stdev'] == 1.3]
    focus_df = focus_df[focus_df['sampling'] == sampling]
    metric = "loss_full_dataset"

    mean = focus_df.groupby('n_obs')[metric].mean()
    count = focus_df.groupby('n_obs')[metric].count()

    se = focus_df.groupby('n_obs')[metric].std() / (count ** 0.5)
    ci = 1.96 * se

    ax.plot(mean.index, mean, color=colors[i], marker=markers[i], label=method_alias[sampling], markevery=int(400/increment))
    ax.fill_between(mean.index, mean - ci, mean + ci, color=colors[i], alpha=0.2, linestyle=line_styles[i])

    # Show zero line.
    ax.axhline(y=0.0, color='black', linestyle=':')
    
    # Add the last number as percentage to the very right of the figure at each line.
    #ax.text(mean.index[-1] + 20, mean.iloc[-1], "{:.1%}".format(mean.iloc[-1]), color=colors[i], fontsize=11)

# Show in lower left corner in light gray the number of simulation runs.
ax.text(0.80, 0.07, f"{int(count.mean())} sim. runs", transform=ax.transAxes, fontsize=9, color='gray')

#ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))

ax.set_xlabel("Labeled Observations")
ax.set_ylabel("Loss Full Dataset")
ax.set_xlim(xlim)

# Legend
ax.legend(loc='lower left', title="Candidate Selection by", frameon=True, facecolor='lightgray')

fig.tight_layout()

plt.savefig(script_dir + "../paper/generated/loss.pdf")

################################################################################################

fig, ax = plt.subplots(1, 1, figsize=default_figsize,  dpi=100)

#ax.set_title("Candidate Selection")

correlation = "False"
increment = 200

for i, sampling in enumerate(["random", "entropy", "low", "high"]):
    focus_df = df
    focus_df = focus_df[focus_df["increment"] == increment]
    focus_df = focus_df[focus_df['correlation'] == correlation]
    focus_df = focus_df[focus_df['sampling'] == sampling]
    metric = "n_pos_frac_obs"


    mean = focus_df.groupby('n_obs')[metric].mean()
    count = focus_df.groupby('n_obs')[metric].count()

    se = focus_df.groupby('n_obs')[metric].std() / (count ** 0.5)
    ci = 1.96 * se

    ax.plot(mean.index, mean, color=colors[i], marker=markers[i], label=method_alias[sampling], markevery=int(400/increment))
    ax.fill_between(mean.index, mean - ci, mean + ci, color=colors[i], alpha=0.2, linestyle=line_styles[i])

    # Show zero line.
    ax.axhline(y=0.0, color='black', linestyle=':')
    
    # Add the last number as percentage to the very right of the figure at each line.
    ax.text(mean.index[-1] + 20, mean.iloc[-1], "{:.1%}".format(mean.iloc[-1]), color=colors[i], fontsize=11)

# Show in lower left corner in light gray the number of simulation runs.
ax.text(0.80, 0.07, f"{int(count.mean())} sim. runs", transform=ax.transAxes, fontsize=9, color='gray')

ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))

ax.set_xlabel("Labeled Observations")
ax.set_ylabel("Identification Rate")
ax.set_xlim(xlim)

# Legend
ax.legend(loc='upper left', title="Candidate Selection by", frameon=True, facecolor='lightgray')

fig.tight_layout()

plt.savefig(script_dir + "../paper/generated/candidate_selection.pdf")

################################################################################################
################################################################################################
#####################Step size ###########################################
################################################################################################

fig, ax = plt.subplots(1, 1, figsize=default_figsize,  dpi=100)

# ax.set_title("Iteration Increment")
sampling = "high"
correlation = "False"

for i, increment in enumerate([40, 200]):
    focus_df = df
    focus_df = focus_df[focus_df["increment"] == increment]
    focus_df = focus_df[focus_df['correlation'] == correlation]
    focus_df = focus_df[focus_df['sampling'] == sampling]
    metric = "n_pos_frac_obs"

    mean = focus_df.groupby('n_obs')[metric].mean()
    count = focus_df.groupby('n_obs')[metric].count()

    se = focus_df.groupby('n_obs')[metric].std() / (count ** 0.5)
    ci = 1.96 * se

    ax.plot(mean.index, mean, color=colors[i], marker=markers[i], label= f"{increment} candidates", markevery=int(400/increment))
    ax.fill_between(mean.index, mean - ci, mean + ci, color=colors[i], alpha=0.2, linestyle=line_styles[i])

    # Show zero line.
    ax.axhline(y=0.0, color='black', linestyle=':')

    # Add the last number as percentage to the very right of the figure at each line.
    ax.text(mean.index[-1] + 20, mean.iloc[-1], "{:.1%}".format(mean.iloc[-1]), color=colors[i], fontsize=11)


# Show in lower left corner in light gray the number of simulation runs.
ax.text(0.80, 0.07, f"{int(count.mean())} sim. runs", transform=ax.transAxes, fontsize=9, color='gray')
ax.set_xlim(xlim)

ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))

ax.set_xlabel("Labeled Observations")
ax.set_ylabel("Identification Rate")
 
# Legend
ax.legend(loc='upper left', title="Iteration increments by", frameon=True, facecolor='lightgray')

fig.tight_layout()

plt.savefig(script_dir + "../paper/generated/step_size.pdf")

################################################################################################
##################### Correlation ###########################################
################################################################################################

fig, ax = plt.subplots(1, 1, figsize=default_figsize,  dpi=100)

# ax.set_title("Correlation")
sampling = "high"
correlation = "False"
increment = 40

for i, correlation in enumerate(["True", "False"]):
    focus_df = df
    focus_df = focus_df[focus_df["increment"] == increment]
    focus_df = focus_df[focus_df['correlation'] == correlation]
    focus_df = focus_df[focus_df['sampling'] == sampling]
    metric = "n_pos_frac_obs"

    mean = focus_df.groupby('n_obs')[metric].mean()
    count = focus_df.groupby('n_obs')[metric].count()

    se = focus_df.groupby('n_obs')[metric].std() / (count ** 0.5)
    ci = 1.96 * se

    ax.plot(mean.index, mean, color=colors[i], marker=markers[i], label= f"{correlation}", markevery=int(400/increment))
    ax.fill_between(mean.index, mean - ci, mean + ci, color=colors[i], alpha=0.2, linestyle=line_styles[i])

    # Show zero line.
    ax.axhline(y=0.0, color='black', linestyle=':')

    # Add the last number as percentage to the very right of the figure at each line.
    ax.text(mean.index[-1] + 20, mean.iloc[-1], "{:.1%}".format(mean.iloc[-1]), color=colors[i], fontsize=11)

# Show in lower left corner in light gray the number of simulation runs.
ax.text(0.80, 0.07, f"{int(count.mean())} sim. runs", transform=ax.transAxes, fontsize=9, color='gray')
ax.set_xlim(xlim)

ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))

ax.set_xlabel("Labeled Observations")
ax.set_ylabel("Identification Rate")
 
# Legend
ax.legend(loc='upper left', title="Correlation in Input", frameon=True, facecolor='lightgray')

fig.tight_layout()

plt.savefig(script_dir + "../paper/generated/correlation.pdf")

################################################################################################
################################### Impact of predictability.  #################################
################################################################################################

fig, axs = plt.subplots(1, 1, figsize=default_figsize, dpi=100,)

#axs.set_title("Predictability")
sampling = "high"
ax = axs
i = -1
alias_logit_stdev = {
    0.7 : "0.7 (Low)",
    1.0 : "1.0 (Normal)",
    1.3 : "1.3 (High)"
}
# distinct logit_stdev
for logit_stdev in [0.7, 1.0, 1.3]:    
    i += 1
    focus_df = df
    focus_df = focus_df[focus_df['increment'] == 40]
    focus_df = focus_df[focus_df['sampling'] == sampling]
    focus_df = focus_df[focus_df['logit_stdev'] == logit_stdev]

    # for debug reasons, also print n_pos_frac
    print(" for logit_stdev: " + str(logit_stdev))
    print(focus_df["n_pos_frac"].mean())
    
    metric = "n_pos_frac_obs"

    mean = focus_df.groupby('n_obs')[metric].mean()
    count = focus_df.groupby('n_obs')[metric].count()

    se = focus_df.groupby('n_obs')[metric].std() / (count ** 0.5)
    ci = 1.96 * se

    ax.plot(mean.index, mean, color=colors[i],  label= alias_logit_stdev[logit_stdev], marker=markers[i], markevery=int(10))
    ax.fill_between(mean.index, mean - ci, mean + ci, color=colors[i], alpha=0.2, linestyle=line_styles[i])

    # Add the last number as percentage to the very right of the figure at each line.
    ax.text(mean.index[-1] + 20, mean.iloc[-1], "{:.1%}".format(mean.iloc[-1]), color=colors[i], fontsize=11)

# Show in lower left corner in light gray the number of simulation runs.
ax.text(0.80, 0.07, f"{int(count.mean())} sim. runs", transform=ax.transAxes, fontsize=9, color='gray')
ax.set_xlim(xlim)

# Show zero line.
ax.axhline(y=0.0, color='black', linestyle=':')

# precentage
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
ax.legend(loc='upper left', title="Label Predictability", frameon=True, facecolor='lightgray')
ax.set_ylabel("Identification Rate")
ax.set_xlabel("Labeled Observations")

fig.tight_layout()

plt.savefig(script_dir + "../paper/generated/predictability.pdf")

################################################################################################
################################### Imbalance in the Population (n_pos_frac).  #################
################################################################################################
################################################################################################

fig, axs = plt.subplots(1, 1, figsize=(6, 4), dpi=100,)

axs.set_title("Imbalance in the Population")

focus_df = df

X = focus_df["n_pos_frac"]
X = 1 / (1 + np.exp(-X))
X = X.values.reshape(-1, 1)

kde = KernelDensity(kernel='gaussian', bandwidth=0.001).fit(X)

# Plot KDE
X_plot = np.linspace(0, 0.1, 300)[:, np.newaxis]
X_plot_log = 1 / (1 + np.exp(-X_plot))

dens = kde.score_samples(X_plot_log)

axs.plot(X_plot, np.exp(dens), color="black")

# plot the mean as a vertical line.
axs.axvline(x=(focus_df["n_pos_frac"]).mean() , color="red", linestyle="--")

mean_value = (focus_df["n_pos_frac"]).mean()
axs.text(mean_value + 0.001, 0.2, f"Mean ({mean_value:.1%})", color="red")

# Format x as percentage.
axs.xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
axs.set_xlabel("Rare-Label")
axs.set_ylabel("Density")
axs.yaxis.set_ticks([])
fig.tight_layout()


plt.savefig(script_dir + "../paper/generated/imbalance.pdf")

################################################################################################


fig, ax = plt.subplots(1, 1, figsize=default_figsize, dpi=100)

sampling = "high"

# ax.set_title("Overfitting")

i = 0
for epoch in [5, 10, 30]:
    i += 1
    focus_df = df
    focus_df = focus_df[focus_df['correlation'] == "False"]
    focus_df = focus_df[focus_df['increment'] == 40]
    focus_df = focus_df[focus_df['sampling'] == sampling]
    focus_df = focus_df[focus_df['epochs'] == epoch]

    metric = "n_pos_frac_obs"
    # conpute the 0.05 and 0.95 quantile for each n_obs.
    # quantiles = focus_df.groupby('n_obs')["n_pos_frac_obs"].quantile([0.45, 0.55]).unstack()
    mean = focus_df.groupby('n_obs')[metric].mean()
    count = focus_df.groupby('n_obs')[metric].count()

    se = focus_df.groupby('n_obs')[metric].std() / (count ** 0.5)
    ci = 1.96 * se

    ax.plot(mean.index, mean, color=colors[i], marker=markers[i], label="" + str(epoch) + " epochs", markevery=int(400/40)) # , markevery = 2
    ax.fill_between(mean.index, mean - ci, mean + ci, color=colors[i], alpha=0.2, linestyle=line_styles[i])

    # Show zero line.
    ax.axhline(y=0.0, color='black', linestyle=':')

    # Add the last number as percentage to the very right of the figure at each line.
    ax.text(mean.index[-1] + 20, mean.iloc[-1], "{:.1%}".format(mean.iloc[-1]), color=colors[i], fontsize=11)

# Show in lower left corner in light gray the number of simulation runs.
ax.text(0.80, 0.07, f"{int(count.mean())} sim. runs", transform=ax.transAxes, fontsize=9, color='gray')
ax.set_xlim(xlim)

    # # Ax ticks as percentage.
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))

# Show in lower left corner in light gray the number of simulation runs.
ax.text(0.80, 0.07, f"{int(count.mean())} sim. runs", transform=ax.transAxes, fontsize=9, color='gray')

ax.set_ylabel("Identification Rate")
ax.set_xlabel("Labeled Observations")

ax.legend(loc='upper left', title="Epochs Model Fit", frameon=True, facecolor='lightgray')

fig.tight_layout()

plt.savefig(script_dir + "../paper/generated/overfitting.pdf")

print("done")
