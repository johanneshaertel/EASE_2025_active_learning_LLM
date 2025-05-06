import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

script_dir = os.path.dirname(os.path.abspath(__file__)) + "/"

plt.rcParams['font.size'] = '14'

# Make paper generated dir if not exists.

os.makedirs(script_dir  + "../paper/generated" , exist_ok=True)

# show more columns.
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

labelmap = {"no": "No Security Defect","yes": "Potential Security Defect","unclear": "Unclear","bot": "Bot", "language": "Non-English", "broken": "Broken"}

# count tags for iterations.

# Post qa.
df = pd.read_json(script_dir + "../data/final.json", lines = True)

# count tuples tag_pre_qa and tag_post_qa.

qa_changes = df.groupby(["tag_pre_qa", "tag_post_qa"]).size().reset_index(name = "count")
print(qa_changes)

# rewrite tag.
df["tag"] = df["tag_post_qa"]

# count the numer of broken
# count all tags and print numbers,

print(df.groupby("tag").size().reset_index(name = "count"))


print("Broken links: ", len(df[df["tag"] == "broken"]))

# Remove the broken links.
df = df[df["tag"] != "broken"]

# pivot
part2 = df.pivot_table(index = "tag", columns = "iteration", aggfunc = "size", fill_value = 0)
part2.loc["Total"] = part2.sum()

# Replace everythin with the cummulative sum.
part2 = part2.cumsum(axis = 1)

print(part2)

fit, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)

colors = ["red", "blue", "green", "black", "orange"]
tags = ["language", "no", "unclear", "yes"] # , "broken"
marker = ["o", "s", "D", "^", "v"]

# Do very light vertical lines that show the identification rate in 10% steps.
for i in range(1, 7):
    ax.axhline(0.1 * i, color = "gray", linestyle = "-", alpha = 0.2)

for i in range(1, len(part2.loc["Total"]) + 1):
    post = " (boot)" if i == 1 else ""

    ax.axvline(x = part2.loc["Total"][i], color = "gray", linestyle = "--")

    ax.text(part2.loc["Total"][i] + 10, 0.41, "iteration " + str(i) + post, color=(0.2, 0.2, 0.2), fontsize=11, rotation='vertical')


for tag, color, marker in zip(tags, colors, marker):
    ax.plot(part2.loc["Total"], (part2.loc[tag] / part2.loc["Total"]), marker = marker, label = labelmap[tag], color = color)
    
    # Add the last number as percentage to the very right of the figure at each line.
    ax.text(part2.loc["Total"][7] + 20, (part2.loc[tag] / part2.loc["Total"])[7], "{:.1%}".format((part2.loc[tag] / part2.loc["Total"])[7]), color=color, fontsize=11)



# Entropy/Rare Candidates
ax.add_patch(plt.Rectangle((380, -0.01), 680, 0.7, color = "red", alpha = 0.15))
ax.text(390, 0.65, "Entropy/Rare Candidates", color=(0.8, 0.4, 0.4), fontsize=13)

# Rare Candidates
ax.add_patch(plt.Rectangle((1110, -0.01), 300, 0.7, color = "green", alpha = 0.15))
ax.text(1120, 0.65, "Rare Candidates", color=(0.2, 0.8, 0.2), fontsize=13)


ax.set_xlabel('Labeled Observations')
ax.set_ylabel('Identification Rates')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))

ax.legend(title="Labels", loc='center')

plt.tight_layout()

plt.savefig(script_dir + "../paper/generated/empirical_numbers.pdf")
