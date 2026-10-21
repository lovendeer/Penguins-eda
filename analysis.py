# Penguin Body Measurements: Exploratory Data Analysis
# Author: Lovender Kweyu
#
# Goal: Explore the Palmer Penguins dataset to understand how body measurements
# differ across species, islands and sex, and test whether the differences are
# statistically meaningful.
# Skills: pandas (cleaning, grouping, joining), functions, descriptive statistics
# and hypothesis testing (scipy), matplotlib and seaborn visualization.
# Data source: Palmer Penguins dataset (Gorman et al., 2014), loaded from the
# seaborn-data repository on GitHub.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid")
URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
penguins = pd.read_csv(URL)

# ---------- 1. Inspect the data ----------
print("Rows, columns:", penguins.shape)
print()
penguins.info()
print()
print("Missing values per column:")
print(penguins.isna().sum())
print()
print(penguins.describe())

# ---------- 2. Clean the data ----------
def clean_penguins(df):
    """Drop rows with missing values and tidy the text columns."""
    out = df.dropna().copy()
    out["sex"] = out["sex"].str.title()
    out = out.reset_index(drop=True)
    return out

clean = clean_penguins(penguins)
print(f"Rows before: {len(penguins)}  |  Rows after: {len(clean)}  |  Dropped: {len(penguins) - len(clean)}")

# ---------- 3. Join with a species lookup table ----------
species_info = pd.DataFrame({
    "species": ["Adelie", "Chinstrap", "Gentoo"],
    "scientific_name": ["Pygoscelis adeliae", "Pygoscelis antarcticus", "Pygoscelis papua"],
})

merged = clean.merge(species_info, on="species", how="left")
print("Unmatched species after join:", merged["scientific_name"].isna().sum())

# ---------- 4. Summarise by group ----------
summary = (
    merged.groupby("species")
    .agg(
        count=("body_mass_g", "size"),
        mean_mass_g=("body_mass_g", "mean"),
        median_mass_g=("body_mass_g", "median"),
        std_mass_g=("body_mass_g", "std"),
        mean_flipper_mm=("flipper_length_mm", "mean"),
    )
    .round(1)
    .sort_values("mean_mass_g", ascending=False)
)
print(summary)
print()
print(pd.crosstab(merged["species"], merged["island"]))

# ---------- 5. Statistics ----------
num_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
corr = merged[num_cols].corr().round(2)
print(corr)

# Body mass: males vs females (Welch's t-test, does not assume equal variances)
male = merged.loc[merged["sex"] == "Male", "body_mass_g"]
female = merged.loc[merged["sex"] == "Female", "body_mass_g"]

t_stat, p_value = stats.ttest_ind(male, female, equal_var=False)
print(f"Mean mass, male:   {male.mean():.0f} g")
print(f"Mean mass, female: {female.mean():.0f} g")
print(f"t = {t_stat:.2f}, p = {p_value:.2e}")

# Does the sex difference hold inside each species?
rows = []
for sp, grp in merged.groupby("species"):
    m = grp.loc[grp["sex"] == "Male", "body_mass_g"]
    f = grp.loc[grp["sex"] == "Female", "body_mass_g"]
    t, p = stats.ttest_ind(m, f, equal_var=False)
    rows.append({"species": sp, "male_mean_g": round(m.mean()),
                 "female_mean_g": round(f.mean()), "p_value": p})
print(pd.DataFrame(rows))

# A surprising pattern: bill length vs bill depth.
# Across all penguins the correlation looks negative. Does it hold within species?
overall = merged["bill_length_mm"].corr(merged["bill_depth_mm"])
within = merged.groupby("species").apply(
    lambda g: g["bill_length_mm"].corr(g["bill_depth_mm"]), include_groups=False
).round(2)
print(f"Overall correlation: {overall:.2f}")
print("Within each species:")
print(within)

# ---------- 6. Visualizations ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].hist(merged["body_mass_g"], bins=20, color="steelblue", edgecolor="white")
axes[0].set_title("Distribution of body mass")
axes[0].set_xlabel("Body mass (g)")
axes[0].set_ylabel("Number of penguins")
sns.boxplot(data=merged, x="species", y="body_mass_g", hue="sex", ax=axes[1])
axes[1].set_title("Body mass by species and sex")
axes[1].set_xlabel("Species")
axes[1].set_ylabel("Body mass (g)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5.5))
sns.scatterplot(data=merged, x="flipper_length_mm", y="body_mass_g",
                hue="species", style="sex", s=70)
plt.title("Flipper length vs body mass")
plt.xlabel("Flipper length (mm)")
plt.ylabel("Body mass (g)")
plt.show()

plt.figure(figsize=(6, 4.5))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation between measurements")
plt.show()

sns.pairplot(merged, vars=num_cols, hue="species", height=2.2)
plt.show()

plt.figure(figsize=(8, 5.5))
sns.scatterplot(data=merged, x="bill_length_mm", y="bill_depth_mm", hue="species", s=60)
sns.regplot(data=merged, x="bill_length_mm", y="bill_depth_mm", scatter=False,
            color="black", line_kws={"linestyle": "--"})
plt.title("Bill depth vs bill length: overall trend (dashed) vs species")
plt.xlabel("Bill length (mm)")
plt.ylabel("Bill depth (mm)")
plt.show()

# ---------- 7. Key findings ----------
# 1. Gentoo penguins are the heaviest (mean about 5,092 g); Adelie (about 3,706 g)
#    and Chinstrap (about 3,733 g) are similar.
# 2. Males are heavier than females (4,546 g vs 3,862 g, Welch's t-test p < 0.001).
#    The difference holds inside every species.
# 3. Flipper length is the best single predictor of body mass (correlation 0.87).
# 4. Simpson's paradox: bill length and depth are negatively correlated overall
#    (-0.23) but positively correlated inside each species (0.39 to 0.65).
# 5. 11 of 344 rows (3.2%) had missing values and were dropped.
#
# Limitations / next steps: missing rows were dropped rather than imputed, and the
# analysis is descriptive. A next step is a model predicting species or body mass.
