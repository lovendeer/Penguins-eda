# Penguin Body Measurements: Exploratory Data Analysis

An exploratory analysis of the Palmer Penguins dataset (344 penguins, 3 species) in Python.

## Questions
1. How do body measurements differ across species, islands and sex?
2. Are the differences statistically significant?
3. Which measurements move together?

## What I did
- Inspected and cleaned the data with pandas, using a reusable function
- Joined the data with a species lookup table (pd.merge)
- Summarised with groupby and aggregation
- Tested hypotheses with scipy (Welch's t-test) and computed correlations
- Visualised with matplotlib and seaborn

## Key findings
- Gentoo penguins are the heaviest (about 5,092 g); Adelie and Chinstrap are similar (about 3,700 g).
- Males are heavier than females (4,546 g vs 3,862 g, p < 0.001), and this holds within every species.
- Flipper length is the strongest predictor of body mass (r = 0.87).
- Simpson's paradox: bill length and depth correlate negatively overall (-0.23) but positively within each species (0.39 to 0.65).

## Run it
pip install pandas numpy matplotlib seaborn scipy
python analysis.py

## Data source
Gorman KB, Williams TD, Fraser WR (2014). PLoS ONE 9(3): e90081.

## Author
Lovender Kweyu
