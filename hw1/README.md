# Understanding the HECATE Galaxy Catalogue and our Pipeline
Dimitris Papachristopoulos (math250018)

- [<span class="toc-section-number">0.1</span> Dataset
  Description](#dataset-description)
- [<span class="toc-section-number">1</span> Reducing the
  dataset](#reducing-the-dataset)
  - [<span class="toc-section-number">1.1</span> Error
    Ratios](#error-ratios)
  - [<span class="toc-section-number">1.2</span>
    Metallicity](#metallicity)
  - [<span class="toc-section-number">1.3</span> Reduced
    dataset](#reduced-dataset)
- [<span class="toc-section-number">2</span>
  Preprocessing](#preprocessing)
  - [<span class="toc-section-number">2.1</span> Imbalance
    Problem](#imbalance-problem)
- [<span class="toc-section-number">3</span> PCA](#pca)
- [<span class="toc-section-number">4</span> Classical
  Models](#classical-models)
- [<span class="toc-section-number">5</span> Neural
  Network](#neural-network)
- [<span class="toc-section-number">6</span> Best Model](#best-model)
- [<span class="toc-section-number">7</span> Installation &
  Execution](#installation--execution)
  - [<span class="toc-section-number">7.0.1</span> 1. Clone the
    Repository](#1-clone-the-repository)
- [<span class="toc-section-number">8</span> Setup Virtual
  Environment](#setup-virtual-environment)
  - [<span class="toc-section-number">8.1</span> Using
    Poetry](#using-poetry)
  - [<span class="toc-section-number">8.2</span> Using
    venv](#using-venv)
- [<span class="toc-section-number">9</span> Run the
  Pipeline](#run-the-pipeline)
  - [<span class="toc-section-number">9.1</span> With
    Poetry](#with-poetry)
  - [<span class="toc-section-number">9.2</span> With venv](#with-venv)

For our project we are using the HECATE galaxy catalogue, which is a
catalogue of galaxies (Kovlakas K., Zezas A., Andrews J. J., Basu-Zych
A., Fragos T., Hornschemeier A., Kouroumpatzakis K., Lehmer B., Ptak. A
(2021). “The Heraklion Extragalactic Catalogue (HECATE): a value added
galaxy catalogue for multi-messenger astrophysics”. MNRAS in press.ADS
link)

You can download the catalogue from
[Kaggle](https://www.kaggle.com/datasets/tanmayshukla05/hecate-galaxy-catalogue?resource=download)
or from the original [source](https://hecate.ia.forth.gr/catalog.php),
and it is stored in the `data/` folder as `HECATE.csv`.

For my project I choose a classification task, where I want to predict
the nuclear activity classification of galaxies. This is usefull, since
for our analysis we usally classify galaxies based on their nuclear
activity.

However since many of the galaxies do not have a reliable measurement of
their nuclear activity, the completness of our sample is not good,
especially for near and very distant galaxies (the near galaxies can are
usually not well documented and the very distant galaxies are too faint
to be observed ).

The target variable is `CLASS_SP`, which is the nuclear activity
classification of galaxies. It is a categorical variable with the
following values:

- 0: star-forming
- 1: Seyfert
- 2: LINER
- 3: composite
- -1: unknown

## Dataset Description

``` python
import pandas as pd

df = pd.read_csv("data/HECATE.csv")

print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))
```

    Number of rows: 204733
    Number of columns: 100

Since the dataset is quite large and most of the columns are not
necessary for our analysis, we will only be using a subset of the
columns for our analysis. We will be using the following columns:

1.  `logM_HEC`: Decimal logarithm of the total stellar mass in solar
    masses.
2.  `logSFR_HEC`: Decimal logarithm of the homogenised star-formation
    rate in solar masses per year.
3.  `CLASS_SP`: Nuclear activity classification using the method in
    Stampoulis et al. 2019: 0=star-forming, 1=Seyfert, 2=LINER,
    3=composite, -1=unknown.
4.  `AGN_HEC`: Adopted activity classification based on the combination
    of class_sp and agn_s17: Y=AGN, N=non-AGN, ?=unknown.
5.  `T`: Numerical Hubble-type following the de Vaucouleurs et al. 1976
    system.
6.  `WF1`: 3.3μm-band (W1) apparent magnitude in the WISE forced
    photometry catalog (mag).
7.  `WF2`: 4.6μm-band (W2) apparent magnitude in the WISE forced
    photometry catalog (mag).
8.  `WF3`: 12μm-band (W3) apparent magnitude in the WISE forced
    photometry catalog (mag).
9.  `WF4`: 22μm-band (W4) apparent magnitude in the WISE forced
    photometry catalog (mag).
10. `U`: u-band SDSS apparent magnitude (mag).
11. `R`: r-band SDSS apparent magnitude (mag).
12. `G`: g-band SDSS apparent magnitude (mag).
13. `I`: i-band SDSS apparent magnitude (mag).
14. `Z`: z-band SDSS apparent magnitude (mag).
15. `UT`: Total U-band apparent magnitude (mag).
16. `BT`: B-band apparent magnitude (mag).
17. `VT`: V-band apparent magnitude (mag).
18. `IT`: I-band apparent magnitude (mag).

These columns will allow us to study the connections between star
formation, metallicity, Hubble type, and AGN activity. We will also use
the WISE W1 and W3 bands together with the UV and optical magnitudes to
investigate galaxy spectral properties.

To support this, we will create the color indices `u-g` and `g-r`.
Galaxy colors are useful because younger, star-forming systems tend to
be bluer, while older, more evolved galaxies are redder.

The WISE W1 and W3 bands are particularly helpful for this analysis. The
`W3-UT` color can trace hidden star formation from dust-obscured
regions, while the `(W3 + UV)/W1` color can serve as a proxy for the
stage of star formation in the galaxy (high ratio indicates ongoing star
formation).

We use both the total apparent magnitudes and the SDSS apparent
magnitudes, since they are measured differently and can provide
different information about the galaxy.

# Reducing the dataset

Before we move on to the analysis, we will sneak a quick peak into the
data to understand the distribution of the data and identify any
potential issues.

The final preprocessing will be done in the `preprocessing.py` script,
but we will modify the dataset here to create the reduced dataset that
we will use for our analysis. We will also check the distribution of the
key columns and the color indices to understand the data better.

Why are we doing this?: - Because we want to make sure that the data we
are using for our analysis is reliable and that we are not including any
data points that have large errors. This will help us to get more
accurate results from our analysis and to avoid any biases that may
arise from including unreliable data points. - The HECATE catalogue
contains a large number of data points, but not all of them are reliable
or even usefull for analysis. By reducing the dataset to only include
data points with reliable measurements, we can ensure that our analysis
is based on high-quality data and that our results are more robust. -
Since we can create a synthetic dataset for the homework, we can afford
to be more selective with the data points we include in our analysis, if
we are careful to maintain a large enough sample size for our analysis
and not introduce any biases by excluding certain types of galaxies or
measurements.

``` python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use('bmh')
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, 4)))

# New columns for the color indices
df['u-g'] = df['U'] - df['G']
df['g-r'] = df['G'] - df['R']
df['W3-UT'] = df['WF3'] - df['UT']
df['(W3+UT)/W1'] = (df['WF3'] + df['UT']) / df['WF1']

# Check the distribution of the key columns
key_columns_with_errors = ['T', 'WF1','WF2', 'WF3', 'WF4', 'UT', 'BT', 'VT','IT', 'U', 'R', 'G', 'R', 'I', 'Z']
error_columns = ['E_T', 'E_WF1','E_WF2', 'E_WF3', 'E_WF4', 'E_UT', 'E_BT', 'E_VT', 'E_IT', 'E_U', 'E_R', 'E_G']


key_columns_with_no_errors = ['CLASS_SP', 'AGN_HEC', 'logM_HEC', 'logSFR_HEC']
key_columns_with_flags = ['METAL']
flag_columns = ['FLAG_METAL']

key_columns = key_columns_with_errors + key_columns_with_no_errors + key_columns_with_flags

color_columns = ['u-g', 'g-r', 'W3-UT', '(W3+UT)/W1']

df_key_columns= df[key_columns+color_columns]
```

``` python
# Distribution of the key columns
df.hist(column=key_columns_with_errors, bins=30, figsize=(15, 10))
plt.suptitle('Distribution of Key Columns')
plt.show()
```

![](README_files/figure-commonmark/cell-4-output-1.png)

As we can see most of the distributions are skewed, which is why we will
imputate the NaN values with the median of the column.

``` python
df.hist(column=key_columns_with_no_errors, bins=30, figsize=(15, 10))
plt.suptitle('Distribution of Key Columns with Flags')
plt.show()
```

<div id="fig-key-columns-with-no-errors">

<img
src="README_files/figure-commonmark/fig-key-columns-with-no-errors-output-1.png"
id="fig-key-columns-with-no-errors" />

Figure 1

</div>

Most of our galaxies are of Class -1, which means they are unknown
(Explaining the problem from the introduction better). Because of this,
we will have to remove the data points with Class -1 from our dataset.

``` python
df.hist(column=color_columns, bins=30, figsize=(15, 10))
plt.suptitle('Distribution of Color Indices')
plt.show()
```

![](README_files/figure-commonmark/cell-6-output-1.png)

## Error Ratios

``` python
# check if e_* has any zero values to avoid division by zero
for err_col in error_columns:
    if (df[err_col] == 0).any():
        print(f"Warning: Column {err_col} contains zero values. These will be ignored in the ratio calculation.")
    else:
        print(f"Column {err_col} does not contain zero values.")
```

    Column E_T does not contain zero values.
    Column E_WF1 does not contain zero values.
    Column E_WF2 does not contain zero values.
    Column E_WF3 does not contain zero values.
    Column E_WF4 does not contain zero values.
    Column E_UT does not contain zero values.
    Column E_BT does not contain zero values.
    Column E_VT does not contain zero values.
    Column E_IT does not contain zero values.
    Column E_U does not contain zero values.
    Column E_R does not contain zero values.
    Column E_G does not contain zero values.

``` python
# New ratio df
ratio_df = pd.DataFrame()
for key_col, err_col in zip(key_columns_with_errors, error_columns):
    ratio_col_name = f"{key_col}_ratio"
    ratio_df[ratio_col_name] = np.abs(df[key_col] / df[err_col])*100
    print(f"Calculated ratio for {key_col} and {err_col} as {ratio_col_name}")

ratio_df.describe()
```

    Calculated ratio for T and E_T as T_ratio
    Calculated ratio for WF1 and E_WF1 as WF1_ratio
    Calculated ratio for WF2 and E_WF2 as WF2_ratio
    Calculated ratio for WF3 and E_WF3 as WF3_ratio
    Calculated ratio for WF4 and E_WF4 as WF4_ratio
    Calculated ratio for UT and E_UT as UT_ratio
    Calculated ratio for BT and E_BT as BT_ratio
    Calculated ratio for VT and E_VT as VT_ratio
    Calculated ratio for IT and E_IT as IT_ratio
    Calculated ratio for U and E_U as U_ratio
    Calculated ratio for R and E_R as R_ratio
    Calculated ratio for G and E_G as G_ratio

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | T_ratio | WF1_ratio | WF2_ratio | WF3_ratio | WF4_ratio | UT_ratio | BT_ratio | VT_ratio | IT_ratio | U_ratio | R_ratio | G_ratio |
|----|----|----|----|----|----|----|----|----|----|----|----|----|
| count | 136267.000000 | 1.230350e+05 | 1.221570e+05 | 1.119310e+05 | 9.641100e+04 | 99520.000000 | 185420.000000 | 7894.000000 | 7894.000000 | 123705.000000 | 1.237060e+05 | 1.237050e+05 |
| mean | 255.338703 | 5.085501e+05 | 1.620411e+05 | 3.192076e+04 | 7.564000e+03 | 36445.560091 | 4743.282594 | 17038.367859 | 17038.367859 | 73259.018864 | 3.640088e+05 | 3.429541e+05 |
| std | 313.904000 | 8.650206e+05 | 3.336339e+05 | 7.263040e+04 | 3.203298e+04 | 35953.248217 | 4835.825820 | 15414.099909 | 15414.099909 | 54500.142960 | 2.005090e+05 | 1.827832e+05 |
| min | 0.000000 | 2.775289e+00 | 1.213507e+00 | 6.001792e-01 | 5.375968e-02 | 124.341085 | 124.341085 | 45.271318 | 45.271318 | 51.575421 | 1.316937e+02 | 6.044271e+01 |
| 25% | 100.000000 | 8.478112e+04 | 2.581774e+04 | 3.872181e+03 | 1.214972e+03 | 12704.973776 | 3372.400000 | 11858.387474 | 11858.387474 | 36371.153846 | 2.063531e+05 | 2.027111e+05 |
| 50% | 188.461538 | 2.207350e+05 | 6.865842e+04 | 1.093788e+04 | 2.741662e+03 | 24216.774892 | 3670.000000 | 15795.000000 | 15795.000000 | 60648.387097 | 3.342200e+05 | 3.262600e+05 |
| 75% | 250.000000 | 6.254943e+05 | 1.960397e+05 | 3.245651e+04 | 6.854551e+03 | 47310.064103 | 4802.818891 | 18197.183099 | 18197.183099 | 95868.421053 | 5.096000e+05 | 5.003333e+05 |
| max | 4950.000000 | 3.832589e+07 | 3.065699e+07 | 4.280679e+06 | 8.396190e+06 | 708550.000000 | 90666.666667 | 269375.000000 | 269375.000000 | 685300.000000 | 1.329600e+06 | 1.337700e+06 |

</div>

``` python
ratio_df.hist(bins=30, figsize=(15, 10))
plt.suptitle('Distribution of Key Column to Error Ratios')
plt.show()
```

![](README_files/figure-commonmark/cell-9-output-1.png)

As we can see we have most galaxies with a Magnitude to Error ratio
greater than 3. This means that these galaxies have reliable
measurements. We will keep only the data points with a Magnitude to
Error ratio greater than 3.

``` python
# if ratio is less than 3, we will ignore those data points
for ratio_col in ratio_df.columns:
    ratio_df_reduced = ratio_df[ratio_df[ratio_col] > 3]  # keeping only data points with ratio greater than 3
ratio_df_reduced.count()
```

    T_ratio       96304
    WF1_ratio    123034
    WF2_ratio    122156
    WF3_ratio    111930
    WF4_ratio     96410
    UT_ratio      97133
    BT_ratio     115885
    VT_ratio       3798
    IT_ratio       3798
    U_ratio      123705
    R_ratio      123705
    G_ratio      123705
    dtype: int64

## Metallicity

The metallicity of a galaxy is an important property that can provide
insights into its formation and evolution. However, the metallicity
measurements in our dataset have a flag column that indicates whether
the measurement is reliable or not. We will check the distribution of
the metallicity values and the flags to understand how many data points
we can use for our analysis.

``` python
# Check the distribution of metallicity and its flag
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['METAL'], bins=30, kde=True)
plt.title('Distribution of Metallicity')
plt.xlabel('Metallicity')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(df['FLAG_METAL'], bins=30, kde=False)
plt.title('Distribution of Metallicity Flags')
plt.xlabel('Flag Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

# Grouping by 'FLAG_METAL'
sns.histplot(data=df, x='T', hue='FLAG_METAL', bins=30, kde=True, 
             element="step", palette="viridis")

plt.title('Distribution of Metallicity Grouped by Flag Value')
plt.xlabel('Metallicity (12+log(O/H))')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.3)
plt.show()
```

![](README_files/figure-commonmark/cell-11-output-1.png)

![](README_files/figure-commonmark/cell-11-output-2.png)

With the flags being:

- -1=missing

- 0=reliable

- 1=O3N2 ratio \>2 (outside the PP04 range)

- 2=low signal-to-noise ratio (\<3 for the weakest line).

Since the metallicity flags -1 and 0 follow a similar distribution for
different Hubble types and the flags 1 and 2 correspond to a small
number of data points, we can cut the datapoints with flags -1 without
being biased towards a specific type of galaxy. This is why we will keep
only 0.5% of the data points with flag -1, to simulate the effect of
having missing data but without losing the important information that
the metallicity measurements provide for our analysis.

``` python
# Separate the -1 flags from the rest
df_flag_minus_1 = df[df['FLAG_METAL'] == -1]
df_others = df[df['FLAG_METAL'] != -1]

# Sample 2% of the -1 flags
df_flag_minus_1_reduced = df_flag_minus_1.sample(frac=0.005, random_state=42)

# Combine them back together
df_metal_reduced = pd.concat([df_others, df_flag_minus_1_reduced])

# Verify the new distribution
print(df_metal_reduced['FLAG_METAL'].value_counts())
```

    FLAG_METAL
     0    62728
     2      882
    -1      702
     1      670
    Name: count, dtype: int64

## Reduced dataset

``` python
# reduce the original df to only include rows where all key column to error ratios are greater than 3 and only include 0.5% of the data points with flag -1 
# First, we will create a mask for the ratio conditions
ratio_mask = np.ones(len(df), dtype=bool)
for ratio_col in ratio_df.columns:
    df_reduced = df_key_columns[ratio_df[ratio_col] > 3]
# Then, we will create a mask for the metallicity flag condition
metal_mask = (df['FLAG_METAL'] != -1) | ((df['FLAG_METAL'] == -1) & (df.index.isin(df_flag_minus_1_reduced.index)))
# Finally, we will combine the masks to create the reduced dataset
df_reduced = df[ratio_mask & metal_mask]
df_reduced = df_reduced[key_columns + color_columns]
df_reduced.info()
```

    <class 'pandas.DataFrame'>
    Index: 64982 entries, 4 to 204700
    Data columns (total 24 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   T           58858 non-null  float64
     1   WF1         64438 non-null  float64
     2   WF2         64302 non-null  float64
     3   WF3         61241 non-null  float64
     4   WF4         54032 non-null  float64
     5   UT          64048 non-null  float64
     6   BT          64811 non-null  float64
     7   VT          1040 non-null   float64
     8   IT          1040 non-null   float64
     9   U           64558 non-null  float64
     10  R           64559 non-null  float64
     11  G           64558 non-null  float64
     12  R           64559 non-null  float64
     13  I           64559 non-null  float64
     14  Z           64557 non-null  float64
     15  CLASS_SP    64982 non-null  int64  
     16  AGN_HEC     64982 non-null  str    
     17  logM_HEC    34663 non-null  float64
     18  logSFR_HEC  49977 non-null  float64
     19  METAL       64280 non-null  float64
     20  u-g         64558 non-null  float64
     21  g-r         64558 non-null  float64
     22  W3-UT       60773 non-null  float64
     23  (W3+UT)/W1  60727 non-null  float64
    dtypes: float64(22), int64(1), str(1)
    memory usage: 12.4 MB

Now that we have a reduced dataset with reliable measurements, we can
proceed with our analysis.

``` python
print("Original dataset shape:", df.shape)
print("Reduced dataset shape:", df_reduced.shape)
```

    Original dataset shape: (204733, 104)
    Reduced dataset shape: (64982, 24)

It is important to note that the reduced dataset still contains a large
number of data points, more than 8,000 rows and 8 columns, as required
from our project.

# Preprocessing

To begin with, we drop the columns that are not useful for our analysis
and we remove the rows with no class label. We also drop most of the
rows with metallicity flag -1, since they are not reliable (keep 0.5% of
them to simulate noise).

After, we spit the data into train, validation and test sets(80%, 10%,
10%) and: - we encode the target variable using Label Encoding. - We cap
the outliers using the IQR method. - We imputate the NaN values using
the median of the column. - We filter out the data with an Error Ratio
Magnitude/Error\< 3. - We calculate the new colors as explained in the
introduction. - We scale the data using the Standard Scaler, since we
don’t have any outliers after the treatment (Robust Scaler) and our data
are not bound or uniformly distributed (MinMax Scaler).

## Imbalance Problem

As shown by <a href="#fig-key-columns-with-no-errors"
class="quarto-xref">Figure 1</a>, the dataset is imbalanced. Even
without the Class -1 galaxies, we have a lot of galaxies of Class 0,
which means they are Star Forming Galaxies.

``` python
# Drop class -1 galaxies
df_reduced = df_reduced[df_reduced['CLASS_SP'] != -1]

# Check the percentages of the target variable
print("Percentage of each class in the reduced dataset:")
print(df_reduced['CLASS_SP'].value_counts(normalize=True) * 100)
```

    Percentage of each class in the reduced dataset:
    CLASS_SP
    0    84.463286
    3     8.677660
    2     4.945551
    1     1.913503
    Name: proportion, dtype: float64

Usually, the solution to this problem is to use SMOTE or
RandomUnderSampler, however we will use a hybrid solution to this
problem. We will use Undersampling for the majority class and SMOTE for
the minority class. My final dataset should not be more than 10% larger
from the original.

The reason for using this hybrid approach is to avoid the information
loss that comes with undersampling and the overfitting that can come
with oversampling/creation of synthetic and duplicate datapoints.

# PCA

# Classical Models

We use 5 classical models to solve this problem:

- Logistic Regression
- Random Forest
- Support Vector Machine
- Decision Tree
- Xgboost

We train all of the models using a grid of parameters and we evaluate
them using ROC-AUC

The best model is the one with the highest ROC-AUC on the validation
set.

We use ROC-AUC as the main evaluation metric because it is robust to
class imbalance and evaluates the model’s ability to distinguish between
classes across all possible classification thresholds, rather than just
relying on a single fixed threshold (like 0.5). (we use One-vs-Rest for
multiclass classification)

# Neural Network

We have created a NN with 3 hidden layers \[128, 64, 32\] with ReLU
activation function and dropout of 0.2 between the layers. We train the
model using the Adam optimizer and the cross-entropy loss function. We
use early stopping to prevent overfitting.

# Best Model

The pipeline automatically identifies and saves the overall “Best Model”
by performing a head-to-head comparison between the optimal classical
model (found via grid search) and the trained neural network. Both
models are evaluated on the unseen test set using ROC-AUC as the primary
metric (falling back to Accuracy if necessary). The winning model is
persisted in the `/models` directory as `best_model.pkl` (for classical)
or `best_model.pt` (for neural network), ensuring that the most reliable
predictor is always available for subsequent inference or deployment.

# Installation & Execution

Follow these steps to set up the environment and run the complete
machine learning pipeline.

### 1. Clone the Repository

``` {bash}
git clone https://github.com/dim-papach/AI-Handson-Assigments.git
cd AI-Handson-Assigments/hw1
```

# Setup Virtual Environment

## Using Poetry

``` {bash}
poetry install
```

## Using venv

``` {bash}
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Run the Pipeline

## With Poetry

``` {bash}
poetry run python hw1/src/main.py
```

## With venv

Can also be used with poetry

``` {bash}
./venv/bin/python hw1/src/main.py
```
