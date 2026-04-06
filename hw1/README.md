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
3.  `class_sp`: Nuclear activity classification using the method in
    Stampoulis et al. 2019: 0=star-forming, 1=Seyfert, 2=LINER,
    3=composite, -1=unknown.
4.  `agn_hec`: Adopted activity classification based on the combination
    of class_sp and agn_s17: Y=AGN, N=non-AGN, ?=unknown.
5.  `t`: Numerical Hubble-type following the de Vaucouleurs et al. 1976
    system.
6.  `wf1`: 3.3μm-band (W1) apparent magnitude in the WISE forced
    photometry catalog (mag).
7.  `wf3`: 12μm-band (W3) apparent magnitude in the WISE forced
    photometry catalog (mag).
8.  `ut`: Total U-band apparent magnitude (mag).
9.  `u`: u-band SDSS apparent magnitude (mag).
10. `r`: r-band SDSS apparent magnitude (mag).

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
key_columns_with_errors = ['T', 'WF1', 'WF3', 'UT', 'U', 'R', 'G']
error_columns = ['E_T', 'E_WF1', 'E_WF3', 'E_UT', 'E_U', 'E_R', 'E_G']


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

``` python
df.hist(column=key_columns_with_no_errors, bins=30, figsize=(15, 10))
plt.suptitle('Distribution of Key Columns with Flags')
plt.show()
```

![](README_files/figure-commonmark/cell-5-output-1.png)

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
    Column E_WF3 does not contain zero values.
    Column E_UT does not contain zero values.
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
    Calculated ratio for WF3 and E_WF3 as WF3_ratio
    Calculated ratio for UT and E_UT as UT_ratio
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

|  | T_ratio | WF1_ratio | WF3_ratio | UT_ratio | U_ratio | R_ratio | G_ratio |
|----|----|----|----|----|----|----|----|
| count | 136267.000000 | 1.230350e+05 | 1.119310e+05 | 99520.000000 | 123705.000000 | 1.237060e+05 | 1.237050e+05 |
| mean | 255.338703 | 5.085501e+05 | 3.192076e+04 | 36445.560091 | 73259.018864 | 3.640088e+05 | 3.429541e+05 |
| std | 313.904000 | 8.650206e+05 | 7.263040e+04 | 35953.248217 | 54500.142960 | 2.005090e+05 | 1.827832e+05 |
| min | 0.000000 | 2.775289e+00 | 6.001792e-01 | 124.341085 | 51.575421 | 1.316937e+02 | 6.044271e+01 |
| 25% | 100.000000 | 8.478112e+04 | 3.872181e+03 | 12704.973776 | 36371.153846 | 2.063531e+05 | 2.027111e+05 |
| 50% | 188.461538 | 2.207350e+05 | 1.093788e+04 | 24216.774892 | 60648.387097 | 3.342200e+05 | 3.262600e+05 |
| 75% | 250.000000 | 6.254943e+05 | 3.245651e+04 | 47310.064103 | 95868.421053 | 5.096000e+05 | 5.003333e+05 |
| max | 4950.000000 | 3.832589e+07 | 4.280679e+06 | 708550.000000 | 685300.000000 | 1.329600e+06 | 1.337700e+06 |

</div>

``` python
ratio_df.hist(bins=30, figsize=(15, 10))
plt.suptitle('Distribution of Key Column to Error Ratios')
plt.show()
```

![](README_files/figure-commonmark/cell-9-output-1.png)

``` python
# if ratio is less than 3, we will ignore those data points
for ratio_col in ratio_df.columns:
    ratio_df_reduced = ratio_df[ratio_df[ratio_col] > 3]  # keeping only data points with ratio greater than 3
ratio_df_reduced.count()
```

    T_ratio       96304
    WF1_ratio    123034
    WF3_ratio    111930
    UT_ratio      97133
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
    Data columns (total 16 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   T           58858 non-null  float64
     1   WF1         64438 non-null  float64
     2   WF3         61241 non-null  float64
     3   UT          64048 non-null  float64
     4   U           64558 non-null  float64
     5   R           64559 non-null  float64
     6   G           64558 non-null  float64
     7   CLASS_SP    64982 non-null  int64  
     8   AGN_HEC     64982 non-null  str    
     9   logM_HEC    34663 non-null  float64
     10  logSFR_HEC  49977 non-null  float64
     11  METAL       64280 non-null  float64
     12  u-g         64558 non-null  float64
     13  g-r         64558 non-null  float64
     14  W3-UT       60773 non-null  float64
     15  (W3+UT)/W1  60727 non-null  float64
    dtypes: float64(14), int64(1), str(1)
    memory usage: 8.4 MB

Now that we have a reduced dataset with reliable measurements, we can
proceed with our analysis.

``` python
print("Original dataset shape:", df.shape)
print("Reduced dataset shape:", df_reduced.shape)
```

    Original dataset shape: (204733, 104)
    Reduced dataset shape: (64982, 16)

It is important to note that the reduced dataset still contains a large
number of data points, more than 8,000 rows and 8 columns, as required
from our project.
