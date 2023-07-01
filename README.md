# Bike Sharing Demand
:::

::: {.cell .markdown id="RLYJi1KP8_sN"}
## Introduction

In the following project we\'ll predict bike rental demand in the
Capital Bikeshare program in Washington, D.C using historical weather
data from the Bike Sharing Demand dataset available through Kaggle. For
this purpose, we will be using the random forests algorithm.
:::

::: {.cell .markdown id="Mv2QKFFZ9SgO"}
![bike](vertopal_59a6188fe9e84b3fac77a2736905bb99/d78ce9bb512ba2daf191b2d02729ba41ab1924ee.jpg)
:::

::: {.cell .code execution_count="4" id="ShFSg8p09Hlw"}
``` python
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error as MSE
```
:::

::: {.cell .markdown id="FJi2FNH1-gYT"}
## Dataset
:::

::: {.cell .code execution_count="5" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="ERjoP7_S9rw0" outputId="e5194198-71f3-491a-e9dc-25ff022dcd32"}
``` python
df_bikes = pd.read_csv('bikes.csv')
print(df_bikes)
```

::: {.output .stream .stdout}
          hr  holiday  workingday  temp   hum  windspeed  cnt  instant  mnth  yr  \
    0      0        0           0  0.76  0.66     0.0000  149    13004     7   1   
    1      1        0           0  0.74  0.70     0.1343   93    13005     7   1   
    2      2        0           0  0.72  0.74     0.0896   90    13006     7   1   
    3      3        0           0  0.72  0.84     0.1343   33    13007     7   1   
    4      4        0           0  0.70  0.79     0.1940    4    13008     7   1   
    ...   ..      ...         ...   ...   ...        ...  ...      ...   ...  ..   
    1483  19        0           1  0.80  0.49     0.1343  452    14487     8   1   
    1484  20        0           1  0.80  0.49     0.1343  356    14488     8   1   
    1485  21        0           1  0.76  0.58     0.1940  303    14489     8   1   
    1486  22        0           1  0.76  0.58     0.1940  277    14490     8   1   
    1487  23        0           1  0.74  0.62     0.1045  174    14491     8   1   

          Clear to partly cloudy  Light Precipitation  Misty  
    0                          1                    0      0  
    1                          1                    0      0  
    2                          1                    0      0  
    3                          1                    0      0  
    4                          1                    0      0  
    ...                      ...                  ...    ...  
    1483                       1                    0      0  
    1484                       1                    0      0  
    1485                       1                    0      0  
    1486                       1                    0      0  
    1487                       1                    0      0  

    [1488 rows x 13 columns]
:::
:::

::: {.cell .code execution_count="21" id="tSYLOoBYGlFX"}
``` python
sns.set_palette("rocket")
```
:::

::: {.cell .markdown id="aHYSSL7zGNCG"}
## Data Visualization
:::

::: {.cell .code execution_count="22" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":625}" id="rmwaZS7bGIr5" outputId="ab21ffef-6596-4255-bce5-5b4667800618"}
``` python
# Create a jointplot similar to the JointGrid
sns.jointplot(x="hum",
        y="cnt",
        kind='reg',
        data=df_bikes)

plt.show()
plt.clf()
```

::: {.output .display_data}
![](vertopal_59a6188fe9e84b3fac77a2736905bb99/2a4fc15b1f12a75a7fe34d6ad70d5c01a8849c77.png)
:::

::: {.output .display_data}
    <Figure size 640x480 with 0 Axes>
:::
:::

::: {.cell .code execution_count="25" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":625}" id="WA2szlLOHstx" outputId="37126e81-9a19-4633-f1d1-6ddcf67ce887"}
``` python
# Plot temp vs. total_rentals as a regression plot
sns.jointplot(x="temp",
         y="cnt",
         kind='reg',
         data=df_bikes,
         order=2,
         xlim=(0, 1)
         )

plt.show()
plt.clf()
```

::: {.output .display_data}
![](vertopal_59a6188fe9e84b3fac77a2736905bb99/ee0ba11ba38801831aabc930a064f84111b6a6e7.png)
:::

::: {.output .display_data}
    <Figure size 640x480 with 0 Axes>
:::
:::

::: {.cell .code execution_count="28" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":625}" id="kV1P3jCcIUlx" outputId="b88e388a-a777-4f80-8e4a-3981dfd68e8b"}
``` python
# Replicate the above plot but only for registered riders
g = sns.jointplot(x="windspeed",
             y="cnt",
             kind='scatter',
             data=df_bikes,
             marginal_kws=dict(bins=10))
g.plot_joint(sns.kdeplot)

plt.show()
plt.clf()
```

::: {.output .display_data}
![](vertopal_59a6188fe9e84b3fac77a2736905bb99/6f6f9d009394aa536976395b925a30e4db207db2.png)
:::

::: {.output .display_data}
    <Figure size 640x480 with 0 Axes>
:::
:::

::: {.cell .markdown id="hYJytWJW-eVj"}
## Train/Test split
:::

::: {.cell .code id="VideEEwh9y1q"}
``` python
X = df_bikes.drop(['cnt'], axis=1)
y = df_bikes[['cnt']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
```
:::

::: {.cell .markdown id="MHf1fB8M-wvG"}
# Instantiate RandomForest Model
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":130}" id="-Ne_AoKZ-dKY" outputId="b1138f1c-5895-49c5-836a-3e6f614e30a1"}
``` python
# Instantiate rf
rf = RandomForestRegressor(n_estimators=25, random_state=2)

# Fit to the training data
rf.fit(X_train, y_train)
```

::: {.output .stream .stderr}
    <ipython-input-7-2791df2ad065>:5: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      rf.fit(X_train, y_train)
:::

::: {.output .execute_result execution_count="7"}
```{=html}
<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestRegressor(n_estimators=25, random_state=2)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestRegressor</label><div class="sk-toggleable__content"><pre>RandomForestRegressor(n_estimators=25, random_state=2)</pre></div></div></div></div></div>
```
:::
:::

::: {.cell .markdown id="E1tGcQ5R_NzS"}
## Evaluate rf
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="sM-zELcg-vMX" outputId="e108fbc8-daf8-4b9d-ee95-84b7f2bcd706"}
``` python
# Predict on test data
y_pred = rf.predict(X_test)

# Evaluate the test
rmse_test = MSE(y_test, y_pred) ** (1/2)

print('Test set RMSE of rf: {:.3f}'.format(rmse_test))
```

::: {.output .stream .stdout}
    Test set RMSE of rf: 50.424
:::
:::

::: {.cell .markdown id="p9j2VGkp_diL"}
## Result

The test set RMSE achieved by rf is significantly smaller than that
achieved by a single CART!
:::

::: {.cell .markdown id="xndBEzl9_1eQ"}
## Feature importance
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":421}" id="hXHRF8Qg_W7E" outputId="3e610ed4-0b1e-41f8-ccdd-c4a1b95781b7"}
``` python
# Create a pd.Series of features importance
importances = pd.Series(data=rf.feature_importances_, index=X_train.columns)

# sort importances
importances_sorted = importances.sort_values()

# Barplot
importances_sorted.plot(kind='barh', color='lightgreen')
plt.title('Feature Importances')
plt.show()
```

::: {.output .display_data}
![](vertopal_59a6188fe9e84b3fac77a2736905bb99/417037c1e603386e1ada412bd8466a159f05698b.png)
:::
:::

::: {.cell .markdown id="ezQpqoV3CDRD"}
## Gradient Boosing regressor
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="CQiLX4SAAZE8" outputId="5ca8985b-7a41-4317-9992-33f79c4b760b"}
``` python
# Instantiate GB
gb = GradientBoostingRegressor(max_depth=4, n_estimators=200, random_state=2)

# Fit
gb.fit(X_train, y_train)

# Predict
y_pred = gb.predict(X_test)
```

::: {.output .stream .stderr}
    /usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_gb.py:437: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
:::
:::

::: {.cell .markdown id="eNPUUf0ECfoN"}
## Evaluate gb
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="vXuxKhQtCPnn" outputId="51c44c0a-694b-44e8-f669-e0789402dcb4"}
``` python
# MSE
mse_test = MSE(y_test, y_pred)

# RMSE
rmse_test = mse_test ** (1/2)

print('Test set RMSE of gb: {:.3f}'.format(rmse_test))
```

::: {.output .stream .stdout}
    Test set RMSE of gb: 50.340
:::
:::

::: {.cell .markdown id="s3C5uG_TC7sC"}
## Stochastic Gradient Boosting Regressor
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="9syFT0evCyYf" outputId="35c28dce-af50-4daa-b441-e63dc24dc874"}
``` python
# Instantiate sgbr
sgbr = GradientBoostingRegressor(max_depth=4,
            subsample=0.9,
            max_features=0.75,
            n_estimators=200,
            random_state=2)

# Fit sgbr to the training set
sgbr.fit(X_train, y_train)

# Predict test set labels
y_pred = sgbr.predict(X_test)
```

::: {.output .stream .stderr}
    /usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_gb.py:437: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
:::
:::

::: {.cell .markdown id="gQQte2ULDGbw"}
## Evaluate sgbr
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="4CeGZUfCDAhO" outputId="fdffa794-7007-4369-cc1b-adc78f292818"}
``` python
# Compute test set MSE
mse_test = MSE(y_test, y_pred)

# Compute test set RMSE
rmse_test = mse_test ** (0.5)

# Print rmse_test
print('Test set RMSE of sgbr: {:.3f}'.format(rmse_test))
```

::: {.output .stream .stdout}
    Test set RMSE of sgbr: 50.294
:::
:::

::: {.cell .markdown id="DynsYjAbFxyi"}
## Hyperparameter grid of rf
:::

::: {.cell .code id="iSndUcQgDHh3"}
``` python
# Define the dictionary 'params_rf'
params_rf = {"n_estimators":[100, 350, 500], 'max_features': ['log2', 'auto', 'sqrt'], 'min_samples_leaf': [2, 10, 30]}
```
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":227}" id="rE3qnw0vF3JO" outputId="04d23c92-7ed3-412b-dd98-5a0e89b666f4"}
``` python
# Instantiate grid_rf
grid_rf = GridSearchCV(estimator=rf, param_grid=params_rf, scoring='neg_mean_squared_error', cv=3, verbose=1, n_jobs=-1)

grid_rf.fit(X_train, y_train)
```

::: {.output .stream .stdout}
    Fitting 3 folds for each of 27 candidates, totalling 81 fits
:::

::: {.output .stream .stderr}
    /usr/local/lib/python3.10/dist-packages/sklearn/model_selection/_search.py:909: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      self.best_estimator_.fit(X, y, **fit_params)
    /usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:413: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features=1.0` or remove this parameter as it is also the default value for RandomForestRegressors and ExtraTreesRegressors.
      warn(
:::

::: {.output .execute_result execution_count="23"}
```{=html}
<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=3,
             estimator=RandomForestRegressor(n_estimators=25, random_state=2),
             n_jobs=-1,
             param_grid={&#x27;max_features&#x27;: [&#x27;log2&#x27;, &#x27;auto&#x27;, &#x27;sqrt&#x27;],
                         &#x27;min_samples_leaf&#x27;: [2, 10, 30],
                         &#x27;n_estimators&#x27;: [100, 350, 500]},
             scoring=&#x27;neg_mean_squared_error&#x27;, verbose=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=3,
             estimator=RandomForestRegressor(n_estimators=25, random_state=2),
             n_jobs=-1,
             param_grid={&#x27;max_features&#x27;: [&#x27;log2&#x27;, &#x27;auto&#x27;, &#x27;sqrt&#x27;],
                         &#x27;min_samples_leaf&#x27;: [2, 10, 30],
                         &#x27;n_estimators&#x27;: [100, 350, 500]},
             scoring=&#x27;neg_mean_squared_error&#x27;, verbose=1)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: RandomForestRegressor</label><div class="sk-toggleable__content"><pre>RandomForestRegressor(n_estimators=25, random_state=2)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestRegressor</label><div class="sk-toggleable__content"><pre>RandomForestRegressor(n_estimators=25, random_state=2)</pre></div></div></div></div></div></div></div></div></div></div>
```
:::
:::

::: {.cell .markdown id="RHIto4NFGU9N"}
## Evaluate optimal forest
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="JsKVeVueGRfJ" outputId="410c341e-8ad2-47da-b4e3-5441c4901859"}
``` python
# Best estimator
best_model = grid_rf.best_estimator_

# Predict
y_pred = best_model.predict(X_test)

# Compute rmse_test
rmse_test = MSE(y_test, y_pred) ** (1/2)

# Print rmse_test
print('Test RMSE of best model: {:.3f}'.format(rmse_test))
```

::: {.output .stream .stdout}
    Test RMSE of best model: 47.950
:::
:::

::: {.cell .code id="MYFDJiXMGxUK"}
``` python
```
:::
