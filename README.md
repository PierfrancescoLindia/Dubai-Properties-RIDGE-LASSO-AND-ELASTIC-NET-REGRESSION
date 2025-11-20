# Dubai-Properties-RIDGE-LASSO-AND-ELASTIC-NET-REGRESSION

# Statistical Learning for Dubai Real Estate Prices

This project analyses the determinants of property prices in Dubai using a dataset of 1,905 observations and 38 variables, sourced from Kaggle. The focus is on building and comparing statistical learning models (Ridge, Lasso, Elastic Net) for predictive purposes in the presence of strong multicollinearity.

---

## 1. Objective of the Study

The work analyses the determinants of property prices in Dubai using a dataset of 1,905 observations and 38 variables, sourced from Kaggle.  
The aims are to:

- build a robust predictive model of the property price (`Price`);
- handle the strong multicollinearity among regressors;
- compare penalized regression techniques (Ridge, Lasso, Elastic Net);
- select the best model by comparing **Mean Squared Error (MSE)** values obtained via cross-validation techniques (LOOCV and K-fold).

---

## 2. Dataset and Data Preparation

### 2.1 Context

The dataset refers to apartments located mainly in prestigious areas of Dubai (Downtown Dubai, Dubai Marina and other locations). The variables include:

- **Structural characteristics**: number of bedrooms and bathrooms, size in sqft, total price, price per sqft.
- **Location**: neighborhood (`Neighborhood`), latitude, longitude.
- **Quality**: qualitative level of the property.
- **Facilities and amenities**: presence of balconies, barbecue area, built-in wardrobes, central air conditioning, children’s play areas, shared/private pool, gym, spa, private garden, security, view of landmarks and water, etc. (originally boolean variables).

All variables are used as regressors to explain `Price`.

### 2.2 Treatment of Variables

Main preprocessing choices (see tables and descriptions on pp. 4–5):

- **Removal of `ID`**: dropped because it is only a unique identifier, not informative for prediction and a potential source of overfitting.
- **Recoding of `Quality`**: transformed into three dummies:
  - `qualityhigh`, `qualitymedium`, `qualitylow` (c–1 dummy coding scheme).
- **Recoding of `Neighborhood`**:
  - 0 = Downtown Dubai  
  - 1 = Dubai Marina  
  - 2 = other locations
- **Boolean variables** (facilities, amenities): transformed into binary dummies 0/1 (e.g. `balcony`, `shared_pool`, `security`, `pets_allowed`, etc.).

---

## 3. Initial Linear Model and Multicollinearity

### 3.1 Correlation Analysis

Using a **correlation heatmap** (pp. 6–7), relevant linear relationships are identified:

- **Strong positive correlations**:
  - `no_of_bedrooms` – `size_in_sqft`
  - `price` – `size_in_sqft`
  - `price` – `no_of_bathrooms`
  - `price_per_sqft` – `price`
- **Strong negative correlations**:
  - `quality_Low` – `quality_High`
  - geographic coordinates (`latitude`, `longitude`) with `neighborhood`

These relationships suggest potential multicollinearity issues among regressors.

### 3.2 Estimation of the Initial OLS Model

A full linear model is estimated with **dependent variable `price`** and all available regressors (R script on p. 7).

Model goodness-of-fit indicators:

- **Residual Standard Error (RSE)** ≈ 902,100 on 1,866 degrees of freedom  
- **R²** ≈ 0.906  
- **Adjusted R²** ≈ 0.9041  
- **F-statistic** ≈ 473.4 with p-value < 2.2e-16  

Thus, the model shows high explanatory power overall, but:

- many coefficients are **not significant** (high p-values);
- this, together with high correlations among regressors, is a clear sign of **multicollinearity**.

### 3.3 VIF and Tolerance Analysis

To formally measure multicollinearity, **Variance Inflation Factor (VIF)** and **Tolerance** are computed for each regressor (detailed table on pp. 10–11):

- Interpretation:
  - VIF > 5 → high multicollinearity  
  - VIF > 10 → severe multicollinearity  
  - Tolerance < 0.2 → high multicollinearity

Some results:

- Quality variables:
  - `quality_High` → VIF ≈ 20.3
  - `quality_Low` → VIF ≈ 87.0
  - `quality_Medium` → VIF ≈ 92.2  
    → very low tolerance (~0.01): extremely severe multicollinearity.
- Structural variables:
  - `no_of_bedrooms` VIF ≈ 4.8
  - `no_of_bathrooms` VIF ≈ 4.1
- Other variables with high VIF: `childrens_pool`, `lobby_in_building`, `networked`, `vastu_compliant`, etc.

Conclusion: the dataset exhibits **widespread multicollinearity**, which makes OLS estimates unreliable and justifies the use of **regularization techniques**.

---

## 4. Resampling Techniques (Cross-Validation)

Since the analysis is **predictive**, the focus is on prediction error rather than on the individual significance of coefficients. Two validation schemes are used:

1. **Leave-One-Out Cross Validation (LOOCV)**  
   - each observation serves in turn as the test set (n models fitted on n−1 observations);  
   - the average MSE is computed across all iterations.

2. **K-Fold Cross Validation (K=10)**  
   - the dataset is split into 10 folds of equal size;  
   - in turn, 9 folds are used for training and 1 as test;  
   - the average MSE across the 10 test folds is computed.

Both techniques are applied to a sequence of models of increasing complexity (including interactions and polynomial terms for `latitude`, `longitude`, ratios between `size_in_sqft` and `price_per_sqft`, etc., scripts on pp. 12–13).

Result: **Model 4** (the most complex one) shows the lowest MSE in both validation schemes, with LOOCV providing a slightly better but more biased estimate compared to K-fold. However, the two techniques yield consistent results.

---

## 5. Ridge Regression

### 5.1 Setup

Ridge Regression introduces an **L2 penalty** on the sum of squared coefficients, controlled by the parameter **λ**:

- λ = 0 → standard OLS regression;
- large λ → coefficients are shrunk towards zero (but never exactly zero).

The dataset is split into:

- **X**: matrix of all independent variables;
- **y**: price vector.

A Ridge model is estimated on the entire dataset (p. 14) and the **coefficient path as a function of log(λ)** is analysed: as λ increases, coefficients gradually converge to smaller values.

### 5.2 Choice of λ via Cross-Validation

The following are used:

- **K-Fold CV (K=10)** to choose λ (MSE vs log(λ) plot on pp. 14–15);
- **LOOCV** as an additional check (p. 16).

In both cases:

- the MSE curve has a clear minimum, highlighted in the graph;
- the **best λ** is essentially the same for K-Fold and LOOCV (≈ 235,498).

Using this λ, the final Ridge model is estimated, yielding more stable and shrunk coefficients compared to OLS.

---

## 6. Lasso Regression

### 6.1 Features and K-Fold CV

Lasso Regression uses an **L1 penalty** (α = 1) and has the property of:

- **setting some coefficients exactly to zero**, thus performing automatic variable selection.

As with Ridge, the following are constructed:

- matrix X of the covariates;
- vector y of the price;
- coefficients estimated over a grid of λ values (p. 17, coefficient paths vs log(λ)).

With **K-Fold CV (K=10)**:

- the average MSE for each λ is computed (plots pp. 17–18);
- the **λ that minimises MSE** is selected (≈ 7,320.807).

The final Lasso model:

- eliminates (coefficient = 0) many variables deemed little informative;
- keeps and shrinks those with greater impact on `price`.

### 6.2 LOOCV for Lasso

The procedure is repeated with **LOOCV** (p. 19):

- the MSE vs log(λ) curve leads to a new optimal λ (≈ 3,497.073);
- with this λ, coefficients are re-estimated.

Patterns remain similar: many coefficients are shrunk to zero and the resulting model is more parsimonious.

---

## 7. Elastic Net

### 7.1 Setup

Elastic Net combines **L1 (Lasso)** and **L2 (Ridge)** penalties:

- the overall penalty is controlled by **λ**;
- the mixing is controlled by **α** (α=0 → Ridge; α=1 → Lasso).

To combine the advantages of both approaches, three values of α are considered:

- **α = 0.1**
- **α = 0.5**
- **α = 0.9**

For each α:

- coefficient paths are estimated over a grid of λ values (coefficient vs log(λ) plots, p. 21);
- **K-Fold CV (K=10)** is applied to select the optimal λ (pp. 22–24).

### 7.2 Results for Different α

- **α = 0.1**
  - Best λ ≈ 8,662.513 (p. 22)
  - Most coefficients are shrunk but few are set to zero → behaviour closer to Ridge.
- **α = 0.5**
  - Best λ ≈ 5,994.141 (p. 23)
  - Intermediate model: more non-zero coefficients than pure Lasso, but with substantial shrinkage → good compromise between selection and stabilization.
- **α = 0.9**
  - Best λ ≈ 4,668.260 (p. 24)
  - Behaviour close to Lasso: stronger selection, but still less extreme than pure Lasso.

In all cases, Elastic Net results in fewer zero coefficients than Lasso, but more regularization than Ridge.

---

## 8. Final Comparison Between Ridge, Lasso and Elastic Net

The **comparative table** on p. 25 lists side by side the coefficients estimated by the different models:

- **Lasso**:
  - sets almost all coefficients to zero, leaving only a few active variables;
  - maximum parsimony but risk of over-simplification.
- **Ridge**:
  - no coefficient is exactly zero;
  - all regressors remain in the model, but with strongly reduced magnitudes.
- **Elastic Net**:
  - intermediate approach:
    - sets only a limited number of variables to zero (in the best case, only `shared_gym`, `shared_spa`, `view_of_landmark`, `childrens_pool`);
    - keeps and shrinks the remaining coefficients;
  - mitigates both the rigidity of Lasso and the lack of selectivity of Ridge.

---

## 9. Choice of the Best Model (MSE Criterion)

Among all estimated models (OLS, models with interactions/polynomials, Ridge with CV, Lasso with CV, Elastic Net with different α), the model with the **lowest MSE** is selected (p. 26):

- the **best-performing** model in terms of mean squared error is:
  - **Elastic Net with α = 0.5** (with corresponding optimal λ from K-Fold CV).

### Operational Conclusions

- The **Elastic Net model (α = 0.5)**:
  - adequately handles **multicollinearity**;
  - keeps a reasonable number of explanatory variables, avoiding excessive exclusions;
  - achieves the **lowest MSE** among all procedures considered;
  - offers an optimal trade-off between **bias** and **variance**, making it the most suitable statistical learning model to predict property prices in Dubai for the dataset analysed.

In summary, the use of regularization techniques (particularly Elastic Net with α = 0.5) proves essential to build an accurate, stable, and interpretable predictive model in the presence of many highly correlated variables.
