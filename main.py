import pandas as pd
import itertools
import statsmodels.api as sm
from sklearn.metrics import accuracy_score
import numpy as np


# Load heart disease data
df = pd.read_csv('heart.csv')
target = 'target'  # 0 = no heart disease, 1 = yes heart disease
predictors = [col for col in df.columns if col != target]


# Multivariate Regression
def all_possible_logistic_regression(df, target, predictors):
    results = []

    for k in range(1, len(predictors) + 1):
        for subset in itertools.combinations(predictors, k):
            X = df[list(subset)]
            X = sm.add_constant(X)
            y = df[target]

            try:
                model = sm.Logit(y, X).fit(disp=0)
                aic = model.aic
                bic = model.bic
                pseudo_r2 = model.prsquared

                # Predict class labels
                y_pred_prob = model.predict(X)
                y_pred = (y_pred_prob >= 0.5).astype(int)

                # Classification error
                accuracy = accuracy_score(y, y_pred)
                classification_error = 1 - accuracy

                results.append({
                    'predictors': subset,
                    'AIC': aic,
                    'BIC': bic,
                    'Pseudo_R2': pseudo_r2,
                    'Classification_Error': classification_error
                })

            except Exception as e:
                continue

    result_df = pd.DataFrame(results)
    return result_df.sort_values(by='AIC')


results_df = all_possible_logistic_regression(df, target, predictors)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
print(results_df.head(1))


# -----------------------------------------------------------------------
# Monte Carlo to simulate the effect of Unmeasured Confounders
# Define parameters for unmeasured confounder
n_simulations = 1000

# Assumed prevalence of the unmeasured confounder (U)
prevalence_U = 0.3

log_odds_U_on_outcome = 0.7  # OR ≈ 2
log_odds_U_on_cholesterol = 0.5  # makes cholesterol more likely if U = 1

# Variable names
exposure = 'chol'
outcome = 'target'

# Step 3: Run MC bias analysis
adjusted_betas = []

for _ in range(n_simulations):
    df_sim = df.copy()

    # Simulate unmeasured confounder
    df_sim['U'] = np.random.binomial(1, prevalence_U, size=len(df_sim))

    # Inject confounding:
    # a) Adjust exposure based on U
    df_sim[exposure] += df_sim['U'] * np.random.normal(10, 2)  # exposure shift if U=1

    # b) Adjust outcome probability based on U
    logits = np.log(df_sim[outcome] + 1e-5) + df_sim['U'] * log_odds_U_on_outcome
    probs = 1 / (1 + np.exp(-logits))
    df_sim[outcome] = np.random.binomial(1, probs)

    # Fit logistic regression WITHOUT U (we pretend it's unmeasured)
    X = sm.add_constant(df_sim[[exposure]])
    y = df_sim[outcome]
    model = sm.Logit(y, X).fit(disp=0)

    # Store coefficient for the exposure
    adjusted_betas.append(model.params[exposure])

# Step 4: Analyze simulation results
adjusted_betas = np.array(adjusted_betas)
beta_mean = adjusted_betas.mean()
beta_ci_lower = np.percentile(adjusted_betas, 2.5)
beta_ci_upper = np.percentile(adjusted_betas, 97.5)

print(f"\nMonte Carlo Bias Analysis Results for '{exposure}':")
print(f"Mean beta (log-odds): {beta_mean:.3f}")
print(f"95% CI: [{beta_ci_lower:.3f}, {beta_ci_upper:.3f}]")
print(f"Approximate OR: {np.exp(beta_mean):.2f} (CI: {np.exp(beta_ci_lower):.2f} – {np.exp(beta_ci_upper):.2f})")


# Check standard regression to see impact of cholesterol
# Define predictor and target
X = sm.add_constant(df['chol'])  # adds intercept term
y = df['target']

# Fit logistic regression
model = sm.Logit(y, X).fit()

# Show summary
print(model.summary())

# Calculate and print odds ratio
odds_ratio = np.exp(model.params['chol'])
print(f"\nOdds Ratio for chol: {odds_ratio:.3f}")

