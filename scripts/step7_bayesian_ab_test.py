"""
STEP 7: Bayesian A/B Test
Instead of a p-value, directly answer: "What is the probability that ad
truly beats psa?" and "What do we risk by picking the wrong one?"
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid")

# Values carried over from Step 3
n_ad, n_psa = 564577, 23524
conv_ad, conv_psa = 14423, 420

# =========================================================
# 1. Set up priors and compute posteriors (Beta-Binomial conjugacy)
# =========================================================
# Uninformative prior: Beta(1,1) = Uniform(0,1) -- we let the data speak
alpha_prior, beta_prior = 1, 1

alpha_ad_post = alpha_prior + conv_ad
beta_ad_post = beta_prior + (n_ad - conv_ad)

alpha_psa_post = alpha_prior + conv_psa
beta_psa_post = beta_prior + (n_psa - conv_psa)

print("=== Posterior distributions ===")
print(f"Ad:  Beta({alpha_ad_post}, {beta_ad_post})")
print(f"PSA: Beta({alpha_psa_post}, {beta_psa_post})")

posterior_ad = stats.beta(alpha_ad_post, beta_ad_post)
posterior_psa = stats.beta(alpha_psa_post, beta_psa_post)

print(f"\nPosterior mean (ad):  {posterior_ad.mean()*100:.4f}%")
print(f"Posterior mean (psa): {posterior_psa.mean()*100:.4f}%")

ci_ad = posterior_ad.ppf([0.025, 0.975])
ci_psa = posterior_psa.ppf([0.025, 0.975])
print(f"95% credible interval (ad):  [{ci_ad[0]*100:.4f}%, {ci_ad[1]*100:.4f}%]")
print(f"95% credible interval (psa): [{ci_psa[0]*100:.4f}%, {ci_psa[1]*100:.4f}%]")

# =========================================================
# 2. Monte Carlo simulation from the posteriors
# =========================================================
np.random.seed(42)
n_sim = 1_000_000
samples_ad = np.random.beta(alpha_ad_post, beta_ad_post, size=n_sim)
samples_psa = np.random.beta(alpha_psa_post, beta_psa_post, size=n_sim)

prob_ad_better = np.mean(samples_ad > samples_psa)
print(f"\n=== P(ad truly beats psa) ===")
print(f"P(p_ad > p_psa) = {prob_ad_better:.6f}  ({prob_ad_better*100:.4f}%)")

# =========================================================
# 3. Relative uplift distribution
# =========================================================
relative_uplift = (samples_ad - samples_psa) / samples_psa * 100
uplift_mean = relative_uplift.mean()
uplift_ci = np.percentile(relative_uplift, [2.5, 97.5])
print(f"\n=== Relative uplift distribution ===")
print(f"Mean relative uplift: {uplift_mean:.2f}%")
print(f"95% credible interval: [{uplift_ci[0]:.2f}%, {uplift_ci[1]:.2f}%]")

# =========================================================
# 4. Expected loss (decision-theoretic risk of each choice)
# =========================================================
loss_if_choose_ad = np.mean(np.maximum(samples_psa - samples_ad, 0))
loss_if_choose_psa = np.mean(np.maximum(samples_ad - samples_psa, 0))
print(f"\n=== Expected loss ===")
print(f"Expected loss if we choose 'ad' (and we're wrong):  {loss_if_choose_ad*100:.6f} pp")
print(f"Expected loss if we choose 'psa' (and we're wrong): {loss_if_choose_psa*100:.6f} pp")

# =========================================================
# 5. Plot: posterior distributions overlaid
# =========================================================
x = np.linspace(0.01, 0.04, 2000)
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(x*100, posterior_ad.pdf(x), label="Ad posterior", color="#4C72B0", linewidth=2)
ax.fill_between(x*100, posterior_ad.pdf(x), alpha=0.3, color="#4C72B0")
ax.plot(x*100, posterior_psa.pdf(x), label="PSA posterior", color="#DD8452", linewidth=2)
ax.fill_between(x*100, posterior_psa.pdf(x), alpha=0.3, color="#DD8452")
ax.set_xlabel("Conversion Rate (%)")
ax.set_ylabel("Probability Density")
ax.set_title("Bayesian Posterior Distributions: Ad vs PSA Conversion Rate")
ax.legend()
plt.tight_layout()
plt.savefig("plot15_bayesian_posteriors.png", dpi=120)
plt.close()

# =========================================================
# 6. Plot: relative uplift distribution
# =========================================================
fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(relative_uplift, bins=100, ax=ax, color="#55A868", stat="density")
ax.axvline(0, color="red", linestyle="--", label="No difference (0%)")
ax.axvline(uplift_mean, color="black", linestyle="-", label=f"Mean uplift ({uplift_mean:.1f}%)")
ax.set_xlabel("Relative Uplift of Ad over PSA (%)")
ax.set_ylabel("Density")
ax.set_title("Distribution of Simulated Relative Uplift")
ax.legend()
plt.tight_layout()
plt.savefig("plot16_relative_uplift.png", dpi=120)
plt.close()

# =========================================================
# 7. Verdict
# =========================================================
print(f"\n=== VERDICT ===")
print(f"P(ad beats psa) = {prob_ad_better*100:.2f}% -- essentially certain.")
print(f"Expected loss from wrongly choosing psa ({loss_if_choose_psa*100:.4f}pp) is far higher")
print(f"than the expected loss from wrongly choosing ad ({loss_if_choose_ad*100:.6f}pp).")
print("Recommendation: roll out 'ad'. The risk of being wrong is negligible either way,")
print("but choosing 'ad' is the safer, higher-expected-value decision.")

print("\nPlots saved.")
