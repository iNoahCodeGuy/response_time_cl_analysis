# =============================================================================
# LaTeX Formulas Module
# =============================================================================
# This module contains LaTeX formulas for all statistical tests.
#
# WHY THIS MODULE EXISTS:
# -----------------------
# Some users want to see the math behind the statistics.
# LaTeX renders beautifully in Streamlit using st.latex().
#
# USAGE:
# ------
# from explanations.formulas import get_formula
# formula = get_formula('chi_square')
# st.latex(formula)
# =============================================================================


LATEX_FORMULAS = {
    
    # =========================================================================
    # CHI-SQUARE TEST
    # =========================================================================
    'chi_square': {
        'name': 'Chi-Square Test Statistic',
        'formula': r"\chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}",
        'components': {
            r"O_i": "Observed count in cell i",
            r"E_i": "Expected count in cell i (if no relationship)",
            r"k": "Number of cells in the table"
        },
        'components_explained': {
            'O (Observed)': 'What actually happened. For example, 419 leads responded to in 0-15 min became customers.',
            'E (Expected)': 'What we\'d EXPECT if response time had NO effect. If the overall close rate is 9.8%, we\'d expect 9.8% of each bucket to close.',
            'Σ (Sum)': 'Add up the values for all buckets',
            '(O-E)²': 'Square the difference so that "more than expected" and "less than expected" both count as surprising',
            '÷ E': 'Divide by expected to normalize — a 50-order difference in a bucket of 1000 matters less than in a bucket of 100'
        },
        'step_by_step': """
**Step 1:** For each bucket, calculate: How many sales ACTUALLY happened (O)?
**Step 2:** Calculate: How many would we EXPECT if response time didn't matter (E)?
**Step 3:** Find the difference: (O - E)
**Step 4:** Square it: (O - E)²  — this makes all differences positive
**Step 5:** Divide by expected: (O - E)² / E  — this accounts for bucket size
**Step 6:** Add up all the buckets to get χ²

The bigger χ², the more "surprising" our data is if response time truly didn't matter.
""",
        'intuition': """
**The Big Idea:** If response time doesn't matter, every bucket should have roughly the same close rate.
We check: "How different are the actual results from what we'd see if response time had zero effect?"
Large differences → large χ² → "Response time probably DOES matter!"
""",
        'example': """
**Real Example:**
- 0-15 min bucket: Actually got 419 sales. If no effect, we'd expect about 314.
- That's 105 MORE sales than expected!
- (419 - 314)² / 314 = 35.1 contribution to χ²

Add up all buckets → χ² = 77.72 (very high!)
This means reality is VERY different from "no effect" — response time matters.
"""
    },
    
    # =========================================================================
    # Z-TEST FOR PROPORTIONS
    # =========================================================================
    'z_test': {
        'name': 'Z-Test for Two Proportions',
        'formula': r"z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_1} + \frac{1}{n_2}\right)}}",
        'components': {
            r"\hat{p}_1, \hat{p}_2": "Sample proportions (close rates)",
            r"\hat{p}": "Pooled proportion (combined close rate)",
            r"n_1, n_2": "Sample sizes"
        },
        'intuition': """
We measure how many "standard errors" apart the two proportions are.
If they're more than ~2 standard errors apart, the difference is significant.
"""
    },
    
    # =========================================================================
    # WILSON CONFIDENCE INTERVAL
    # =========================================================================
    'wilson_ci': {
        'name': 'Wilson Score Confidence Interval',
        'formula': r"\frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}",
        'components': {
            r"\hat{p}": "Sample proportion (close rate)",
            r"n": "Sample size",
            r"z": "Z-score for desired confidence level (1.96 for 95%)"
        },
        'components_explained': {
            'p̂ (p-hat)': 'The close rate we measured from our sample — e.g., 13.0%',
            'n': 'How many leads we have in this bucket — e.g., 3,217',
            'z': 'A number that controls confidence level. 1.96 gives 95% confidence.',
            '±': 'This creates the lower and upper bounds of the range'
        },
        'intuition': """
**The Big Idea:**
Our measured close rate (13.0%) is our best guess, but it's based on a sample.
The TRUE close rate might be a bit higher or lower.

The confidence interval says: "We're 95% confident the true rate is between X% and Y%"

**Why this matters:**
- Narrow range (12.5% - 13.5%) = Very confident, have lots of data
- Wide range (10% - 16%) = Less confident, need more data
- If two buckets' ranges don't overlap = Probably a real difference
""",
        'example': """
**Your 0-15 min bucket:**
- Measured close rate: 13.0%
- Sample size: 3,217 leads
- 95% Confidence Interval: 11.9% - 14.2%

**What this means:**
"We measured 13.0%, but the true close rate for fast responders 
is probably somewhere between 11.9% and 14.2%."

The range is fairly narrow because we have lots of data (3,217 leads).
"""
    },
    
    # =========================================================================
    # LOGISTIC REGRESSION
    # =========================================================================
    'logistic_regression': {
        'name': 'Logistic Regression Model',
        'formula': r"\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_k x_k",
        'components': {
            r"p": "Probability of ordering",
            r"\frac{p}{1-p}": "Odds of ordering",
            r"\beta_0": "Intercept (baseline log-odds)",
            r"\beta_i": "Effect of predictor i on log-odds"
        },
        'components_explained': {
            'p': 'The chance a lead becomes a customer (between 0 and 1). Example: 0.13 means 13% chance.',
            'p/(1-p)': 'The "odds" — another way to express chance. If p=0.13, odds = 0.13/0.87 ≈ 0.15, or "about 1 in 7".',
            'log(...)': 'A mathematical transformation that makes the equation work. Don\'t worry about this part.',
            'β₀ (beta-zero)': 'The starting point — the log-odds for our reference group (slow responders).',
            'β₁ (beta-one)': 'How much being in a different response bucket changes the log-odds.',
            'x₁, x₂, ...': 'The input variables — like response time bucket and lead source.'
        },
        'step_by_step': """
**What regression does (in plain English):**

1. Start with the baseline (slow responders from a typical lead source)
2. Add or subtract based on response speed — e.g., fast responders get a bonus
3. Add or subtract based on lead source — e.g., referrals get a bonus
4. Convert back to a probability

**The key output:** For each factor (like response speed), we get an "odds ratio" 
that tells us: "How many times more likely is this group to close?"
""",
        'intuition': """
**Why we need this:**
Simple comparison might show "fast responders close more" — but what if that's 
because referrals (high-quality leads) happen to get faster responses?

Regression lets us ask: "Among ONLY referral leads, do fast responders still win?"
And: "Among ONLY website leads, do fast responders still win?"

If yes, speed genuinely matters. If no, it was just an illusion caused by lead mix.
""",
        'example': """
**Real Example:**
Without controlling for lead source:
- Fast responders: 13% close rate
- Slow responders: 6% close rate
- Difference: 7 points!

After controlling for lead source (regression):
- Fast responders: Still ~2x better odds
- Lead source didn't explain away the pattern
- Conclusion: Speed genuinely helps!
"""
    },
    
    # =========================================================================
    # ODDS RATIO
    # =========================================================================
    'odds_ratio': {
        'name': 'Odds Ratio',
        'formula': r"OR = \frac{p_1 / (1-p_1)}{p_2 / (1-p_2)} = e^{\beta}",
        'components': {
            r"p_1, p_2": "Probabilities in each group",
            r"\beta": "Regression coefficient",
            r"e^\beta": "Odds ratio from regression"
        },
        'components_explained': {
            'p₁': 'Close rate for the faster responders (e.g., 0.13 = 13%)',
            'p₂': 'Close rate for the slower responders (e.g., 0.063 = 6.3%)',
            'p/(1-p)': 'Converts probability to odds. 13% → 13/87 ≈ 0.15',
            'OR': 'Odds Ratio — the multiplier comparing the two groups',
            'e^β': 'The mathematical way we get OR from regression. e ≈ 2.718'
        },
        'intuition': """
**Think of it as a "success multiplier":**

| Odds Ratio | What it means | Real example |
|:-----------|:--------------|:-------------|
| OR = 1.0 | Same odds as comparison group | No advantage |
| OR = 1.5 | 50% better odds | If they get 10 sales, you get 15 |
| OR = 2.0 | Twice the odds | If they get 10 sales, you get 20 |
| OR = 0.5 | Half the odds | If they get 10 sales, you get 5 |

**Why "odds" instead of probabilities?**
Odds work better mathematically and are easier to combine across different situations.
""",
        'example': """
**Your Data:**
- Fast responders (0-15 min): 13.0% close rate
- Slow responders (60+ min): 6.3% close rate

Odds for fast: 0.13 / 0.87 = 0.149
Odds for slow: 0.063 / 0.937 = 0.067

Odds Ratio = 0.149 / 0.067 = **2.2x**

Fast responders have 2.2 times the odds of closing!
"""
    },
    
    # =========================================================================
    # CONFIDENCE INTERVAL FOR ODDS RATIO
    # =========================================================================
    'or_ci': {
        'name': 'Confidence Interval for Odds Ratio',
        'formula': r"95\% \text{ CI} = \left[ e^{\beta - 1.96 \cdot SE(\beta)}, e^{\beta + 1.96 \cdot SE(\beta)} \right]",
        'components': {
            r"\beta": "Regression coefficient",
            r"SE(\beta)": "Standard error of the coefficient",
            "1.96": "Z-score for 95% confidence"
        },
        'intuition': """
This gives a range where the true odds ratio likely falls.
If the CI includes 1, the effect is not statistically significant.
"""
    },
    
    # =========================================================================
    # CLOSE RATE CALCULATION
    # =========================================================================
    'close_rate': {
        'name': 'Close Rate',
        'formula': r"\text{Close Rate} = \frac{\text{Number of Orders}}{\text{Number of Leads}} = \frac{\sum_{i=1}^{n} \mathbb{1}[\text{ordered}_i]}{n}",
        'components': {
            r"\mathbb{1}[\text{ordered}_i]": "1 if lead i ordered, 0 otherwise",
            r"n": "Total number of leads"
        },
        'intuition': """
Simply the fraction of leads that converted to orders.
Also called "conversion rate" or "win rate".
"""
    },
    
    # =========================================================================
    # STANDARD ERROR OF PROPORTION
    # =========================================================================
    'se_proportion': {
        'name': 'Standard Error of a Proportion',
        'formula': r"SE(\hat{p}) = \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}",
        'components': {
            r"\hat{p}": "Sample proportion (close rate)",
            r"n": "Sample size"
        },
        'intuition': """
This measures how much we'd expect the proportion to vary 
if we took different random samples of the same size.
Larger n → smaller SE → more precise estimate.
"""
    },
    
    # =========================================================================
    # COHEN'S H (EFFECT SIZE)
    # =========================================================================
    'cohens_h': {
        'name': "Cohen's h (Effect Size for Proportions)",
        'formula': r"h = 2 \arcsin(\sqrt{p_1}) - 2 \arcsin(\sqrt{p_2})",
        'components': {
            r"p_1, p_2": "The two proportions being compared",
            r"\arcsin": "Arcsine function"
        },
        'interpretation': """
|h| < 0.2: Small effect
|h| = 0.2-0.5: Medium effect  
|h| > 0.5: Large effect
""",
        'intuition': """
Unlike the raw difference, Cohen's h accounts for the fact that 
a 5% vs 10% difference is more meaningful than 45% vs 50%.
"""
    },
    
    # =========================================================================
    # MIXED EFFECTS MODEL
    # =========================================================================
    'mixed_effects': {
        'name': 'Mixed Effects Model',
        'formula': r"y_{ij} = \beta_0 + \beta_1 x_{ij} + u_j + \epsilon_{ij}",
        'components': {
            r"y_{ij}": "Outcome for lead i with rep j",
            r"x_{ij}": "Response time bucket",
            r"\beta": "Fixed effects (what we estimate)",
            r"u_j": "Random effect for rep j ~ N(0, σ²ᵤ)",
            r"\epsilon_{ij}": "Residual error"
        },
        'intuition': """
We allow each rep to have their own baseline (random intercept).
The fixed effect β₁ captures the within-rep effect of response time.
"""
    },
    
    # =========================================================================
    # INTRACLASS CORRELATION
    # =========================================================================
    'icc': {
        'name': 'Intraclass Correlation Coefficient (ICC)',
        'formula': r"ICC = \frac{\sigma^2_u}{\sigma^2_u + \sigma^2_\epsilon}",
        'components': {
            r"\sigma^2_u": "Between-group (rep) variance",
            r"\sigma^2_\epsilon": "Within-group (residual) variance"
        },
        'intuition': """
ICC tells us what fraction of the variation is due to rep-level differences.
ICC = 0.10 means 10% of variation is between reps.
High ICC → controlling for rep is important.
"""
    }
}


def get_formula(formula_type: str) -> dict:
    """
    Get the LaTeX formula and explanation for a given formula type.
    
    PARAMETERS:
    -----------
    formula_type : str
        One of the keys in LATEX_FORMULAS
        
    RETURNS:
    --------
    dict
        Contains 'formula', 'components', and 'intuition'
        
    EXAMPLE:
    --------
    >>> chi_sq = get_formula('chi_square')
    >>> st.latex(chi_sq['formula'])
    >>> st.write(chi_sq['intuition'])
    """
    if formula_type not in LATEX_FORMULAS:
        return {
            'name': formula_type,
            'formula': r"\text{Formula not available}",
            'components': {},
            'intuition': ""
        }
    
    return LATEX_FORMULAS[formula_type]


def render_formula_with_explanation(formula_type: str) -> str:
    """
    Create a formatted string with the formula and component explanations.
    
    Useful for displaying in Streamlit expanders.
    
    PARAMETERS:
    -----------
    formula_type : str
        One of the keys in LATEX_FORMULAS
        
    RETURNS:
    --------
    str
        Markdown-formatted explanation
    """
    if formula_type not in LATEX_FORMULAS:
        return f"No formula available for {formula_type}"
    
    f = LATEX_FORMULAS[formula_type]
    
    output = f"### {f['name']}\n\n"
    output += "**Formula:**\n\n"
    output += f"$${f['formula']}$$\n\n"
    
    if f.get('components'):
        output += "**Where:**\n"
        for symbol, meaning in f['components'].items():
            output += f"- ${symbol}$ = {meaning}\n"
        output += "\n"
    
    if f.get('intuition'):
        output += "**Intuition:**\n"
        output += f"{f['intuition']}\n"
    
    return output

