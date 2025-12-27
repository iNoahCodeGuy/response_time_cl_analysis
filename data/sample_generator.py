# =============================================================================
# Sample Data Generator Module
# =============================================================================
# This module generates realistic sample data for testing and demos.
#
# WHY THIS MODULE EXISTS:
# -----------------------
# Users may want to explore the app before uploading their own data.
# Realistic sample data helps them understand what insights the app provides.
#
# DESIGN DECISIONS:
# -----------------
# The sample data is designed to demonstrate several key concepts:
# 1. Different lead sources have different base close rates
# 2. Sales reps vary in both speed and effectiveness
# 3. There's a correlation between response speed and close rate
# 4. But some of this correlation is due to confounding (rep skill)
#
# This allows users to see how the statistical controls work.
# =============================================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import sys
import os
import time
import secrets

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import SAMPLE_DATA_CONFIG


def generate_sample_data(
    n_leads: Optional[int] = None,
    n_weeks: int = 8,
    random_seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate realistic sample lead data for the Response Time Analyzer.
    
    WHY THIS MATTERS:
    -----------------
    Sample data allows users to:
    1. Test the app without uploading real data
    2. Understand what the analysis shows
    3. Learn about the statistical concepts
    
    HOW THE DATA IS STRUCTURED:
    ---------------------------
    The generated data includes realistic patterns:
    
    1. LEAD SOURCES: Different sources have different base close rates
       - Dealer Referral: 15% (high intent)
       - Website Form: 6% (browsing)
       - Phone Call: 12% (interested)
       - Walk-in: 18% (very high intent)
       - Third Party Lead: 4% (low quality)
    
    2. SALES REPS: 12 reps with varying "skill levels"
       - Skill affects both response speed AND close rate
       - This creates realistic confounding to demonstrate
    
    3. RESPONSE TIME: Log-normal distribution
       - Median around 25 minutes
       - Long tail (some very slow responses)
       - Modified by rep skill (better reps are faster)
    
    4. CLOSE RATE: Affected by:
       - Lead source base rate
       - Sales rep skill
       - Response time (true causal effect + confounding)
    
    PARAMETERS:
    -----------
    n_leads : int, optional
        Number of leads to generate (default from settings).
        If not provided, defaults to ~1,250 leads per week.
    n_weeks : int
        Number of weeks of data to generate (default: 8).
        More weeks = more data for week-over-week analysis.
        Recommended: 4-12 weeks for best results.
    random_seed : int, optional
        Seed for reproducibility. If None (default), data will be different
        each time the function is called. Set to an integer for reproducible
        results (useful for testing).
        
    RETURNS:
    --------
    pd.DataFrame
        Generated sample data with all required columns
        
    EXAMPLE:
    --------
    >>> # Generate 8 weeks of data (default, different each time)
    >>> sample_df = generate_sample_data()
    >>> print(f"Generated {len(sample_df)} sample leads")
    
    >>> # Generate 4 weeks of data for quicker testing
    >>> sample_df = generate_sample_data(n_weeks=4)
    
    >>> # Generate reproducible data with a fixed seed
    >>> sample_df = generate_sample_data(n_weeks=8, random_seed=42)
    """
    # Set random seed only if provided (for reproducibility)
    # If None, use a combination of high-resolution time and random bits to ensure uniqueness
    if random_seed is not None:
        np.random.seed(random_seed)
    else:
        # Combine high-resolution time with random bits for true uniqueness
        # time.time_ns() gives nanosecond precision (Python 3.7+)
        # secrets.randbits(32) adds 32 random bits to prevent collisions
        # This ensures different data every time, even if called in quick succession
        try:
            # Use nanoseconds if available (Python 3.7+)
            time_component = time.time_ns()
        except AttributeError:
            # Fallback for older Python versions: combine time with process ID
            time_component = int(time.time() * 1_000_000_000) + os.getpid()
        
        random_component = secrets.randbits(32)
        combined_seed = (time_component ^ random_component) % (2**32)
        np.random.seed(combined_seed)
    
    # Get config values
    config = SAMPLE_DATA_CONFIG
    
    # Calculate number of leads based on weeks
    # Default is ~1,250 leads per week (10,000 leads / 8 weeks = 1,250)
    leads_per_week = config['n_leads'] // 8  # ~1,250 leads per week
    n = n_leads if n_leads else (leads_per_week * n_weeks)
    
    # =========================================================================
    # STEP 1: Generate Lead Sources with VARIED base close rates
    # =========================================================================
    # Each lead comes from one source, weighted by source popularity
    # Vary base close rates slightly to create more realistic variation
    
    source_names = list(config['lead_sources'].keys())
    source_weights = [config['lead_sources'][s]['weight'] for s in source_names]
    
    lead_sources = np.random.choice(
        source_names, 
        size=n, 
        p=source_weights
    )
    
    # Vary base close rates by ±20% to add realism (some periods have better/worse sources)
    base_close_rate_variation = np.random.uniform(0.85, 1.15, len(source_names))
    source_base_rates_varied = {
        source: config['lead_sources'][source]['base_close_rate'] * base_close_rate_variation[i]
        for i, source in enumerate(source_names)
    }
    # Ensure rates stay within reasonable bounds (2% to 25%)
    for source in source_base_rates_varied:
        source_base_rates_varied[source] = np.clip(source_base_rates_varied[source], 0.02, 0.25)
    
    # Get base close rate for each lead's source (with variation)
    base_close_rates = np.array([
        source_base_rates_varied[source] 
        for source in lead_sources
    ])
    
    # =========================================================================
    # STEP 2: Generate Sales Reps with Skill Levels
    # =========================================================================
    # Each rep has a "skill" that affects both speed and close rate
    # This creates realistic confounding
    
    n_reps = config['n_reps']
    rep_names = [f"Rep_{i+1:02d}" for i in range(n_reps)]
    
    # Skill is normally distributed (some reps are better than others)
    # Vary the skill variation to reduce confounding in some datasets
    # Sometimes reps are very similar (low confounding), sometimes very different (high confounding)
    skill_std_multiplier = np.random.uniform(0.5, 1.2)  # Reduce skill variation by up to 50%
    rep_skill_std = config['rep_skill_std'] * skill_std_multiplier
    rep_skills = np.random.normal(1.0, rep_skill_std, n_reps)
    rep_skills = np.clip(rep_skills, 0.5, 1.5)  # Tighter bounds to reduce extreme variation
    
    # Assign leads to reps (roughly equal distribution with some variation)
    rep_assignments = np.random.choice(range(n_reps), size=n)
    sales_reps = np.array([rep_names[i] for i in rep_assignments])
    rep_skill_values = rep_skills[rep_assignments]
    
    # =========================================================================
    # STEP 3: Generate Lead Arrival Times
    # =========================================================================
    # Spread over the specified number of weeks, with realistic daily patterns
    
    end_date = datetime.now()
    start_date = end_date - timedelta(weeks=n_weeks)
    
    # Random dates within the range
    date_range_seconds = (end_date - start_date).total_seconds()
    random_seconds = np.random.uniform(0, date_range_seconds, n)
    lead_times = [start_date + timedelta(seconds=s) for s in random_seconds]
    
    # Adjust for business hours (more leads during 9am-6pm)
    # This is a simplification; real data would have more complex patterns
    lead_times = pd.Series(lead_times)
    
    # =========================================================================
    # STEP 4: Generate Response Times with VARIED distribution
    # =========================================================================
    # Log-normal distribution modified by rep skill
    # Better reps respond faster
    # Vary the median response time slightly to add realism
    
    # Base response time parameters (in log-minutes)
    # Vary median by ±30% (some periods are busier, slower responses)
    median_variation = np.random.uniform(0.7, 1.3)
    median_response_mins = config['response_time_median_mins'] * median_variation
    log_std = config['response_time_std_log']
    
    # Generate base response times (log-normal)
    log_response_times = np.random.normal(
        np.log(median_response_mins), 
        log_std, 
        n
    )
    
    # Modify by rep skill (higher skill = faster response)
    # Skill of 1.0 means normal speed, 1.5 means 1.5x faster
    skill_modifier = 1.0 / rep_skill_values
    log_response_times = log_response_times + np.log(skill_modifier)
    
    # Convert back from log scale and ensure minimum 1 minute
    response_time_mins = np.exp(log_response_times)
    response_time_mins = np.maximum(response_time_mins, 1)
    
    # Cap at reasonable maximum (8 hours)
    response_time_mins = np.minimum(response_time_mins, 480)
    
    # Calculate first response times
    first_response_times = [
        lt + timedelta(minutes=rt) 
        for lt, rt in zip(lead_times, response_time_mins)
    ]
    
    # =========================================================================
    # STEP 5: Generate Order Outcomes with VARIED response time effects
    # =========================================================================
    # Close probability is affected by:
    # 1. Lead source (base rate)
    # 2. Rep skill (multiplier)
    # 3. Response time (true effect + confounding)
    
    # Vary the response time effect strength dramatically to create realistic variation
    # Sometimes the effect is strong (high significance), sometimes weak/none (low significance)
    # Use uniform distribution with wide range to get true variety
    # Allow 0.0 to create datasets with essentially no response time effect
    effect_strength = np.random.uniform(0.0, 1.5)  # Range 0.0 (no effect) to 1.5 (strong effect)
    
    # True effect of response time on close rate (with wide variation)
    # Fast responses (<15 min) have a boost, slow responses have a penalty
    # When effect_strength is 0, there's no effect (all multipliers = 1.0)
    fast_boost = 1.0 + (0.25 * effect_strength)  # Ranges from 1.0 (no effect) to 1.375 (37.5% boost)
    medium_boost = 1.0 + (0.08 * effect_strength)  # Ranges from 1.0 to 1.12
    slow_penalty = 1.0 - (0.08 * effect_strength)  # Ranges from 1.0 to 0.88
    very_slow_penalty = 1.0 - (0.25 * effect_strength)  # Ranges from 1.0 to 0.625
    
    response_time_effect = np.where(
        response_time_mins <= 15, fast_boost,
        np.where(
            response_time_mins <= 30, medium_boost,
            np.where(
                response_time_mins <= 60, slow_penalty,
                very_slow_penalty
            )
        )
    )
    
    # Vary rep skill effect strength to reduce confounding in some datasets
    # Sometimes rep skill has strong effect (creates confounding), sometimes weaker
    # IMPORTANT: When effect_strength is very low, reduce rep skill effect to prevent
    # confounding from creating spurious significance when there's no true response time effect
    if effect_strength < 0.2:
        # Very low response time effect - reduce rep skill variation to avoid confounding
        rep_skill_strength = np.random.uniform(0.3, 0.6)  # Much weaker rep skill effect
    else:
        rep_skill_strength = np.random.uniform(0.7, 1.0)  # Normal rep skill impact
    
    rep_skill_effect = 1.0 + (rep_skill_values - 1.0) * rep_skill_strength
    # Add per-lead variation
    rep_skill_effect = rep_skill_effect * np.random.uniform(0.92, 1.08, n)
    
    # Combined close probability
    close_prob = base_close_rates * response_time_effect * rep_skill_effect
    
    # Add adaptive random noise to make relationships less deterministic
    # When effect_strength is low, use MUCH stronger noise to ensure non-significant results
    # This simulates other unmeasured factors affecting outcomes
    # Stronger noise when effects are weak helps create realistic variation in p-values
    if effect_strength < 0.2:
        # Very weak effect - use very strong noise (60% to 140%) to mask any weak associations
        noise_range = (0.60, 1.40)
    elif effect_strength < 0.5:
        # Weak effect - use moderate-strong noise (70% to 130%)
        noise_range = (0.70, 1.30)
    elif effect_strength < 1.0:
        # Moderate effect - use moderate noise (75% to 125%)
        noise_range = (0.75, 1.25)
    else:
        # Strong effect - use normal noise (80% to 120%)
        noise_range = (0.80, 1.20)
    
    noise_factor = np.random.uniform(noise_range[0], noise_range[1], n)
    close_prob = close_prob * noise_factor
    
    # Ensure probabilities are in valid range [0, 1]
    close_prob = np.clip(close_prob, 0, 0.95)
    
    # Generate actual outcomes
    ordered = np.random.binomial(1, close_prob)
    
    # =========================================================================
    # STEP 6: Assemble DataFrame
    # =========================================================================
    
    df = pd.DataFrame({
        'lead_id': [f"LEAD_{i+1:06d}" for i in range(n)],
        'lead_time': lead_times,
        'first_response_time': first_response_times,
        'response_time_mins': response_time_mins.round(1),
        'lead_source': lead_sources,
        'sales_rep': sales_reps,
        'ordered': ordered
    })
    
    # Sort by lead time
    df = df.sort_values('lead_time').reset_index(drop=True)
    
    return df


def get_sample_data_summary(df: pd.DataFrame) -> dict:
    """
    Get a summary of the sample data for display to users.
    
    WHY THIS MATTERS:
    -----------------
    Helps users understand the sample data before running analysis.
    Shows key statistics that set expectations for results.
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        The sample data
        
    RETURNS:
    --------
    dict
        Summary statistics about the sample data
    """
    summary = {
        'total_leads': len(df),
        'date_range': {
            'start': df['lead_time'].min().strftime('%Y-%m-%d'),
            'end': df['lead_time'].max().strftime('%Y-%m-%d')
        },
        'lead_sources': df['lead_source'].value_counts().to_dict(),
        'sales_reps': {
            'count': df['sales_rep'].nunique(),
            'list': df['sales_rep'].unique().tolist()
        },
        'response_time': {
            'median_mins': df['response_time_mins'].median(),
            'mean_mins': df['response_time_mins'].mean(),
            'min_mins': df['response_time_mins'].min(),
            'max_mins': df['response_time_mins'].max()
        },
        'overall_close_rate': df['ordered'].mean(),
        'total_orders': df['ordered'].sum()
    }
    
    return summary


# =============================================================================
# For testing this module directly
# =============================================================================
if __name__ == "__main__":
    print("Generating sample data...")
    df = generate_sample_data(n_leads=1000)
    
    print(f"\nGenerated {len(df)} leads")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    print(f"\nSummary:")
    summary = get_sample_data_summary(df)
    print(f"  Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"  Lead sources: {summary['lead_sources']}")
    print(f"  Number of reps: {summary['sales_reps']['count']}")
    print(f"  Median response time: {summary['response_time']['median_mins']:.1f} mins")
    print(f"  Overall close rate: {summary['overall_close_rate']:.1%}")

