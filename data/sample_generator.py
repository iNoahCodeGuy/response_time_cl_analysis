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

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import SAMPLE_DATA_CONFIG


def generate_sample_data(
    n_leads: Optional[int] = None,
    n_weeks: int = 8,
    random_seed: int = 42
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
    random_seed : int
        Seed for reproducibility
        
    RETURNS:
    --------
    pd.DataFrame
        Generated sample data with all required columns
        
    EXAMPLE:
    --------
    >>> # Generate 8 weeks of data (default)
    >>> sample_df = generate_sample_data()
    >>> print(f"Generated {len(sample_df)} sample leads")
    
    >>> # Generate 4 weeks of data for quicker testing
    >>> sample_df = generate_sample_data(n_weeks=4)
    
    >>> # Generate 12 weeks to see longer-term trends
    >>> sample_df = generate_sample_data(n_weeks=12)
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Get config values
    config = SAMPLE_DATA_CONFIG
    
    # Calculate number of leads based on weeks
    # Default is ~1,250 leads per week (10,000 leads / 8 weeks = 1,250)
    leads_per_week = config['n_leads'] // 8  # ~1,250 leads per week
    n = n_leads if n_leads else (leads_per_week * n_weeks)
    
    # =========================================================================
    # STEP 1: Generate Lead Sources
    # =========================================================================
    # Each lead comes from one source, weighted by source popularity
    
    source_names = list(config['lead_sources'].keys())
    source_weights = [config['lead_sources'][s]['weight'] for s in source_names]
    
    lead_sources = np.random.choice(
        source_names, 
        size=n, 
        p=source_weights
    )
    
    # Get base close rate for each lead's source
    base_close_rates = np.array([
        config['lead_sources'][source]['base_close_rate'] 
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
    # Skill ranges from about 0.4 to 1.6 (mean 1.0)
    rep_skills = np.random.normal(1.0, config['rep_skill_std'], n_reps)
    rep_skills = np.clip(rep_skills, 0.4, 1.8)  # Reasonable bounds
    
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
    # STEP 4: Generate Response Times
    # =========================================================================
    # Log-normal distribution modified by rep skill
    # Better reps respond faster
    
    # Base response time parameters (in log-minutes)
    median_response_mins = config['response_time_median_mins']
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
    # STEP 5: Generate Order Outcomes
    # =========================================================================
    # Close probability is affected by:
    # 1. Lead source (base rate)
    # 2. Rep skill (multiplier)
    # 3. Response time (true effect + confounding)
    
    # True effect of response time on close rate
    # Fast responses (<15 min) have a boost, slow responses have a penalty
    response_time_effect = np.where(
        response_time_mins <= 15, 1.3,  # 30% boost for fast response
        np.where(
            response_time_mins <= 30, 1.1,  # 10% boost for medium-fast
            np.where(
                response_time_mins <= 60, 0.9,  # 10% penalty for slow
                0.7  # 30% penalty for very slow
            )
        )
    )
    
    # Rep skill multiplier on close rate
    rep_skill_effect = rep_skill_values
    
    # Combined close probability
    close_prob = base_close_rates * response_time_effect * rep_skill_effect
    
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

