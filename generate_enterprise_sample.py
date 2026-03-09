#!/usr/bin/env python3
"""
Generate Enterprise-Scale Sample Data
======================================
Creates a realistic dataset representing 6 months of lead data
for a large multi-location enterprise (e.g., auto dealer group).

Volume: ~75,000 leads across 24 weeks, 35 reps, 8 lead sources, 5 regions.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis.preprocessing import create_response_buckets


def generate_enterprise_data(n_leads=75000, n_weeks=24, random_seed=42):
    np.random.seed(random_seed)

    # =========================================================================
    # ENTERPRISE CONFIGURATION
    # =========================================================================

    lead_sources = {
        'Website Form':      {'base_rate': 0.065, 'weight': 0.28, 'speed': 1.0},
        'Phone Call':         {'base_rate': 0.120, 'weight': 0.18, 'speed': 0.7},
        'Walk-in':            {'base_rate': 0.195, 'weight': 0.10, 'speed': 0.4},
        'Dealer Referral':    {'base_rate': 0.155, 'weight': 0.08, 'speed': 0.9},
        'Third Party Lead':   {'base_rate': 0.042, 'weight': 0.20, 'speed': 1.3},
        'Social Media Ad':    {'base_rate': 0.048, 'weight': 0.08, 'speed': 1.1},
        'Email Campaign':     {'base_rate': 0.072, 'weight': 0.05, 'speed': 1.0},
        'Chat Widget':        {'base_rate': 0.088, 'weight': 0.03, 'speed': 0.5},
    }

    regions = {
        'Northeast': {'weight': 0.22, 'rate_mult': 1.05, 'speed_mult': 0.95},
        'Southeast': {'weight': 0.25, 'rate_mult': 0.95, 'speed_mult': 1.05},
        'Midwest':   {'weight': 0.20, 'rate_mult': 1.00, 'speed_mult': 1.00},
        'Southwest': {'weight': 0.18, 'rate_mult': 0.98, 'speed_mult': 1.10},
        'West':      {'weight': 0.15, 'rate_mult': 1.08, 'speed_mult': 0.90},
    }

    n_reps = 35
    rep_names = [f"Rep_{i+1:02d}" for i in range(n_reps)]
    rep_skills = np.clip(np.random.normal(1.0, 0.25, n_reps), 0.55, 1.45)

    # Assign reps to regions
    region_names = list(regions.keys())
    region_weights_list = [regions[r]['weight'] for r in region_names]
    rep_regions = np.random.choice(region_names, size=n_reps, p=region_weights_list)

    # =========================================================================
    # GENERATE LEADS
    # =========================================================================

    source_names = list(lead_sources.keys())
    source_weights = [lead_sources[s]['weight'] for s in source_names]

    # Lead sources
    sources = np.random.choice(source_names, size=n_leads, p=source_weights)

    # Regions (weighted)
    lead_regions = np.random.choice(
        region_names, size=n_leads,
        p=region_weights_list
    )

    # Sales reps (assigned within region with some cross-region)
    rep_assignments = np.random.choice(range(n_reps), size=n_leads)
    sales_reps = np.array([rep_names[i] for i in rep_assignments])
    skill_values = rep_skills[rep_assignments]

    # =========================================================================
    # LEAD ARRIVAL TIMES (business-hours weighted)
    # =========================================================================
    end_date = datetime(2025, 12, 31, 18, 0, 0)
    start_date = end_date - timedelta(weeks=n_weeks)
    date_range_seconds = (end_date - start_date).total_seconds()

    random_seconds = np.random.uniform(0, date_range_seconds, n_leads)
    lead_times = pd.Series([start_date + timedelta(seconds=s) for s in random_seconds])

    # Add weekly seasonality: more leads mid-week
    day_of_week = lead_times.dt.dayofweek
    weekday_boost = np.where(day_of_week < 5, 1.0, 0.4)  # fewer weekend leads
    keep_mask = np.random.random(n_leads) < weekday_boost
    # Re-sample weekend leads to weekdays
    for i in range(n_leads):
        if not keep_mask[i]:
            # Push to a random weekday
            offset = np.random.randint(1, 4)
            lead_times.iloc[i] = lead_times.iloc[i] + timedelta(days=offset)

    # =========================================================================
    # RESPONSE TIMES (log-normal, modified by source, rep skill, region)
    # =========================================================================
    median_response = 28  # minutes
    log_std = 1.15

    log_response = np.random.normal(np.log(median_response), log_std, n_leads)

    # Source speed modifier
    source_speed = np.array([lead_sources[s]['speed'] for s in sources])
    log_response += np.log(source_speed)

    # Rep skill modifier (skilled reps respond faster)
    log_response -= np.log(skill_values) * 0.8

    # Region modifier
    region_speed = np.array([regions[r]['speed_mult'] for r in lead_regions])
    log_response += np.log(region_speed) * 0.3

    # Add time-of-day effect: leads arriving after hours get slower responses
    hour = lead_times.dt.hour
    after_hours_penalty = np.where(
        (hour < 8) | (hour >= 18), np.random.uniform(0.3, 0.8, n_leads), 0
    )
    log_response += after_hours_penalty

    response_time_mins = np.clip(np.exp(log_response), 1.0, 480.0)

    first_response_times = [
        lt + timedelta(minutes=float(rt))
        for lt, rt in zip(lead_times, response_time_mins)
    ]

    # =========================================================================
    # CLOSE OUTCOMES
    # =========================================================================
    # Base rate from source
    base_rates = np.array([lead_sources[s]['base_rate'] for s in sources])

    # Region modifier
    region_rate_mult = np.array([regions[r]['rate_mult'] for r in lead_regions])
    close_prob = base_rates * region_rate_mult

    # Rep skill effect
    close_prob *= (1.0 + (skill_values - 1.0) * 0.8)

    # Response time effect (the signal we want to detect)
    effect_strength = 1.2
    response_effect = np.where(
        response_time_mins <= 15, 1.0 + 0.30 * effect_strength,
        np.where(
            response_time_mins <= 30, 1.0 + 0.10 * effect_strength,
            np.where(
                response_time_mins <= 60, 1.0 - 0.10 * effect_strength,
                1.0 - 0.28 * effect_strength
            )
        )
    )
    close_prob *= response_effect

    # Add noise
    close_prob *= np.random.uniform(0.82, 1.18, n_leads)
    close_prob = np.clip(close_prob, 0.005, 0.95)

    ordered = np.random.binomial(1, close_prob)

    # =========================================================================
    # ASSEMBLE DATAFRAME
    # =========================================================================
    df = pd.DataFrame({
        'lead_id': [f"ENT-{i+1:06d}" for i in range(n_leads)],
        'lead_time': lead_times,
        'first_response_time': first_response_times,
        'response_time_mins': response_time_mins.round(1),
        'lead_source': sources,
        'sales_rep': sales_reps,
        'ordered': ordered,
    })

    df = df.sort_values('lead_time').reset_index(drop=True)
    return df


def main():
    print("Generating enterprise-scale dataset (75,000 leads)...")
    df = generate_enterprise_data()

    output_file = "sample_lead_data.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("ENTERPRISE DATASET SUMMARY")
    print("=" * 70)
    print(f"Total leads:          {len(df):,}")
    print(f"Date range:           {df['lead_time'].min().strftime('%Y-%m-%d')} to {df['lead_time'].max().strftime('%Y-%m-%d')}")
    print(f"Weeks of data:        {(df['lead_time'].max() - df['lead_time'].min()).days // 7}")
    print(f"Total orders:         {df['ordered'].sum():,}")
    print(f"Overall close rate:   {df['ordered'].mean():.1%}")
    print(f"Sales reps:           {df['sales_rep'].nunique()}")
    print(f"Lead sources:         {df['lead_source'].nunique()}")
    print(f"Median response time: {df['response_time_mins'].median():.1f} min")
    print(f"Mean response time:   {df['response_time_mins'].mean():.1f} min")

    # Bucket breakdown
    buckets, _ = create_response_buckets(df['response_time_mins'])
    df['response_bucket'] = buckets
    print("\nClose Rate by Response Bucket:")
    bucket_stats = df.groupby('response_bucket', observed=True).agg(
        n_leads=('ordered', 'count'),
        close_rate=('ordered', 'mean')
    )
    for bucket, row in bucket_stats.iterrows():
        print(f"  {bucket:>12s}:  {row['close_rate']:6.1%}  ({int(row['n_leads']):>6,} leads)")

    # Source breakdown
    print("\nClose Rate by Lead Source:")
    source_stats = df.groupby('lead_source')['ordered'].agg(['count', 'mean']).sort_values('mean', ascending=False)
    for source, row in source_stats.iterrows():
        print(f"  {source:>20s}:  {row['mean']:6.1%}  ({int(row['count']):>6,} leads)")

    # Rep correlation
    rep_stats = df.groupby('sales_rep').agg(
        median_response=('response_time_mins', 'median'),
        close_rate=('ordered', 'mean')
    )
    corr = rep_stats['median_response'].corr(rep_stats['close_rate'])
    print(f"\nRep speed-skill correlation: {corr:.2f}")

    print(f"\nFile saved to: {os.path.abspath(output_file)}")


if __name__ == "__main__":
    main()
