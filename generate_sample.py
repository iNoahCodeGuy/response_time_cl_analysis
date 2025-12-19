#!/usr/bin/env python3
"""
Generate Sample Lead Data
=========================
Run this script to create a realistic sample CSV file for testing.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.sample_generator import generate_sample_data, get_sample_data_summary
from analysis.preprocessing import create_response_buckets
import pandas as pd

def main():
    print("Generating 10,000 realistic leads...")
    df = generate_sample_data(n_leads=10000, random_seed=42)
    
    # Save to CSV
    output_file = "sample_lead_data.csv"
    df.to_csv(output_file, index=False)
    print("Saved to {}".format(output_file))
    
    # Show summary
    print("\n" + "="*60)
    print("SAMPLE DATA SUMMARY")
    print("="*60)
    
    summary = get_sample_data_summary(df)
    print("Total leads: {:,}".format(summary['total_leads']))
    print("Date range: {} to {}".format(summary['date_range']['start'], summary['date_range']['end']))
    print("Total orders: {:,}".format(summary['total_orders']))
    print("Overall close rate: {:.1%}".format(summary['overall_close_rate']))
    print("Median response time: {:.1f} minutes".format(summary['response_time']['median_mins']))
    print("Number of sales reps: {}".format(summary['sales_reps']['count']))
    
    print("\nLead Sources:")
    for source, count in summary['lead_sources'].items():
        pct = count / summary['total_leads'] * 100
        print("  - {}: {:,} ({:.1f}%)".format(source, count, pct))
    
    # Create buckets for display
    buckets, bucket_summary = create_response_buckets(df['response_time_mins'])
    df['response_bucket'] = buckets
    
    print("\nClose rates by response bucket:")
    bucket_rates = df.groupby('response_bucket', observed=True).agg(
        n_leads=('ordered', 'count'),
        close_rate=('ordered', 'mean')
    )
    for bucket, row in bucket_rates.iterrows():
        print("  - {}: {:.1%} ({:,} leads)".format(bucket, row['close_rate'], int(row['n_leads'])))
    
    print("\n" + "="*60)
    print("SAMPLE DATA PREVIEW (first 5 rows)")
    print("="*60)
    preview_cols = ['lead_id', 'lead_time', 'first_response_time', 'response_time_mins', 
                    'lead_source', 'sales_rep', 'ordered']
    print(df[preview_cols].head().to_string())
    
    print("\n" + "="*60)
    print("KEY PATTERNS IN THE DATA")
    print("="*60)
    
    # Rep correlation
    rep_stats = df.groupby('sales_rep').agg(
        median_response=('response_time_mins', 'median'),
        close_rate=('ordered', 'mean')
    )
    corr = rep_stats['median_response'].corr(rep_stats['close_rate'])
    print("\nRep speed-skill correlation: {:.2f}".format(corr))
    print("(Negative = faster reps also close more, indicating confounding)")
    
    # Source differences
    print("\nClose rate by lead source:")
    source_rates = df.groupby('lead_source')['ordered'].mean().sort_values(ascending=False)
    for source, rate in source_rates.items():
        print("  - {}: {:.1%}".format(source, rate))
    
    print("\nDone! File saved to: {}".format(os.path.abspath(output_file)))

if __name__ == "__main__":
    main()

