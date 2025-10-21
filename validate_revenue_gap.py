import pandas as pd
import numpy as np

# Load the final dataset
print("Loading the synthetic dataset...")
df = pd.read_csv('Synthetic_Sales_Performance_Data_GPBased_Final (2).csv')

print(f"Total dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Step 1: Filter for December month only
print("\n=== STEP 1: Filter for December month ===")
dec_data = df[df['Month'] == 'Dec'].copy()
print(f"December records: {len(dec_data)}")

# Step 2: Filter by Performance <= 90%
print("\n=== STEP 2: Filter by Performance <= 90% ===")
underperformers_dec = dec_data[dec_data['Performance'] <= 90].copy()
print(f"December underperformers (≤90%): {len(underperformers_dec)}")

# Step 3: Calculate sums of YTD_Quota and YTD_GP
print("\n=== STEP 3: Calculate Revenue Gap ===")
total_ytd_quota = underperformers_dec['YTD_Quota'].sum()
total_ytd_gp = underperformers_dec['YTD_GP'].sum()
revenue_gap = total_ytd_quota - total_ytd_gp

print(f"Sum of YTD_Quota (underperformers in Dec): ${total_ytd_quota:,.2f}")
print(f"Sum of YTD_GP (underperformers in Dec):    ${total_ytd_gp:,.2f}")
print(f"Revenue Gap (Quota - GP):                  ${revenue_gap:,.2f}")

# Step 4: Verify against target
print(f"\n=== STEP 4: Target Validation ===")
target_revenue_gap = 827_000_000
gap_difference = revenue_gap - target_revenue_gap
gap_percentage = (revenue_gap / target_revenue_gap) * 100

print(f"Target Revenue Gap:     ${target_revenue_gap:,.2f}")
print(f"Actual Revenue Gap:     ${revenue_gap:,.2f}")
print(f"Difference:             ${gap_difference:,.2f}")
print(f"Percentage of target:   {gap_percentage:.1f}%")

if 750_000_000 <= revenue_gap <= 900_000_000:
    print("✅ Revenue Gap is within acceptable range ($750M - $900M)")
else:
    print("❌ Revenue Gap is outside acceptable range")

# Step 5: Additional analysis - Rep-level breakdown
print(f"\n=== STEP 5: Rep-Level Analysis ===")
# Group by Rep_ID to get rep-level performance
rep_level = dec_data.groupby('Rep_ID').agg({
    'YTD_Quota': 'sum',
    'YTD_GP': 'sum',
    'Performance': 'first'  # Performance should be same across all products for same rep
}).reset_index()

rep_level['Rep_Achievement'] = (rep_level['YTD_GP'] / rep_level['YTD_Quota']) * 100
underperformer_reps = rep_level[rep_level['Rep_Achievement'] <= 90]

print(f"Total unique reps in December: {len(rep_level)}")
print(f"Underperformer reps (≤90%): {len(underperformer_reps)}")
print(f"Percentage of underperformers: {len(underperformer_reps)/len(rep_level)*100:.1f}%")

# Performance distribution
severe = len(rep_level[rep_level['Rep_Achievement'] < 50])
critical = len(rep_level[(rep_level['Rep_Achievement'] >= 50) & (rep_level['Rep_Achievement'] < 70)])
at_risk = len(rep_level[(rep_level['Rep_Achievement'] >= 70) & (rep_level['Rep_Achievement'] < 90)])
acceptable = len(rep_level[(rep_level['Rep_Achievement'] >= 90) & (rep_level['Rep_Achievement'] < 100)])
high_performer = len(rep_level[rep_level['Rep_Achievement'] >= 100])

print(f"\nPerformance Distribution:")
print(f"  Severe (<50%):        {severe} reps")
print(f"  Critical (50-69%):    {critical} reps")
print(f"  At Risk (70-89%):     {at_risk} reps")
print(f"  Acceptable (90-99%):  {acceptable} reps")
print(f"  High Performers (≥100%): {high_performer} reps")
print(f"  Total:                {severe + critical + at_risk + acceptable + high_performer} reps")

# Step 6: Verify the formula matches TODO.md specification
print(f"\n=== STEP 6: Formula Verification ===")
print("TODO.md Formula: Revenue Gap = Σ(YTD_Quota for Dec) − Σ(YTD_GP for Dec) for underperformers")
print(f"Applied Formula: {total_ytd_quota:,.2f} - {total_ytd_gp:,.2f} = {revenue_gap:,.2f}")

# Step 7: Save filtered data for inspection
underperformers_dec.to_csv('December_Underperformers_Analysis.csv', index=False)
rep_level.to_csv('Rep_Level_December_Analysis.csv', index=False)

print(f"\n=== OUTPUT FILES CREATED ===")
print("📁 December_Underperformers_Analysis.csv - Product-level underperformer data")
print("📁 Rep_Level_December_Analysis.csv - Rep-level aggregated performance")

print(f"\n🎯 FINAL RESULT:")
print(f"Revenue Gap = ${revenue_gap:,.2f}")
print(f"Target = ${target_revenue_gap:,.2f}")
print(f"Status: {'✅ WITHIN RANGE' if 750_000_000 <= revenue_gap <= 900_000_000 else '❌ OUTSIDE RANGE'}")