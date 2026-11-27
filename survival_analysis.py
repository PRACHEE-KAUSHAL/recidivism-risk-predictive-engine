"""
Offender Recidivism & Statistical Behavioral Analysis
Author: Prachi Kaushal (M.Sc. Mathematics, IIT Jodhpur)
Module: Psychometric validation, Cox Proportional Hazards, and Odds Ratios
"""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

def generate_pilot_offender_cohort(n_inmates=120):
    """
    Simulates structured interview scoring rubrics (1-5) across 4 dimensions
    alongside longitudinal tracking post-release (months to re-offense / censorship).
    """
    np.random.seed(42)
    
    # 4 Interview Modules (Scored 1.0 to 5.0)
    cog_rationalization = np.random.uniform(1.0, 5.0, n_inmates)
    impulsivity_index = np.random.uniform(1.0, 5.0, n_inmates)
    escalation_history = np.random.uniform(1.0, 5.0, n_inmates)
    empathy_deficit = np.random.uniform(1.0, 5.0, n_inmates)
    
    # Simulated survival time (months until infraction/re-offense)
    risk_factor = (0.4 * impulsivity_index) + (0.5 * escalation_history) + (0.3 * cog_rationalization)
    time_to_event = np.random.exponential(scale=36.0 / (risk_factor + 0.1), size=n_inmates)
    time_to_event = np.clip(time_to_event, 1.0, 60.0) # Tracked up to 5 years (60 months)
    
    # Censorship: 1 = re-offended, 0 = successfully reformed / no infraction
    observed_event = (risk_factor > np.median(risk_factor) + np.random.normal(0, 0.4, n_inmates)).astype(int)
    
    df = pd.DataFrame({
        "Subject_ID": [f"INM-{i:03d}" for i in range(1, n_inmates + 1)],
        "Cognitive_Rationalization": cog_rationalization,
        "Impulsivity_Index": impulsivity_index,
        "Escalation_History": escalation_history,
        "Empathy_Deficit": empathy_deficit,
        "Duration_Months": time_to_event,
        "Reoffense_Occurred": observed_event
    })
    return df

def run_cox_survival_analysis(df):
    """
    Fits a Cox Proportional Hazards model to quantify how each psychological
    trait alters the hazard rate of recidivism.
    """
    cph = CoxPHFitter()
    analysis_df = df.drop(columns=["Subject_ID"])
    cph.fit(analysis_df, duration_col="Duration_Months", event_col="Reoffense_Occurred")
    
    print("\n=== Cox Proportional Hazards Model Results ===")
    print(cph.summary[['coef', 'exp(coef)', 'p']])
    print("\n* exp(coef) represents Hazard Ratio. HR > 1.0 indicates accelerated risk.")
    return cph

def calculate_psychometric_consistency(df):
    """
    Computes Cronbach's Alpha to test internal psychometric reliability.
    """
    items = df[["Cognitive_Rationalization", "Impulsivity_Index", "Escalation_History", "Empathy_Deficit"]]
    item_variances = items.var(axis=0, ddof=1)
    total_score_variance = items.sum(axis=1).var(ddof=1)
    k = items.shape[1]
    
    alpha = (k / (k - 1)) * (1 - (item_variances.sum() / total_score_variance))
    print(f"\n[✓] Psychometric Scale Reliability (Cronbach's Alpha): {alpha:.3f}")
    return alpha

if __name__ == "__main__":
    cohort_data = generate_pilot_offender_cohort()
    calculate_psychometric_consistency(cohort_data)
    model = run_cox_survival_analysis(cohort_data)
