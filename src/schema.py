"""
This file defines the exact (stripped) column names from your dataset.xlsx,
and feature groups for ablation study.
"""

CANONICAL_COLS = {
    "timestamp": "Timestamp",
    "consent": "Consent",

    "q1_next_plan": "If you do not get a job in your target field within 6 months after graduation, what will you most likely do next?",
    "q2_status": "Your current status",
    "q3_field": "Field of study",
    "q4_university_type": "University type",
    "q5_residence": "Residence",

    "q6_target_career": "Your primary target career after graduation",
    "q7_influence": "The strongest influence on your career choice",
    "q8_constraint": "Your biggest constraint while choosing a career",

    "q9_uni_support": "University support for career preparation",
    "q10_internship": "Internship experience",
    "q11_skill_source": "Main skill-building source",
    "q12_readiness": "Your current job readiness level",
    "q13_skill_gap": "Biggest skill gap you feel",

    "q14_reject_reason": "The most common reason companies reject fresh graduates (your view)",
    "q15_biggest_problem": "Your biggest problem in getting a job",
    "q16_stress": "Stress level during job searching",
    "q17_employment_status": "Current employment status",

    "q18_transparency": "Recruitment processes are transparent (clear criteria, clear steps)",
    "q19_nepotism": "Nepotism or favoritism affects hiring decisions",
    "q20_fairness": "In your opinion, hiring is mostly fair and merit-based in your field",
}

# Ordinal mappings
LIKERT_5 = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly agree": 5
}

STRESS_5 = {
    "Very low": 1,
    "Low": 2,
    "Medium": 3,
    "High": 4,
    "Very high": 5
}

UNI_SUPPORT_5 = {
    "Very poor": 1,
    "Poor": 2,
    "Average": 3,
    "Good": 4,
    "Very good": 5,
    "Not sure": None,
}

READINESS_5 = {
    "Not ready at all": 1,
    "Slightly ready": 2,
    "Moderately ready": 3,
    "Ready": 4,
    "Very ready": 5
}

# Feature groups for ablation study
FEATURE_GROUPS = {
    "demographics": [
        "q2_status", "q3_field", "q4_university_type", "q5_residence",
    ],
    "career_choice": [
        "q1_next_plan", "q6_target_career", "q7_influence", "q8_constraint",
    ],
    "preparation": [
        "q9_uni_support", "q10_internship", "q11_skill_source", "q12_readiness", "q13_skill_gap",
    ],
    "employment_problems": [
        "q14_reject_reason", "q15_biggest_problem", "q16_stress",
    ],
    "ethics": [
        "q18_transparency", "q19_nepotism", "q20_fairness",
    ],
}
