# Data Cleaning and Preparation Report
## Explainable Multi-Source Machine Learning for Early Identification of At-Risk College Students
**A Case of Zamboanga del Sur Provincial Government College (ZSPGC)**

**Researchers:** Nicholas P. Gabato Jr. · Ferdinand Mayol  
**Academic Program / Unit:** Doctor / Master of Information Technology (DIT), Course Code: DIT-005  
**Institution:** Zamboanga del Sur Provincial Government College (ZSPGC), Aurora Campus  
**Date of Report:** October 13, 2026  

---

## 1. Executive Summary

This report documents the end-to-end data intake, cleaning, integration, feature engineering, and validation process executed on institutional records from Zamboanga del Sur Provincial Government College (ZSPGC). The data preparation aligns with the research methodology specified in the *Preliminary Research Methodology Presentation* and the *Dataset Readiness Report*.

A total of **14 raw institutional Excel datasets** collected across four distinct administrative and academic sources were audited, harmonized, cleaned, and de-identified:
- **Total Student-Term Observations Processed:** 736 records across two consecutive academic periods (S.Y. 2025–2026 Second Semester and S.Y. 2026–2027 First Semester).
- **Unique Individual Students Tracked:** 484 distinct undergraduate students across the Bachelor of Science in Biology (BSBIO) and Bachelor of Science in Information Systems (BSIS) programs.
- **Granular Course-Level Enrollments:** 4,181 subject-level enrollment records, with individual lecture/lab unit breakdowns, letter/numeric grades, and promotional remarks.
- **Data Completeness:** 100% successful record linkage across Registrar Enrollment, Registrar Promotional, and Scholarship TES datasets, with **zero missing values** across all machine learning predictor features.

The clean datasets are organized in the [`cleaned_dataset/`](cleaned_dataset/) directory, separated into historical ground-truth training/evaluation subsets (`student_risk_train_eval.csv`, n=380), prospective early-warning test subsets (`student_risk_prospective_test.csv`, n=356), and pre-encoded machine learning matrices (`student_risk_ml_dataset.csv`).

---

## 2. Institutional Context & Research Objectives

The core objective of this study is to predict college students at risk of academic failure, probation, or dropout before critical intervention cutoffs (such as midterms), utilizing explainable artificial intelligence (XAI via SHAP and LIME) to deliver transparent, actionable guidance to academic advisers, department chairs, and student counselors.

### Research Questions Addressed by the Data Architecture
1. **RQ1 · Multi-Source Integration:** How can disparate institutional data streams—academic enrollment history, end-of-term promotional records, socioeconomic and financial aid data (TES/4Ps), and academic honor achievements—be merged into a unified tabular predictive pipeline?
2. **RQ2 · Comparative Classification:** Which supervised classifier (Random Forest, XGBoost, LightGBM, CatBoost, Logistic Regression) yields optimal sensitivity (Recall), F1-Score, and ROC-AUC when identifying vulnerable students under class imbalance?
3. **RQ3 · Explainable AI:** How do SHAP global feature attributions and localized force/waterfall plots uncover the underlying drivers of student academic vulnerability at ZSPGC?

---

## 3. Inventory of Raw Institutional Sources

The data custodian at ZSPGC released 14 raw spreadsheets spanning two school years and two degree programs:

| # | Raw Spreadsheet Filename | Office of Origin | Records | Key Information Contained |
|---|:---|:---|:---|:---|
| 1 | `Enrollment_Report_2025-2026_-_Second_Sem_..._BIOLOGY.xlsx` | Registrar | 225 students, 1,252 courses | Registered subjects, lecture units, lab units, masterlist control numbers |
| 2 | `Enrollment_Report_2025-2026_-_Second_Sem_..._INFORMATION_SYSTEMS.xlsx` | Registrar | 155 students, 844 courses | Registered subjects, lecture units, lab units, masterlist control numbers |
| 3 | `Enrollment_Report_2026-2027_-_First_Sem_..._BIOLOGY.xlsx` | Registrar | 188 students, 993 courses | Registered subjects, lecture units, lab units, masterlist control numbers |
| 4 | `Enrollment_Report_2026-2027_-_First_Sem_..._INFORMATION_SYSTEMS.xlsx` | Registrar | 168 students, 1,092 courses | Registered subjects, lecture units, lab units, masterlist control numbers |
| 5 | `Promotional_Report_2025-2026_-_Second_Sem_..._BIOLOGY.xlsx` | Registrar | 225 students, 1,252 courses | Final subject grades (1.0–5.0), promotional remarks, General Average (G.A.) |
| 6 | `Promotional_Report_2025-2026_-_Second_Sem_..._INFORMATION_SYSTEMS.xlsx` | Registrar | 155 students, 844 courses | Final subject grades, promotional remarks, General Average (G.A.) |
| 7 | `Promotional_Report_2026-2027_-_First_Sem_..._BIOLOGY.xlsx` | Registrar | 188 students, 993 courses | Prospective cohort; enrolled subjects and total units (grades pending) |
| 8 | `Promotional_Report_2026-2027_-_First_Sem_..._INFORMATION_SYSTEMS.xlsx` | Registrar | 168 students, 1,092 courses | Prospective cohort; enrolled subjects and total units (grades pending) |
| 9 | `TES_Application_Data_2025-2026_-_Second_Sem_..._BIOLOGY.xlsx` | Scholarship / Guidance | 225 applicants | Birthdates, sex, permanent address, 4Ps beneficiary ID, PhilSys, first-gen |
| 10 | `TES_Application_Data_2025-2026_-_Second_Sem_..._INFORMATION_SYSTEMS.xlsx` | Scholarship / Guidance | 155 applicants | Birthdates, sex, permanent address, 4Ps beneficiary ID, PhilSys, first-gen |
| 11 | `TES_Application_Data_2026-2027_-_First_Sem_..._BIOLOGY (14).xlsx` | Scholarship / Guidance | 188 applicants | Birthdates, sex, permanent address, 4Ps beneficiary ID, PhilSys, first-gen |
| 12 | `TES_Application_Data_2026-2027_-_First_Sem_..._INFORMATION_SYSTEMS (2).xlsx`| Scholarship / Guidance | 168 applicants | Birthdates, sex, permanent address, 4Ps beneficiary ID, PhilSys, first-gen |
| 13 | `Top_Performing_Students_2025-2026_-_Second_Sem_..._INFORMATION_SYSTEMS.xlsx`| Academic Affairs | 1,038 honor entries | College-wide honor student masterlist, ranks, and GPAs |
| 14 | `Top_Performing_Students_2026-2027_-_First_Sem_..._INFORMATION_SYSTEMS.xlsx` | Academic Affairs | Preliminary entries | College-wide honor student masterlist |

### Source Linkage Verification
Every student in each cohort was matched across Enrollment, Promotional, and TES Application files using their official collegiate Student Number (`student_no`), achieving a **100% 1-to-1 match rate**:
- **2025–2026 Second Sem (BS Bio):** 225 / 225 matched (100.0%)
- **2025–2026 Second Sem (BS IS):** 155 / 155 matched (100.0%)
- **2026–2027 First Sem (BS Bio):** 188 / 188 matched (100.0%)
- **2026–2027 First Sem (BS IS):** 168 / 168 matched (100.0%)

---

## 4. Compliance with Republic Act 10173 (Data Privacy Act of 2012)

In strict accordance with statutory requirements and institutional ethics protocols (Dataset Readiness Report, Section 7):
1. **Pseudonymization:** All direct identifiers (`Student Number`, `Student Name`, `LRN`, `PhilSys ID`, `4Ps ID Number`, `Contact Number`, `Email Address`, and specific street addresses) were detached from all analytical and modeling datasets.
2. **Persistent Pseudonymous Key (`student_pid`):** Each student was assigned a standardized pseudonymous identifier (e.g., `PID-ZSPGC-0001` through `PID-ZSPGC-0484`). This code is preserved across terms for returning students, enabling student-grouped cross-validation without personal re-identification.
3. **Isolated Security Key Table (`student_id_lookup_private.csv`):** An administrative lookup table was generated and segregated in [`cleaned_dataset/student_id_lookup_private.csv`](cleaned_dataset/student_id_lookup_private.csv). This file is restricted to the College Registrar and authorized guidance personnel for intervention routing and must never be fed into model training pipelines.

---

## 5. Detailed Data Cleaning and Transformation Procedures

### 5.1 Geographic Standardization and Commuting Distance Modeling
The ZSPGC main campus is located in **Poblacion, Aurora, Zamboanga del Sur**. The home municipality field (`city_muni`) from TES records was standardized and mapped to road commuting distance in kilometers using the regional transport network:

| Standardized Municipality | Province | Distance to Campus (km) | Locality Class | Observations in Dataset |
|:---|:---|:---|:---|:---|
| **AURORA** | Zamboanga del Sur | 2.0 km | Campus Locality (<=5km) | 499 (67.8%) |
| **RAMON MAGSAYSAY** | Zamboanga del Sur | 14.0 km | Nearby Commuter (5–25km) | 41 (5.6%) |
| **TUKURAN** | Zamboanga del Sur | 17.0 km | Nearby Commuter (5–25km) | 76 (10.3%) |
| **TAMBULIG** | Zamboanga del Sur | 22.0 km | Nearby Commuter (5–25km) | 18 (2.4%) |
| **KAPATAGAN** | Lanao del Norte | 28.0 km | Regional / Distant (>25km) | 51 (6.9%) |
| **LABANGAN** | Zamboanga del Sur | 32.0 km | Regional / Distant (>25km) | 8 (1.1%) |
| **MIDSALIP** | Zamboanga del Sur | 32.0 km | Regional / Distant (>25km) | 7 (1.0%) |
| **SAPAD** | Lanao del Norte | 32.0 km | Regional / Distant (>25km) | 1 (0.1%) |
| **SULTAN NAGA DIMAPORO (S.N.D.)** | Lanao del Norte | 35.0 km | Regional / Distant (>25km) | 10 (1.4%) |
| **LALA** | Lanao del Norte | 36.0 km | Regional / Distant (>25km) | 3 (0.4%) |
| **SOMINOT** | Zamboanga del Sur | 38.0 km | Regional / Distant (>25km) | 7 (1.0%) |
| **MAHAYAG** | Zamboanga del Sur | 38.0 km | Regional / Distant (>25km) | 3 (0.4%) |
| **PAGADIAN CITY** | Zamboanga del Sur | 42.0 km | Regional / Distant (>25km) | 3 (0.4%) |
| **DUMINGAG** | Zamboanga del Sur | 48.0 km | Regional / Distant (>25km) | 2 (0.3%) |
| **BAYOG** | Zamboanga del Sur | 70.0 km | Regional / Distant (>25km) | 1 (0.1%) |
| **DIPLAHAN** | Zamboanga Sibugay | 80.0 km | Regional / Distant (>25km) | 2 (0.3%) |
| **ILIGAN CITY** | Lanao del Norte | 95.0 km | Regional / Distant (>25km) | 2 (0.3%) |
| **OROQUIETA CITY** | Misamis Occidental | 105.0 km | Regional / Distant (>25km) | 1 (0.1%) |
| **IPIL** | Zamboanga Sibugay | 130.0 km | Regional / Distant (>25km) | 1 (0.1%) |

### 5.2 Chronological Demographic Feature Engineering
- **Student Age:** Derived by parsing ISO birthdates (`YYYY-MM-DD`, 100% valid) against the term commencement dates (January 15, 2026 for Sem 2; August 15, 2026 for Sem 1).
  - Observed range: **17 to 33 years** (Mean = 20.48 years, Median = 20.0 years).
- **Biological Sex:** Cross-verified across Registrar records (`M`/`F`) and TES codes (`0` = Male, `1` = Female), achieving 100% consistency. Overall cohort: **63.5% Female, 36.5% Male**.

### 5.3 Socioeconomic & Financial Aid Features
- **4Ps Beneficiary Flag (`is_4ps_beneficiary`):** In the Philippines, the Pantawid Pamilyang Pilipino Program (4Ps) identifies households living below the poverty threshold in the national DSWD Listahanan registry. A binary indicator was extracted from the presence of a verified 4Ps ID number (93 observations in total).
- **Ordinal Household Income Bracket (`income_bracket`):**
  - **Bracket 1 (Indigent / Extreme Poverty):** 4Ps beneficiary households.
  - **Bracket 2 (Low-Income Subsidized):** Non-4Ps students attending under Free Higher Education (RA 10931) and TES subsidy qualification.
- **First-Generation College Student (`is_first_gen_college`):** 160 students (21.7%) reported being the first in their immediate families to enter tertiary education.
- **Special Status Flags:** Documented disabilities (Speech/Language, Visual, Physical: 8 cases) and Indigenous Peoples group affiliations (Iranun, Subanen: 3 cases).

### 5.4 Academic Load & Course Parsing
- **Unit Sanitization:** Addressed institutional formatting variations where non-academic or remedial credits were designated with parenthesized integers (e.g., `(3)`). All values were cleaned to valid numerical floats.
- **Laboratory Course Identification:** Cross-referenced Enrollment Report lecture/lab split columns to extract `has_lab_courses` (binary) and `lab_courses_count` (integer), representing higher course workload and practical examination intensity.
- **Academic Credit Load Categories:**
  - `Underload (<15 units)`: 63 students (16.6% in historical cohort)
  - `Normal Load (15–23 units)`: 310 students (81.6% in historical cohort)
  - `Overload (>23 units)`: 7 students (1.8% in historical cohort)

### 5.5 Institutional Honor Merit Matching
Cross-referenced student numbers against the college-wide *Top Performing Students* honor rosters:
- 163 out of 225 BS Biology students (72.4%) and 146 out of 155 BS Information Systems students (94.2%) in the historical cohort have recorded honor rankings or GPA distinctions.
- Engineered `is_top_performer` (binary) and `honor_rank` (integer) features.

---

## 6. Target Variable Construction & Distribution Analysis

### 6.1 Operational Definition of `at_risk`
In accordance with the *Dataset Readiness Report* and institutional academic regulations at ZSPGC:
$$\text{at\_risk} = \begin{cases} 
1 & \text{if } (\text{failed\_courses} > 0) \lor (\text{inc\_courses} > 0) \lor (\text{dropped\_courses} > 0) \lor (\text{term\_gpa} > 3.0) \\
0 & \text{if } (\text{deficiencies} = 0) \land (\text{term\_gpa} \le 3.0)
\end{cases}$$

In the Philippine collegiate grading system:
- **1.00 to 3.00:** Passing grade (1.00 = Highest Honors / Excellent; 3.00 = Passing).
- **5.00:** Failing grade (Failed).
- **INC:** Incomplete grade (failure to submit term requirements or take final exams).
- **Dropped:** Official/unofficial course withdrawal resulting in academic non-completion.

### 6.2 Ground-Truth Class Distribution (Historical Cohort, n=380)

| Academic Outcome | Count | Proportion | Machine Learning Class |
|:---|:---|:---|:---|
| **Satisfactory Academic Standing** | 286 | 75.26% | **Class 0 (Not At-Risk)** |
| **Probation – Failed Course(s)** | 68 | 17.89% | **Class 1 (At-Risk)** |
| **Deficiency – Incomplete (INC)** | 23 | 6.05% | **Class 1 (At-Risk)** |
| **Attrition – Dropped Course(s)** | 3 | 0.79% | **Class 1 (At-Risk)** |
| **Total** | **380** | **100.00%** | **Class Ratio: 1 : 3.04 (24.74% Minority)** |

> [!NOTE]
> The resulting minority class proportion of **24.74%** mirrors the exact planning target hypothesized in the *Dataset Readiness Report* ("At-risk students are expected to be the minority class around 20–30%"). This natural class imbalance confirms the necessity of using **Recall**, **F1-Score**, and **ROC-AUC** as primary performance indicators.

---

## 7. Exploratory Statistical Findings & Vulnerability Factors

Auditing the historical ground-truth training set revealed distinct educational and demographic vulnerability patterns:

### 7.1 Program Vulnerability Disparity
- **BS Biology:** 84 out of 225 students (**37.33%**) were identified as At-Risk. The natural sciences curriculum features rigorous laboratory courses (e.g., Bio 110 Developmental Biology, Bio 114 Epidemiology, Bio 115 Immunology) where practical examinations generate substantial academic attrition.
- **BS Information Systems:** Only 10 out of 155 students (**6.45%**) were identified as At-Risk, indicating higher pass rates and project-based continuous assessment.

### 7.2 Academic Year Level Dynamics
- **Year 1 (Freshmen):** **41.0% At-Risk** (48 of 117 students). Freshmen experience significant transitional shock adapting to tertiary expectations and general education courses.
- **Year 2 (Sophomores):** **12.9% At-Risk** (8 of 62 students). Stronger academic stabilization.
- **Year 3 (Juniors):** **40.9% At-Risk** (38 of 93 students). Juniors face complex major barrier courses and undergraduate research/thesis proposals (*Thesis 2 / Thesis 3*).
- **Year 4 (Seniors):** **0.0% At-Risk** (0 of 108 students). All seniors who reach the final term consistently complete practicum and final research defense without failure.

### 7.3 Commuting Distance & Locality Risk
- **Campus Area (<= 5 km, Aurora):** **24.5% At-Risk** (64 of 261 students).
- **Nearby Commuter (5–25 km, Tukuran/Ramon Magsaysay):** **18.3% At-Risk** (13 of 71 students).
- **Regional / Distant (> 25 km, Kapatagan/S.N.D./Tambulig):** **35.4% At-Risk** (17 of 48 students).  
  *Students residing more than 25 km away experience a **+10.9% higher risk rate**, reflecting transportation fatigue, rural transit costs, and accommodation instability.*

---

## 8. Cross-Cohort Student Tracking & Progression

Tracking the 484 unique students longitudinally between S.Y. 2025–2026 Second Semester and S.Y. 2026–2027 First Semester revealed clear student progression dynamics:
- **Returning Students:** 252 students (145 in BS Bio, 107 in BS IS) re-enrolled in the subsequent semester. Their historical GPA (`prior_term_gpa`), prior deficiencies (`prior_deficiencies`), and prior risk flag (`prior_at_risk`) were carried forward as predictive features.
- **Graduating Cohort:** 101 Year 4 students (60 in Bio, 41 in IS) completed their final semester and graduated.
- **New Incoming Cohort:** 104 first-time students (43 in Bio, 61 in IS) entered the 2026–2027 academic year.
- **Attrition / Non-Returning:** 27 undergraduate students (20 in Year 1, 3 in Year 2, 4 in Year 3) did not re-enroll in the next term, highlighting early-stage college dropout.

---

## 9. Machine Learning Pipeline Recommendations

To ensure methodological rigor in the upcoming modeling phase (addressing RQ2 and RQ3):

### 9.1 Evaluation Strategy & Validation Splits
1. **Student-Grouped Stratified K-Fold Cross-Validation:**
   Use `GroupKFold` or `StratifiedGroupKFold` ($k = 5$ or $10$) on `student_risk_train_eval.csv` grouped by `student_pid`. This guarantees that records belonging to the same student never leak across training and validation folds simultaneously.
2. **Class Imbalance Treatment:**
   With a 1 : 3.04 positive class ratio, apply algorithmic class weighting (`scale_pos_weight = 3.04` in XGBoost/LightGBM; `class_weight='balanced'` in Random Forest and Logistic Regression) or SMOTE fitted **strictly inside training folds** (never on validation sets).
3. **Primary Evaluation Metrics:**
   - **Recall (Sensitivity):** Crucial to ensure at-risk students are not overlooked.
   - **F1-Score:** Harmonic balance of precision and recall.
   - **ROC-AUC:** Overall diagnostic discriminatory ability across decision thresholds.
4. **Prospective Model Scoring:**
   Once trained and tuned, the optimal classifier will score the prospective holdout set (`student_risk_prospective_test.csv`, n=356), outputting predicted risk probabilities $P(\text{At-Risk})$ to enable pre-midterm intervention by ZSPGC counselors.

---

## 10. Summary of Generated Output Files

All cleaned artifacts have been verified, checksummed, and saved to the [`cleaned_dataset/`](cleaned_dataset/) folder:

1. [`student_risk_full_profile.csv`](cleaned_dataset/student_risk_full_profile.csv): 736 records, 45 consolidated features.
2. [`student_risk_ml_dataset.csv`](cleaned_dataset/student_risk_ml_dataset.csv): 736 records, 31 encoded numerical features for ML models.
3. [`student_risk_train_eval.csv`](cleaned_dataset/student_risk_train_eval.csv): 380 records with verified target `at_risk` (75.3% Satisfactory, 24.7% At-Risk).
4. [`student_risk_prospective_test.csv`](cleaned_dataset/student_risk_prospective_test.csv): 356 prospective observations for early deployment inference.
5. [`course_enrollments_cleaned.csv`](cleaned_dataset/course_enrollments_cleaned.csv): 4,181 granular course enrollment records.
6. [`student_id_lookup_private.csv`](cleaned_dataset/student_id_lookup_private.csv): Isolated dean/registrar de-identification lookup key.
7. [`DATA_DICTIONARY.md`](cleaned_dataset/DATA_DICTIONARY.md): Comprehensive variable catalog and dictionary.
