# ZSPGC Multi-Source Student Risk Dataset: Data Dictionary

**Research Project:** Explainable Multi-Source Machine Learning for Early Identification of At-Risk College Students: A Case of Zamboanga del Sur Provincial Government College  
**Researchers:** Nicholas P. Gabato Jr. · Ferdinand Mayol  
**Institution:** Zamboanga del Sur Provincial Government College (ZSPGC), Aurora Campus  
**Curator / Prepared For:** DIT Research Study (DIT-005)  
**Date Generated:** October 2026  

---

## 1. Overview of Cleaned Dataset Files

| Filename | Records | Columns | Description |
| :--- | :--- | :--- | :--- |
| `student_risk_full_profile.csv` | 736 | 45 | Complete student-term dataset integrating Registrar academic records, TES scholarship & socioeconomic profiles, demographic attributes, and academic outcomes. |
| `student_risk_ml_dataset.csv` | 736 | 31 | De-identified, pre-encoded machine learning feature matrix ready for baseline models (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost) and SHAP explainability. |
| `student_risk_train_eval.csv` | 380 | 45 | Historical benchmark cohort (S.Y. 2025–2026 Second Semester) with verified ground-truth target labels (`at_risk` = 0 / 1) for model training, stratified cross-validation, and hyperparameter tuning. |
| `student_risk_prospective_test.csv` | 356 | 45 | Prospective holdout cohort (S.Y. 2026–2027 First Semester) representing the active deployment term for early warning risk prediction before midterm/final evaluations. |
| `course_enrollments_cleaned.csv` | 4,181 | 16 | Granular course/subject enrollment records including subject codes, descriptive titles, lecture/lab units, grades, and promotional remarks. |
| `student_id_lookup_private.csv` | 736 | 17 | **Restricted Security Key:** Isolated linkage table mapping pseudonymous IDs (`student_pid`) to student numbers and personal identifying information (PII), kept strictly for institutional counselor/dean follow-up under RA 10173. |

---

## 2. Student Profile Data Dictionary (`student_risk_full_profile.csv` & Subsets)

| # | Column Name | Source System | Data Type | Possible / Observed Values | Research Role | Description |
|---|:---|:---|:---|:---|:---|:---|
| 1 | `student_pid` | Generated / Registrar | String | `PID-ZSPGC-0001` to `PID-ZSPGC-0484` | Identifier / Split Key | Pseudonymous student ID replacing original student number and name. Persistent across terms to support student-grouped cross-validation. |
| 2 | `school_year` | Registrar | Categorical | `2025-2026`, `2026-2027` | Split Key | Academic school year of the observation. |
| 3 | `term` | Registrar | Categorical | `2nd Sem`, `1st Sem` | Split Key | Academic semester of enrollment. |
| 4 | `term_code` | Derived | Categorical | `2025-2026_Sem2`, `2026-2027_Sem1` | Split Key | Combined academic year and term code. |
| 5 | `split_type` | Methodology | Categorical | `train_eval`, `prospective_test` | Partition Key | Dataset split partition: `train_eval` (historical ground truth) vs `prospective_test` (prospective holdout). |
| 6 | `program` | Registrar | Categorical | `BSBIO`, `BSIS` | Predictor | Degree program code (`BSBIO` = BS Biology, `BSIS` = BS Information Systems). |
| 7 | `program_title` | Registrar | Categorical | Text description | Descriptive | Complete official degree program title. |
| 8 | `year_level` | Registrar / TES | Ordinal | `1`, `2`, `3`, `4` | Predictor | Academic curriculum year level of student (1 = Freshman, 2 = Sophomore, 3 = Junior, 4 = Senior). |
| 9 | `age` | TES / Guidance | Numeric | `17` to `33` | Predictor | Exact student age in whole years calculated at term commencement from ISO birthdate. |
| 10 | `sex` | Registrar / TES | Categorical | `M`, `F` | Predictor / Fairness Audit | Biological sex as recorded in official college records. |
| 11 | `is_female` | Derived | Binary | `0`, `1` | Predictor (Encoded) | Binary flag (1 = Female, 0 = Male). Audited in SHAP analysis for fairness. |
| 12 | `residence_municipality`| TES / Guidance | Categorical | `AURORA`, `TUKURAN`, `KAPATAGAN`, `RAMON MAGSAYSAY`, `TAMBULIG`, etc. | Predictor | Standardized permanent home municipality or city of residence. |
| 13 | `residence_province` | TES / Guidance | Categorical | `ZAMBOANGA DEL SUR`, `LANAO DEL NORTE`, `ZAMBOANGA SIBUGAY`, etc. | Predictor | Province of residence. |
| 14 | `zipcode` | TES / Guidance | Categorical | Postal zip codes (e.g., `7020`, `7019`) | Demographic | Postal ZIP code of residence. |
| 15 | `distance_km` | Derived / GIS | Numeric | `2.0` to `130.0` km | Predictor | Approximate road transit distance in kilometers from residence municipality to ZSPGC Aurora Campus. |
| 16 | `locality_class` | Derived | Categorical | `Campus Locality (<=5km)`, `Nearby Commuter (5-25km)`, `Regional / Distant (>25km)` | Predictor | Locality distance classification for transportation vulnerability. |
| 17 | `is_4ps_beneficiary` | TES / Scholarship | Binary | `0`, `1` | Predictor | 1 if household is a Pantawid Pamilyang Pilipino Program (4Ps) national indigent beneficiary; 0 otherwise. |
| 18 | `income_bracket` | TES / Scholarship | Ordinal | `1`, `2` | Predictor | Household income classification: 1 = Lowest / Indigent (4Ps), 2 = Low-Income Subsidized (FHE/TES). |
| 19 | `has_philsys_id` | TES / Scholarship | Binary | `0`, `1` | Predictor | 1 if Philippine Identification System (PhilSys) National ID number is registered. |
| 20 | `is_first_gen_college`| TES / Guidance | Binary | `0`, `1` | Predictor | 1 if student is a first-generation college student in their family; 0 otherwise. |
| 21 | `is_dependent` | TES / Guidance | Binary | `0`, `1` | Predictor | 1 if student has dependents; 0 otherwise. |
| 22 | `has_disability` | TES / Guidance | Binary | `0`, `1` | Predictor | 1 if student has documented disability; 0 otherwise. |
| 23 | `disability_type` | TES / Guidance | Categorical | `None`, `SPEECH AND LANGUAGE`, `VISUAL`, `PHYSICAL` | Predictor | Documented disability category. |
| 24 | `is_indigenous` | TES / Guidance | Binary | `0`, `1` | Predictor | 1 if student belongs to an Indigenous People (IP) group; 0 otherwise. |
| 25 | `indigenous_group` | TES / Guidance | Categorical | `None`, `Iranun`, `Subanen` | Predictor | Name of Indigenous People group. |
| 26 | `tes_grantee_flag` | Scholarship Office | Binary | `1` | Predictor (Constant) | Tertiary Education Subsidy grantee/applicant status (100% of cohort is covered). |
| 27 | `enrolled_units` | Registrar | Numeric | `6.0` to `26.0` | Predictor | Total academic credits/units registered in the term. |
| 28 | `course_count` | Registrar | Numeric | `2` to `8` | Predictor | Total number of courses/subjects enrolled in the term. |
| 29 | `has_lab_courses` | Registrar | Binary | `0`, `1` | Predictor | 1 if student is enrolled in courses with laboratory components; 0 otherwise. |
| 30 | `lab_courses_count` | Registrar | Numeric | `0` to `5` | Predictor | Count of courses with laboratory contact hours. |
| 31 | `academic_load_status`| Derived | Categorical | `Underload (<15)`, `Normal Load (15-23)`, `Overload (>23)` | Predictor | Categorized academic credit workload. |
| 32 | `is_top_performer` | Honors Masterlist | Binary | `0`, `1` | Predictor | 1 if student appears on the college-wide Top Performing honor list; 0 otherwise. |
| 33 | `honor_rank` | Honors Masterlist | Numeric | `1` to `1038` (or blank) | Predictor | Institutional academic merit rank in Top Performing list. |
| 34 | `is_returning_student`| Registrar | Binary | `0`, `1` | Predictor | 1 if student continued from prior semester; 0 if new/freshman in dataset. |
| 35 | `prior_term_gpa` | Registrar | Numeric | `1.0` to `5.0` (or blank) | Predictor | Prior semester General Average (GPA) for returning students. |
| 36 | `prior_deficiencies` | Registrar | Numeric | `0` to `5` | Predictor | Count of prior semester failed, incomplete, or dropped subjects. |
| 37 | `prior_at_risk` | Registrar | Binary | `0`, `1` | Predictor | Prior semester at-risk indicator for returning students. |
| 38 | `passed_courses_count`| Registrar (Promo) | Numeric | `0` to `8` (or blank) | Outcome | Number of courses successfully completed with passing grade. |
| 39 | `failed_courses_count`| Registrar (Promo) | Numeric | `0` to `5` (or blank) | Outcome | Number of courses failed (grade 5.0). |
| 40 | `inc_courses_count` | Registrar (Promo) | Numeric | `0` to `4` (or blank) | Outcome | Number of courses marked Incomplete (INC). |
| 41 | `dropped_courses_count`| Registrar (Promo) | Numeric | `0` to `3` (or blank) | Outcome | Number of courses officially dropped/withdrawn. |
| 42 | `passed_courses_ratio`| Derived | Numeric | `0.0` to `1.0` (or blank) | Outcome | Proportion of enrolled courses successfully passed. |
| 43 | `term_gpa` | Registrar (Promo) | Numeric | `1.0` to `5.0` (or blank) | Outcome | End-of-term General Average (G.A.) on standard Philippine grading scale. |
| 44 | `academic_standing` | Derived | Categorical | `Satisfactory`, `Probation - Failed Course`, `Deficiency - Incomplete (INC)`, `Attrition - Dropped Course` | Outcome | Descriptive academic standing status at end of term. |
| 45 | `at_risk` | Target Variable | Binary | `0` = Not At-Risk, `1` = At-Risk (blank for prospective) | **Primary Target** | Ground-truth risk classification: 1 if student suffered academic failure, probation, INC, or dropout; 0 if satisfactory standing. |

---

## 3. Granular Course Records Data Dictionary (`course_enrollments_cleaned.csv`)

| # | Column Name | Data Type | Description |
|---|:---|:---|:---|
| 1 | `student_pid` | String | Pseudonymous student ID linking to student profile. |
| 2 | `school_year` | Categorical | Academic school year (`2025-2026`, `2026-2027`). |
| 3 | `term` | Categorical | Semester of course enrollment (`2nd Sem`, `1st Sem`). |
| 4 | `term_code` | Categorical | Combined term identifier. |
| 5 | `program` | Categorical | Degree program (`BSBIO`, `BSIS`). |
| 6 | `year_level` | Ordinal | Curriculum year level (`1`–`4`). |
| 7 | `course_index` | Numeric | Sequential course index for the student in that term. |
| 8 | `subject_code` | String | Official institutional course catalogue code (e.g., `Bio 115`, `IS 201`, `Thesis 3`). |
| 9 | `subject_title` | String | Descriptive course title (e.g., `Immunology`, `Data Mining`, `Final Research`). |
| 10 | `total_units` | Numeric | Total academic units/credits assigned to the course. |
| 11 | `lecture_units` | Numeric | Lecture units/hours component. |
| 12 | `lab_units` | Numeric | Laboratory units/hours component. |
| 13 | `control_no` | String | Registrar masterlist section control number. |
| 14 | `grade_raw` | String | Raw grade text from promotional report. |
| 15 | `grade_num` | Numeric | Numeric grade on Philippine 1.0–5.0 scale (1.0 = highest, 3.0 = passing, 5.0 = failed). |
| 16 | `remarks` | Categorical | Promotional status: `Passed`, `Failed`, `INC`, `Dropped`, or `In Progress`. |

---

## 4. Machine Learning Feature Matrix Data Dictionary (`student_risk_ml_dataset.csv`)

Pre-encoded, purely numeric format ready for direct intake into scikit-learn, XGBoost, LightGBM, and CatBoost pipelines:

| Variable Name | Type | Definition / Values |
|:---|:---|:---|
| `student_pid` | String | Entity ID for GroupKFold cross-validation. |
| `school_year`, `term`, `split_type` | String | Metadata keys. |
| `program_is_bsis` | Binary | 1 = BS Information Systems, 0 = BS Biology. |
| `year_level` | Integer | 1, 2, 3, 4. |
| `age` | Integer | 17 to 33. |
| `is_female` | Binary | 1 = Female, 0 = Male. |
| `distance_km` | Float | Approximate road travel distance to campus in km. |
| `locality_is_campus_area` | Binary | 1 if home distance <= 5 km, 0 otherwise. |
| `locality_is_distant` | Binary | 1 if home distance > 25 km, 0 otherwise. |
| `is_4ps_beneficiary` | Binary | 1 = Pantawid Pamilyang Pilipino indigent beneficiary, 0 = Non-4Ps. |
| `income_bracket` | Integer | 1 = Lowest / Indigent, 2 = Low-income subsidized. |
| `has_philsys_id` | Binary | 1 = National ID present, 0 = Missing. |
| `is_first_gen_college` | Binary | 1 = First-generation college student, 0 = Continuing generation. |
| `is_dependent` | Binary | 1 = Has dependents, 0 = No dependents. |
| `has_disability` | Binary | 1 = Documented disability, 0 = None. |
| `is_indigenous` | Binary | 1 = Indigenous people member, 0 = General population. |
| `enrolled_units` | Float | Total registered academic units. |
| `course_count` | Integer | Total enrolled courses. |
| `has_lab_courses` | Binary | 1 = Enrolled in courses with laboratory units. |
| `lab_courses_count` | Integer | Number of courses with laboratory units. |
| `is_underload` | Binary | 1 if units < 15, 0 otherwise. |
| `is_overload` | Binary | 1 if units > 23, 0 otherwise. |
| `is_top_performer` | Binary | 1 = Listed in institutional Top Performing honor list, 0 = Unlisted. |
| `is_returning_student` | Binary | 1 = Returning student from previous semester, 0 = Freshman/new. |
| `prior_term_gpa` | Float | Prior GPA (median imputed at 2.2 for freshmen/new students). |
| `has_prior_gpa_recorded` | Binary | Indicator flag: 1 if prior GPA exists, 0 if imputed. |
| `prior_deficiencies` | Integer | Prior semester failed/INC/dropped courses count. |
| `prior_at_risk` | Binary | Prior semester at-risk status. |
| **`at_risk`** | Binary | **Target Variable:** 1 = At-Risk, 0 = Not At-Risk (Historical n=380 labeled; Prospective n=356 blank). |
