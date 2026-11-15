# Cleaned, de-identified datasets (not included)

`build_cleaned_pipeline.py` writes the de-identified analysis files here:
`student_risk_ml_dataset.csv`, `student_risk_full_profile.csv`, `student_risk_train_eval.csv`,
`student_risk_prospective_test.csv`, and `course_enrollments_cleaned.csv`. Their columns are documented in
`docs/DATA_DICTIONARY.md`.

The pseudonymization key (`student_id_lookup_private.csv`) is also produced by the pipeline. It is held only by
the ZSPGC Guidance and Scholarship Services Office and must never be shared or committed.
