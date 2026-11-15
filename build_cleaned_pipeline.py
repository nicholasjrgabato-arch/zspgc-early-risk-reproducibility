import os
import csv
import json
from datetime import datetime
from collections import Counter, defaultdict
from cleaning_helpers import clean_municipality, get_distance_km, get_locality_class
from cohort_parser import load_top_performers, parse_cohort

RAW_DATA_DIR = os.path.join('zdspgc-dataset', 'zdspgc data-20261012T163559Z-1-001', 'zdspgc data')
OUTPUT_DIR = 'cleaned_dataset'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_unit(val):
    if not val:
        return 0.0
    val_clean = str(val).replace('(', '').replace(')', '').strip()
    try:
        return float(val_clean)
    except:
        return 0.0

cohort_configs = [
    {
        'cohort_key': '2025-2026_-_Second_Sem_BACHELOR_OF_SCIENCE_IN_BIOLOGY',
        'school_year': '2025-2026',
        'term': '2nd Sem',
        'term_code': '2025-2026_Sem2',
        'program': 'BSBIO',
        'program_title': 'Bachelor of Science in Biology',
        'term_date': datetime(2026, 1, 15),
        'split_type': 'train_eval'
    },
    {
        'cohort_key': '2025-2026_-_Second_Sem_BACHELOR_OF_SCIENCE_IN_INFORMATION_SYSTEMS',
        'school_year': '2025-2026',
        'term': '2nd Sem',
        'term_code': '2025-2026_Sem2',
        'program': 'BSIS',
        'program_title': 'Bachelor of Science in Information Systems',
        'term_date': datetime(2026, 1, 15),
        'split_type': 'train_eval'
    },
    {
        'cohort_key': '2026-2027_-_First_Sem_BACHELOR_OF_SCIENCE_IN_BIOLOGY',
        'school_year': '2026-2027',
        'term': '1st Sem',
        'term_code': '2026-2027_Sem1',
        'program': 'BSBIO',
        'program_title': 'Bachelor of Science in Biology',
        'term_date': datetime(2026, 8, 15),
        'split_type': 'prospective_test'
    },
    {
        'cohort_key': '2026-2027_-_First_Sem_BACHELOR_OF_SCIENCE_IN_INFORMATION_SYSTEMS',
        'school_year': '2026-2027',
        'term': '1st Sem',
        'term_code': '2026-2027_Sem1',
        'program': 'BSIS',
        'program_title': 'Bachelor of Science in Information Systems',
        'term_date': datetime(2026, 8, 15),
        'split_type': 'prospective_test'
    }
]

top_performers = load_top_performers()

student_pid_map = {}
pid_counter = 1
student_history = {} # sid -> prior dict

all_student_records = []
all_course_records = []
private_lookup_records = []

for cfg in cohort_configs:
    sem_str = '2025-2026_-_Second_Sem' if '2025-2026' in cfg['cohort_key'] else '2026-2027_-_First_Sem'
    prog_str = 'BACHELOR_OF_SCIENCE_IN_BIOLOGY' if 'BIOLOGY' in cfg['cohort_key'] else 'BACHELOR_OF_SCIENCE_IN_INFORMATION_SYSTEMS'
    
    enroll_students, promo_students, tes_students = parse_cohort(sem_str, prog_str, cfg['term_date'])
    
    print(f"\nProcessing {cfg['cohort_key']}:")
    print(f"  Enrolled: {len(enroll_students)}, Promo: {len(promo_students)}, TES: {len(tes_students)}")
    
    student_ids = sorted(list(promo_students.keys()))
    
    for sid in student_ids:
        # Assign persistent student_pid
        if sid not in student_pid_map:
            student_pid_map[sid] = f"PID-ZSPGC-{pid_counter:04d}"
            pid_counter += 1
        pid = student_pid_map[sid]
        
        p_st = promo_students[sid]
        e_st = enroll_students.get(sid, {'courses': []})
        t_st = tes_students.get(sid, {})
        
        # 1. Demographic & Socioeconomic features
        bday_str = t_st.get('bday_str', '')
        bday = datetime.strptime(bday_str, '%Y-%m-%d') if bday_str else None
        age = (cfg['term_date'].year - bday.year - ((cfg['term_date'].month, cfg['term_date'].day) < (bday.month, bday.day))) if bday else None
        
        sex_str = p_st.get('sex', '').upper()
        if not sex_str and t_st.get('sex_code') == '1': sex_str = 'F'
        elif not sex_str and t_st.get('sex_code') == '0': sex_str = 'M'
        is_female = 1 if sex_str == 'F' else 0
        
        city_raw = t_st.get('city_muni', '')
        city_clean = clean_municipality(city_raw)
        distance_km = get_distance_km(city_clean)
        locality_class = get_locality_class(distance_km)
        province_clean = t_st.get('province', 'ZAMBOANGA DEL SUR').strip().upper()
        zipcode = t_st.get('zipcode', '7020').strip()
        
        four_ps_raw = t_st.get('four_ps_id', '').strip()
        is_4ps_beneficiary = 1 if (four_ps_raw and four_ps_raw != '0' and four_ps_raw.lower() != 'none') else 0
        income_bracket = 1 if is_4ps_beneficiary == 1 else 2
        
        philsys_raw = t_st.get('philsys_id', '').strip()
        has_philsys_id = 1 if philsys_raw else 0
        
        is_first_gen = 1 if t_st.get('first_gen', '0').strip() == '1' else 0
        is_dependent = 1 if t_st.get('dependent', '0').strip() == '1' else 0
        
        disability_raw = t_st.get('disability', '').strip()
        has_disability = 1 if disability_raw else 0
        disability_type = disability_raw if disability_raw else 'None'
        
        ip_raw = t_st.get('ip_group', '').strip()
        is_indigenous = 1 if ip_raw else 0
        ip_group_name = ip_raw if ip_raw else 'None'
        
        year_level_raw = p_st.get('year', t_st.get('year_level', '1')).strip()
        try:
            year_level = int(float(year_level_raw))
        except:
            year_level = 1
            
        # 2. Academic Features from Enrollment & Promotional
        courses_promo = p_st.get('courses', [])
        courses_enroll = e_st.get('courses', [])
        
        course_count = len(courses_promo)
        
        total_units = parse_unit(p_st.get('units') or e_st.get('units') or 0.0)
        if total_units == 0.0:
            total_units = sum(parse_unit(c.get('unit', 0)) for c in courses_promo)
            
        lec_units_total = 0.0
        lab_units_total = 0.0
        lab_courses_count = 0
        
        for idx_c, cp in enumerate(courses_promo):
            ce = courses_enroll[idx_c] if idx_c < len(courses_enroll) else {}
            u_lec = parse_unit(ce.get('lec_units', 0))
            u_lab = parse_unit(ce.get('lab_units', 0))
            u_tot = parse_unit(cp.get('unit')) or (u_lec + u_lab) or 0.0
            if u_lab > 0:
                lab_courses_count += 1
            lec_units_total += u_lec
            lab_units_total += u_lab
            
            grade_val = cp.get('grade', '').strip()
            rem_val = cp.get('remarks', '').strip()
            
            grade_num = None
            try:
                grade_num = float(grade_val)
            except:
                pass
                
            all_course_records.append({
                'student_pid': pid,
                'school_year': cfg['school_year'],
                'term': cfg['term'],
                'term_code': cfg['term_code'],
                'program': cfg['program'],
                'year_level': year_level,
                'course_index': idx_c + 1,
                'subject_code': cp.get('code', '').strip(),
                'subject_title': cp.get('title', '').strip(),
                'total_units': u_tot,
                'lecture_units': u_lec,
                'lab_units': u_lab,
                'control_no': ce.get('control_no', '').strip(),
                'grade_raw': grade_val,
                'grade_num': grade_num if grade_num is not None else '',
                'remarks': rem_val if rem_val else ('In Progress' if cfg['split_type'] == 'prospective_test' else 'None')
            })
            
        has_lab_courses = 1 if lab_courses_count > 0 else 0
        
        if total_units < 15:
            academic_load_status = 'Underload (<15)'
        elif total_units <= 23:
            academic_load_status = 'Normal Load (15-23)'
        else:
            academic_load_status = 'Overload (>23)'
            
        top_key = (sem_str, sid)
        is_top_performer = 1 if top_key in top_performers else 0
        honor_rank = top_performers[top_key]['rank'] if is_top_performer else None
        honor_gpa = top_performers[top_key]['gpa'] if is_top_performer else None
        
        is_returning_student = 1 if sid in student_history else 0
        if is_returning_student:
            prev_hist = student_history[sid]
            prior_term_gpa = prev_hist.get('term_gpa')
            prior_deficiencies = prev_hist.get('deficiencies_count', 0)
            prior_at_risk = prev_hist.get('at_risk', 0)
        else:
            prior_term_gpa = None
            prior_deficiencies = 0
            prior_at_risk = 0
            
        # 3. Target Variable & Outcome Assessment
        if cfg['split_type'] == 'train_eval':
            failed_courses = sum(1 for c in courses_promo if c.get('remarks', '').lower() == 'failed' or c.get('grade') in ('5.0', '5'))
            inc_courses = sum(1 for c in courses_promo if 'inc' in c.get('remarks', '').lower())
            dropped_courses = sum(1 for c in courses_promo if 'drop' in c.get('remarks', '').lower())
            passed_courses = sum(1 for c in courses_promo if c.get('remarks', '').lower() == 'passed')
            
            ga_str = (p_st.get('ga') or '').strip()
            try:
                term_gpa = float(ga_str)
            except:
                term_gpa = None
                
            deficiencies_count = failed_courses + inc_courses + dropped_courses
            
            if deficiencies_count > 0 or (term_gpa is not None and term_gpa > 3.0):
                at_risk = 1
            else:
                at_risk = 0
                
            if at_risk == 0:
                standing_category = 'Satisfactory'
            elif failed_courses > 0:
                standing_category = 'Probation - Failed Course'
            elif inc_courses > 0:
                standing_category = 'Deficiency - Incomplete (INC)'
            elif dropped_courses > 0:
                standing_category = 'Attrition - Dropped Course'
            else:
                standing_category = 'Probation - Low GPA'
                
            passed_ratio = round(passed_courses / course_count, 4) if course_count > 0 else 0.0
            
            student_history[sid] = {
                'term_gpa': term_gpa,
                'deficiencies_count': deficiencies_count,
                'at_risk': at_risk
            }
        else:
            failed_courses = None
            inc_courses = None
            dropped_courses = None
            passed_courses = None
            term_gpa = None
            deficiencies_count = None
            at_risk = None
            standing_category = 'Pending Midterm/Final Assessment'
            passed_ratio = None
            
        st_record = {
            'student_pid': pid,
            'school_year': cfg['school_year'],
            'term': cfg['term'],
            'term_code': cfg['term_code'],
            'split_type': cfg['split_type'],
            'program': cfg['program'],
            'program_title': cfg['program_title'],
            'year_level': year_level,
            'age': age,
            'sex': sex_str,
            'is_female': is_female,
            'residence_municipality': city_clean,
            'residence_province': province_clean,
            'zipcode': zipcode,
            'distance_km': distance_km,
            'locality_class': locality_class,
            'is_4ps_beneficiary': is_4ps_beneficiary,
            'income_bracket': income_bracket,
            'has_philsys_id': has_philsys_id,
            'is_first_gen_college': is_first_gen,
            'is_dependent': is_dependent,
            'has_disability': has_disability,
            'disability_type': disability_type,
            'is_indigenous': is_indigenous,
            'indigenous_group': ip_group_name,
            'tes_grantee_flag': 1,
            'enrolled_units': total_units,
            'course_count': course_count,
            'has_lab_courses': has_lab_courses,
            'lab_courses_count': lab_courses_count,
            'academic_load_status': academic_load_status,
            'is_top_performer': is_top_performer,
            'honor_rank': honor_rank if honor_rank is not None else '',
            'is_returning_student': is_returning_student,
            'prior_term_gpa': prior_term_gpa if prior_term_gpa is not None else '',
            'prior_deficiencies': prior_deficiencies,
            'prior_at_risk': prior_at_risk,
            # Target and outcome fields
            'passed_courses_count': passed_courses if passed_courses is not None else '',
            'failed_courses_count': failed_courses if failed_courses is not None else '',
            'inc_courses_count': inc_courses if inc_courses is not None else '',
            'dropped_courses_count': dropped_courses if dropped_courses is not None else '',
            'passed_courses_ratio': passed_ratio if passed_ratio is not None else '',
            'term_gpa': term_gpa if term_gpa is not None else '',
            'academic_standing': standing_category,
            'at_risk': at_risk if at_risk is not None else ''
        }
        all_student_records.append(st_record)
        
        private_lookup_records.append({
            'student_pid': pid,
            'school_year': cfg['school_year'],
            'term': cfg['term'],
            'program': cfg['program'],
            'original_student_number': sid,
            'student_name': p_st.get('name', ''),
            'last_name': t_st.get('last_name', ''),
            'first_name': t_st.get('first_name', ''),
            'middle_name': t_st.get('middle_name', ''),
            'lrn': t_st.get('lrn', ''),
            'philsys_id': t_st.get('philsys_id', ''),
            'four_ps_id': t_st.get('four_ps_id', ''),
            'contact_number': t_st.get('contact_no', ''),
            'email': t_st.get('email', ''),
            'permanent_address': t_st.get('street_brgy', ''),
            'father_name': t_st.get('father_name', ''),
            'mother_name': t_st.get('mother_name', '')
        })

print(f"\n==========================================")
print(f"Total Student-Term Observations: {len(all_student_records)}")
print(f"Unique Individual Students: {len(student_pid_map)}")
print(f"Total Course Enrollments: {len(all_course_records)}")

def write_csv(filepath, rows):
    if not rows: return
    headers = list(rows[0].keys())
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {filepath} ({len(rows)} records, {len(headers)} columns)")

# Full profile
write_csv(os.path.join(OUTPUT_DIR, 'student_risk_full_profile.csv'), all_student_records)

# Granular courses
write_csv(os.path.join(OUTPUT_DIR, 'course_enrollments_cleaned.csv'), all_course_records)

# Private de-identification key
write_csv(os.path.join(OUTPUT_DIR, 'student_id_lookup_private.csv'), private_lookup_records)

# Historical training / evaluation split (n=380)
train_eval_records = [r for r in all_student_records if r['split_type'] == 'train_eval']
write_csv(os.path.join(OUTPUT_DIR, 'student_risk_train_eval.csv'), train_eval_records)

# Prospective holdout test split (n=356)
prospective_test_records = [r for r in all_student_records if r['split_type'] == 'prospective_test']
write_csv(os.path.join(OUTPUT_DIR, 'student_risk_prospective_test.csv'), prospective_test_records)

# Machine Learning Feature Matrix (Numeric / Encoded)
ml_features = []
for r in all_student_records:
    ml_row = {
        'student_pid': r['student_pid'],
        'school_year': r['school_year'],
        'term': r['term'],
        'split_type': r['split_type'],
        # Categorical / Ordinal Predictors
        'program_is_bsis': 1 if r['program'] == 'BSIS' else 0,
        'year_level': r['year_level'],
        'age': r['age'],
        'is_female': r['is_female'],
        'distance_km': r['distance_km'],
        'locality_is_campus_area': 1 if r['distance_km'] <= 5.0 else 0,
        'locality_is_distant': 1 if r['distance_km'] > 25.0 else 0,
        'is_4ps_beneficiary': r['is_4ps_beneficiary'],
        'income_bracket': r['income_bracket'],
        'has_philsys_id': r['has_philsys_id'],
        'is_first_gen_college': r['is_first_gen_college'],
        'is_dependent': r['is_dependent'],
        'has_disability': r['has_disability'],
        'is_indigenous': r['is_indigenous'],
        'enrolled_units': r['enrolled_units'],
        'course_count': r['course_count'],
        'has_lab_courses': r['has_lab_courses'],
        'lab_courses_count': r['lab_courses_count'],
        'is_underload': 1 if r['enrolled_units'] < 15 else 0,
        'is_overload': 1 if r['enrolled_units'] > 23 else 0,
        'is_top_performer': r['is_top_performer'],
        'is_returning_student': r['is_returning_student'],
        'prior_term_gpa': r['prior_term_gpa'] if r['prior_term_gpa'] != '' else 2.2, # median imputation for new students
        'has_prior_gpa_recorded': 1 if r['prior_term_gpa'] != '' else 0,
        'prior_deficiencies': r['prior_deficiencies'],
        'prior_at_risk': r['prior_at_risk'],
        # Target
        'at_risk': r['at_risk']
    }
    ml_features.append(ml_row)

write_csv(os.path.join(OUTPUT_DIR, 'student_risk_ml_dataset.csv'), ml_features)
print("\nPipeline completed successfully!")
