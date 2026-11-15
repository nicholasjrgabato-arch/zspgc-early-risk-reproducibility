import os
import csv
import json
from datetime import datetime
from collections import Counter
from cleaning_helpers import (
    read_xlsx_rows, clean_municipality, get_distance_km, get_locality_class
)

RAW_DATA_DIR = os.path.join('zdspgc-dataset', 'zdspgc data-20261012T163559Z-1-001', 'zdspgc data')
OUTPUT_DIR = 'cleaned_dataset'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load Top Performing Students (Honor lists)
def load_top_performers():
    top_dict = {}
    # 2025-2026 Sem 2
    f1 = os.path.join(RAW_DATA_DIR, 'Top_Performing_Students_2025-2026_-_Second_Sem_BACHELOR_OF_SCIENCE_IN_INFORMATION_SYSTEMS.xlsx')
    rows1 = read_xlsx_rows(f1)
    for r in rows1[7:]:
        sid = (r.get('C') or '').strip()
        if sid and sid != 'Student No.' and not sid.startswith('*'):
            try:
                rank_val = int(float((r.get('A') or '0').strip()))
            except:
                rank_val = None
            try:
                gpa_val = float((r.get('H') or '0').strip())
            except:
                gpa_val = None
            top_dict[('2025-2026_-_Second_Sem', sid)] = {
                'rank': rank_val,
                'gpa': gpa_val,
                'program': (r.get('G') or '').strip()
            }
            
    # 2026-2027 Sem 1
    f2 = os.path.join(RAW_DATA_DIR, 'Top_Performing_Students_2026-2027_-_First_Sem_BACHELOR_OF_SCIENCE_IN_INFORMATION_SYSTEMS.xlsx')
    rows2 = read_xlsx_rows(f2)
    for r in rows2[7:]:
        sid = (r.get('C') or '').strip()
        if sid and sid != 'Student No.' and not sid.startswith('*'):
            try:
                rank_val = int(float((r.get('A') or '0').strip()))
            except:
                rank_val = None
            try:
                gpa_val = float((r.get('H') or '0').strip())
            except:
                gpa_val = None
            top_dict[('2026-2027_-_First_Sem', sid)] = {
                'rank': rank_val,
                'gpa': gpa_val,
                'program': (r.get('G') or '').strip()
            }
    return top_dict

# 2. Parse Cohort Data
def parse_cohort(sem_str, prog_str, term_start_date):
    enroll_f = [f for f in os.listdir(RAW_DATA_DIR) if f.startswith('Enrollment') and sem_str in f and prog_str in f][0]
    promo_f = [f for f in os.listdir(RAW_DATA_DIR) if f.startswith('Promotional') and sem_str in f and prog_str in f][0]
    tes_f = [f for f in os.listdir(RAW_DATA_DIR) if f.startswith('TES') and sem_str in f and prog_str in f][0]
    
    # A. Parse Enrollment Report for subject lecture/lab breakdown & control numbers
    enroll_rows = read_xlsx_rows(os.path.join(RAW_DATA_DIR, enroll_f))
    enroll_students = {}
    curr_e = None
    for r in enroll_rows[8:]:
        b = (r.get('B') or '').strip()
        c = (r.get('C') or '').strip()
        g = (r.get('G') or '').strip()
        f_code = (r.get('F') or '').strip()
        if b and c and b != 'Student Number':
            if curr_e: enroll_students[curr_e['id']] = curr_e
            curr_e = {
                'id': b, 'name': c, 'sex': (r.get('D') or '').strip(), 'year': (r.get('E') or '').strip(),
                'courses': [], 'units': (r.get('H') or '').strip()
            }
        elif curr_e and g.startswith('Total No.'):
            curr_e['units'] = (r.get('H') or '').strip()
        elif curr_e and (f_code or g):
            curr_e['courses'].append({
                'code': f_code,
                'title': g,
                'lec_units': (r.get('H') or '').strip(),
                'lab_units': (r.get('I') or '').strip(),
                'control_no': (r.get('J') or '').strip()
            })
    if curr_e: enroll_students[curr_e['id']] = curr_e
    
    # B. Parse Promotional Report for subjects, grades, remarks, GA
    promo_rows = read_xlsx_rows(os.path.join(RAW_DATA_DIR, promo_f))
    promo_students = {}
    curr_p = None
    for r in promo_rows[8:]:
        b = (r.get('B') or '').strip()
        c = (r.get('C') or '').strip()
        g = (r.get('G') or '').strip()
        f_code = (r.get('F') or '').strip()
        if b and c and b != 'Student Number':
            if curr_p: promo_students[curr_p['id']] = curr_p
            curr_p = {
                'id': b, 'name': c, 'sex': (r.get('D') or '').strip(), 'year': (r.get('E') or '').strip(),
                'courses': [], 'ga': None, 'units': None
            }
        elif curr_p and g.startswith('Total No.'):
            curr_p['units'] = (r.get('H') or '').strip()
            curr_p['ga'] = (r.get('J') or '').strip()
        elif curr_p and (f_code or g):
            curr_p['courses'].append({
                'code': f_code,
                'title': g,
                'unit': (r.get('H') or '').strip(),
                'grade': (r.get('I') or '').strip(),
                'remarks': (r.get('J') or '').strip()
            })
    if curr_p: promo_students[curr_p['id']] = curr_p
    
    # C. Parse TES Application Data
    tes_rows = read_xlsx_rows(os.path.join(RAW_DATA_DIR, tes_f))
    hdr_idx = None
    for idx, r in enumerate(tes_rows[:20]):
        if any('STUDENT ID' in str(v) for v in r.values()):
            hdr_idx = idx
            break
    tes_students = {}
    for r in tes_rows[hdr_idx+2:]:
        sid = (r.get('B') or '').strip()
        if not sid: continue
        tes_students[sid] = {
            'seq': (r.get('A') or '').strip(),
            'id': sid,
            'lrn': (r.get('C') or '').strip(),
            'philsys_id': (r.get('D') or '').strip(),
            'four_ps_id': (r.get('E') or '').strip(),
            'last_name': (r.get('F') or '').strip(),
            'first_name': (r.get('G') or '').strip(),
            'ext_name': (r.get('H') or '').strip(),
            'middle_name': (r.get('I') or '').strip(),
            'sex_code': (r.get('J') or '').strip(),
            'bday_str': (r.get('K') or '').strip(),
            'program_name': (r.get('L') or '').strip(),
            'year_level': (r.get('M') or '').strip(),
            'father_name': f"{(r.get('N') or '').strip()} {(r.get('O') or '').strip()} {(r.get('P') or '').strip()}".strip(),
            'mother_name': f"{(r.get('Q') or '').strip()} {(r.get('R') or '').strip()} {(r.get('S') or '').strip()}".strip(),
            'street_brgy': (r.get('T') or '').strip(),
            'city_muni': (r.get('U') or '').strip(),
            'province': (r.get('V') or '').strip(),
            'region': (r.get('W') or '').strip(),
            'zipcode': (r.get('X') or '').strip(),
            'contact_no': (r.get('Y') or '').strip(),
            'email': (r.get('Z') or '').strip(),
            'disability': (r.get('AA') or '').strip(),
            'dependent': (r.get('AB') or '').strip(),
            'first_gen': (r.get('AC') or '').strip(),
            'ip_group': (r.get('AD') or '').strip(),
        }
        
    return enroll_students, promo_students, tes_students

print('Cohort parsing logic defined.')
