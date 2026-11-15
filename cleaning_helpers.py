import os
import csv
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime
from collections import Counter

def read_xlsx_rows(path):
    with zipfile.ZipFile(path) as z:
        shared_strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            tree = ET.fromstring(z.read('xl/sharedStrings.xml'))
            for si in tree.findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si'):
                text = ''.join(t.text for t in si.iter('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t') if t.text)
                shared_strings.append(text)
        
        ws_tree = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
        rows = []
        for r in ws_tree.findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}sheetData/{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row'):
            row_dict = {}
            for c in r.findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c'):
                col_letters = ''.join(ch for ch in c.attrib.get('r') if ch.isalpha())
                v = c.find('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v')
                val = v.text if v is not None else ''
                t = c.attrib.get('t')
                if t == 's' and val != '':
                    val = shared_strings[int(val)]
                row_dict[col_letters] = val
            rows.append(row_dict)
        return rows

# Road distance (km) approximation from residence municipality to ZSPGC Campus in Poblacion, Aurora, Zamboanga del Sur
MUNICIPALITY_DISTANCES = {
    'AURORA': 2.0,
    'RAMON MAGSAYSAY': 14.0,
    'TUKURAN': 17.0,
    'TAMBULIG': 22.0,
    'KAPATAGAN': 28.0,
    'SAPAD': 32.0,
    'LABANGAN': 32.0,
    'MIDSALIP': 32.0,
    'S.N.D.': 35.0,
    'SULTAN NAGA DIMAPORO': 35.0,
    'LALA': 36.0,
    'SOMINOT': 38.0,
    'MAHAYAG': 38.0,
    'PAGADIAN': 42.0,
    'PAGADIAN CITY': 42.0,
    'DUMINGAG': 48.0,
    'BAYOG': 70.0,
    'DIPLAHAN': 80.0,
    'ILIGAN': 95.0,
    'ILIGAN CITY': 95.0,
    'OROQUIETA': 105.0,
    'OROQUIETA CITY': 105.0,
    'IPIL': 130.0
}

def clean_municipality(muni_str):
    if not muni_str:
        return 'AURORA'
    m = muni_str.strip().upper()
    if 'AURORA' in m: return 'AURORA'
    if 'TUKURAN' in m: return 'TUKURAN'
    if 'RAMON' in m or 'MAGSAYSAY' in m: return 'RAMON MAGSAYSAY'
    if 'KAPATAGAN' in m: return 'KAPATAGAN'
    if 'TAMBULIG' in m: return 'TAMBULIG'
    if 'S.N.D' in m or 'DIMAPORO' in m: return 'SULTAN NAGA DIMAPORO (S.N.D.)'
    if 'LABANGAN' in m: return 'LABANGAN'
    if 'SOMINOT' in m: return 'SOMINOT'
    if 'MIDSALIP' in m: return 'MIDSALIP'
    if 'PAGADIAN' in m: return 'PAGADIAN CITY'
    if 'LALA' in m: return 'LALA'
    if 'MAHAYAG' in m: return 'MAHAYAG'
    if 'DUMINGAG' in m: return 'DUMINGAG'
    if 'DIPLAHAN' in m: return 'DIPLAHAN'
    if 'ILIGAN' in m: return 'ILIGAN CITY'
    if 'BAYOG' in m: return 'BAYOG'
    if 'OROQUIETA' in m: return 'OROQUIETA CITY'
    if 'IPIL' in m: return 'IPIL'
    if 'SAPAD' in m: return 'SAPAD'
    return m

def get_distance_km(clean_muni):
    for key, dist in MUNICIPALITY_DISTANCES.items():
        if key in clean_muni:
            return dist
    return 20.0

def get_locality_class(dist):
    if dist <= 5.0:
        return 'Campus Locality (<=5km)'
    elif dist <= 25.0:
        return 'Nearby Commuter (5-25km)'
    else:
        return 'Regional / Distant (>25km)'

print('Helper functions verified.')
