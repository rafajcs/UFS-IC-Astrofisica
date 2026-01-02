# -*- coding: utf-8 -*-
"""
CRIAÇÃO DO CATÁLOGO PRIMÁRIO DE ANÃS BRANCAS
============================================
Catálogo final com 18.369 WDs com fotometria sintética S-PLUS
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table, Column
from astropy import units as u
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CRIAÇÃO DO CATÁLOGO PRIMÁRIO DE ANÃS BRANCAS")
print("="*80)
print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ==================== PASSO 1: CARREGAR CONVOLUÇÃO ====================

print("PASSO 1: Carregando resultados da convolução")
print("-"*80)

conv_file = "convolution_results.csv"
df_conv = pd.read_csv(conv_file)

print(f"✅ Convoluções carregadas: {len(df_conv):,} objetos")
print(f"   Colunas: {len(df_conv.columns)}")
print(f"   Filtros S-PLUS: {[c for c in df_conv.columns if '_mag' in c and '_err' not in c]}\n")

# ==================== PASSO 2: ENCONTRAR CATÁLOGO UNIFICADO ====================

print("PASSO 2: Procurando catálogo unificado")
print("-"*80)

import glob
import os

# Procurar catálogos possíveis
patterns = [
    "pipeline_*/checkpoints/step1_union.csv",
    "unified_catalog_*/SDSS_Gaia_unified_catalog.csv",
    "final_catalog_*/catalog_complete.csv",
    "*union*.csv",
    "*unified*.csv"
]

found_catalogs = []
for pattern in patterns:
    found_catalogs.extend(glob.glob(pattern))

if not found_catalogs:
    print("⚠️  Nenhum catálogo unificado encontrado automaticamente.\n")
    print("💡 Especifique manualmente:")
    catalog_path = input("   Caminho do catálogo unificado: ").strip()
else:
    print(f"📂 Catálogos encontrados:\n")
    for i, cat in enumerate(found_catalogs[:10]):
        size_mb = os.path.getsize(cat) / (1024**2)
        mtime = datetime.fromtimestamp(os.path.getmtime(cat))
        print(f"   [{i}] {cat}")
        print(f"       {size_mb:.1f} MB | {mtime.strftime('%Y-%m-%d %H:%M')}")
    
    if len(found_catalogs) > 10:
        print(f"   ... e mais {len(found_catalogs)-10}")
    
    print("\n   [m] Especificar manualmente")
    
    choice = input("\nEscolha: ").strip()
    
    if choice == 'm':
        catalog_path = input("Caminho: ").strip()
    else:
        catalog_path = found_catalogs[int(choice)]

print(f"\n📂 Carregando: {catalog_path}")
df_catalog = pd.read_csv(catalog_path)

print(f"✅ Catálogo unificado carregado: {len(df_catalog):,} objetos")
print(f"   Colunas: {len(df_catalog.columns)}\n")

# ==================== PASSO 3: FAZER MATCH ====================

print("PASSO 3: Associando convolução com catálogo")
print("-"*80)

# Identificar colunas de match
match_cols_conv = ['plate', 'mjd', 'fiberid']
match_cols_cat = None

# Procurar colunas correspondentes no catálogo
possible_matches = {
    'plate': ['plate', 'Plate', 'PLATE'],
    'mjd': ['mjd', 'MJD'],
    'fiber': ['fiberid', 'fiber', 'Fiber', 'FIBER']
}

match_cols_cat = []
for key in ['plate', 'mjd', 'fiber']:
    found = None
    for candidate in possible_matches[key]:
        if candidate in df_catalog.columns:
            found = candidate
            break
    if found:
        match_cols_cat.append(found)
        print(f"   ✓ Match: {key:10s} → {found}")
    else:
        print(f"   ✗ Não encontrado: {key}")

if len(match_cols_cat) != 3:
    print("\n❌ ERRO: Colunas de match incompletas!")
    print("   Necessário: plate, mjd, fiber/fiberid\n")
    exit(1)

# Renomear para padronizar
rename_map = dict(zip(match_cols_cat, ['plate', 'mjd', 'fiber']))
df_catalog_renamed = df_catalog.rename(columns=rename_map)



# Fazer merge
print(f"\n🔗 Fazendo merge...")

df_merged = df_conv.merge(
    df_catalog_renamed,
    left_on=['plate', 'mjd', 'fiberid'],
    right_on=['plate', 'mjd', 'fiber'],
    how='left',
    suffixes=('_conv', '_cat')
)

if 'fiberid' not in df_merged.columns and 'fiber' in df_merged.columns:
    df_merged['fiberid'] = df_merged['fiber']


print(f"✅ Match concluído: {len(df_merged):,} objetos")

# Verificar quantos encontraram match
# matched = df_merged['ra'].notna().sum() if 'ra' in df_merged.columns else 0

ra_cols = ['ra', 'RA', 'RAdeg', 'RA_ICRS']
for col in ra_cols:
    if col in df_merged.columns:
        matched = df_merged[col].notna().sum()
        break
else:
    matched = 0


print(f"   Com match no catálogo: {matched:,} ({100*matched/len(df_merged):.1f}%)")
print(f"   Sem match: {len(df_merged) - matched:,}\n")

# ==================== PASSO 4: COMPILAR CATÁLOGO FINAL ====================

print("PASSO 4: Compilando catálogo final")
print("-"*80)

# Criar catálogo estruturado
catalog_data = {}

# 1. Identificação
print("   📋 Identificação...")

# catalog_data['object_id'] = df_merged['object_name'].values

for col in ['object_name', 'SDSS', 'specObjID', 'objid']:
    if col in df_merged.columns:
        catalog_data['object_id'] = df_merged[col].astype(str).values
        break
else:
    catalog_data['object_id'] = np.array([
        f"spec-{p}-{m}-{f}" 
        for p, m, f in zip(df_merged['plate'], df_merged['mjd'], df_merged['fiberid'])
    ])

catalog_data['plate'] = df_merged['plate'].astype('int32').values

catalog_data['mjd'] = df_merged['mjd'].astype('int32').values

catalog_data['fiberid'] = df_merged['fiberid'].astype('int16').values

# 2. Coordenadas
print("   🌐 Coordenadas...")
if 'ra' in df_merged.columns and 'dec' in df_merged.columns:
    catalog_data['ra'] = df_merged['ra'].astype('float64').values
    catalog_data['dec'] = df_merged['dec'].astype('float64').values
elif 'RA' in df_merged.columns and 'Dec' in df_merged.columns:
    catalog_data['ra'] = df_merged['RA'].astype('float64').values
    catalog_data['dec'] = df_merged['Dec'].astype('float64').values
else:
    print("   ⚠️  Coordenadas não encontradas!")
    catalog_data['ra'] = np.full(len(df_merged), np.nan)
    catalog_data['dec'] = np.full(len(df_merged), np.nan)

# 3. Tipo espectral de WD
print("   🔬 Tipos espectrais...")
type_cols = ['Type', 'wd_type', 'wd_subclass', 'spec_subclass', 'subclass']
for col in type_cols:
    if col in df_merged.columns:
        catalog_data['wd_type'] = df_merged[col].fillna('Unknown').astype(str).values
        print(f"      Usando: {col}")
        break
else:
    catalog_data['wd_type'] = np.array(['Unknown'] * len(df_merged))

# 4. Redshift
print("   🌌 Redshift...")
if 'z' in df_merged.columns:
    catalog_data['redshift'] = df_merged['z'].astype('float32').values
else:
    catalog_data['redshift'] = np.full(len(df_merged), np.nan, dtype='float32')

# 5. Parâmetros físicos
print("   ⚛️  Parâmetros físicos...")

# Teff
for col in ['Teff', 'teff', 'teff_H']:
    if col in df_merged.columns:
        catalog_data['teff'] = df_merged[col].astype('float32').values
        break
else:
    catalog_data['teff'] = np.full(len(df_merged), np.nan, dtype='float32')

# log g
for col in ['logg', 'log_g', 'logg_H']:
    if col in df_merged.columns:
        catalog_data['logg'] = df_merged[col].astype('float32').values
        break
else:
    catalog_data['logg'] = np.full(len(df_merged), np.nan, dtype='float32')

# Massa
for col in ['Mass', 'mass', 'mass_H']:
    if col in df_merged.columns:
        catalog_data['mass'] = df_merged[col].astype('float32').values
        break
else:
    catalog_data['mass'] = np.full(len(df_merged), np.nan, dtype='float32')

# 6. Fotometria SDSS original
print("   📷 Fotometria SDSS...")
sdss_bands = ['u', 'g', 'r', 'i', 'z']
for band in sdss_bands:
    col_name = f'{band}mag'
    if col_name in df_merged.columns:
        catalog_data[f'sdss_{band}'] = df_merged[col_name].astype('float32').values
        
        # Erro se disponível
        err_col = f'e_{band}mag'
        if err_col in df_merged.columns:
            catalog_data[f'sdss_{band}_err'] = df_merged[err_col].astype('float32').values

# 7. Fotometria Gaia
print("   🛰️  Fotometria Gaia...")
gaia_bands = {
    'gaia_G': ['gaia_g_mean_mag', 'gaia_G', 'phot_g_mean_mag'],
    'gaia_BP': ['gaia_bp_mean_mag', 'gaia_BP', 'phot_bp_mean_mag'],
    'gaia_RP': ['gaia_rp_mean_mag', 'gaia_RP', 'phot_rp_mean_mag']
}

for target, candidates in gaia_bands.items():
    for col in candidates:
        if col in df_merged.columns:
            catalog_data[target] = df_merged[col].astype('float32').values
            break

# Parallax
for col in ['parallax', 'gaia_parallax']:
    if col in df_merged.columns:
        catalog_data['parallax'] = df_merged[col].astype('float32').values
        break

# 8. Fotometria S-PLUS sintética (DA CONVOLUÇÃO!)
print("   🎨 Fotometria S-PLUS sintética...")
splus_filters = ['uJAVA', 'J0378', 'J0395', 'J0410', 'J0430', 'gSDSS', 
                 'J0515', 'rSDSS', 'J0660', 'iSDSS', 'J0861', 'zSDSS']

for filt in splus_filters:
    mag_col = f'{filt}_mag'
    err_col = f'{filt}_err'
    
    if mag_col in df_merged.columns:
        catalog_data[f'splus_{filt}'] = df_merged[mag_col].astype('float32').values
    
    if err_col in df_merged.columns:
        catalog_data[f'splus_{filt}_err'] = df_merged[err_col].astype('float32').values

# 9. Fonte do catálogo
print("   📚 Fonte...")
if 'catalog_source' in df_merged.columns:
    catalog_data['catalog_source'] = df_merged['catalog_source'].fillna('SDSS_DR7').astype(str).values
else:
    catalog_data['catalog_source'] = np.array(['SDSS_DR7'] * len(df_merged))

print(f"\n✅ Catálogo compilado com {len(catalog_data)} campos\n")

# ==================== PASSO 5: CRIAR ASTROPY TABLE ====================

print("PASSO 5: Criando Astropy Table")
print("-"*80)

# Criar Table
table = Table()

# Adicionar colunas com metadados
print("   Adicionando colunas com metadados...")

# Identificação
table['object_id'] = Column(catalog_data['object_id'], description='Object identifier')
table['plate'] = Column(catalog_data['plate'], description='SDSS Plate number')
table['mjd'] = Column(catalog_data['mjd'], description='Modified Julian Date')
table['fiberid'] = Column(catalog_data['fiberid'], description='Fiber ID')

# Coordenadas
table['ra'] = Column(catalog_data['ra'], unit=u.degree, description='Right Ascension (J2000)')
table['dec'] = Column(catalog_data['dec'], unit=u.degree, description='Declination (J2000)')

# Classificação
table['wd_type'] = Column(catalog_data['wd_type'], description='White dwarf spectral type')
table['redshift'] = Column(catalog_data['redshift'], description='Spectroscopic redshift')

# Parâmetros físicos
table['teff'] = Column(catalog_data['teff'], unit=u.K, description='Effective temperature')
table['logg'] = Column(catalog_data['logg'], description='Surface gravity (log g)')
table['mass'] = Column(catalog_data['mass'], unit=u.solMass, description='Mass')

# Fotometria SDSS
for band in sdss_bands:
    key = f'sdss_{band}'
    if key in catalog_data:
        table[key] = Column(catalog_data[key], unit=u.mag, description=f'SDSS {band}-band magnitude')
        
        err_key = f'{key}_err'
        if err_key in catalog_data:
            table[err_key] = Column(catalog_data[err_key], unit=u.mag, description=f'SDSS {band}-band error')

# Fotometria Gaia
for band in ['gaia_G', 'gaia_BP', 'gaia_RP']:
    if band in catalog_data:
        table[band] = Column(catalog_data[band], unit=u.mag, description=f'{band} magnitude')

if 'parallax' in catalog_data:
    table['parallax'] = Column(catalog_data['parallax'], unit=u.mas, description='Gaia parallax')

# Fotometria S-PLUS sintética
for filt in splus_filters:
    key = f'splus_{filt}'
    if key in catalog_data:
        table[key] = Column(catalog_data[key], unit=u.mag, 
                           description=f'S-PLUS {filt} synthetic magnitude')
        
        err_key = f'{key}_err'
        if err_key in catalog_data:
            table[err_key] = Column(catalog_data[err_key], unit=u.mag,
                                   description=f'S-PLUS {filt} error')

# Fonte
table['catalog_source'] = Column(catalog_data['catalog_source'], description='Source catalog')

print(f"✅ Table criada com {len(table.colnames)} colunas\n")

# ==================== PASSO 6: ADICIONAR METADADOS ====================

print("PASSO 6: Adicionando metadados do catálogo")
print("-"*80)

table.meta['CATALOG'] = 'Primary White Dwarf Catalog'
table.meta['VERSION'] = '1.0'
table.meta['DATE'] = datetime.now().strftime('%Y-%m-%d')
table.meta['AUTHOR'] = 'Your Name'  # AJUSTAR
table.meta['INSTITUTE'] = 'UFS'
table.meta['N_OBJECTS'] = len(table)
table.meta['DESCRIPTION'] = 'White dwarf catalog with S-PLUS synthetic photometry'
table.meta['SOURCES'] = 'SDSS DR7, Gaia EDR3, LAMOST DR8'
table.meta['CONVOLUTION'] = 'S-PLUS 12-band synthetic magnitudes from spectra'
table.meta['FILTERS'] = 'uJAVA, J0378, J0395, J0410, J0430, gSDSS, J0515, rSDSS, J0660, iSDSS, J0861, zSDSS'

print("✅ Metadados adicionados\n")

# ==================== PASSO 7: SALVAR FITS ====================

print("PASSO 7: Salvando catálogo FITS")
print("-"*80)

output_file = f"WD_primary_catalog_{datetime.now().strftime('%Y%m%d')}.fits"

table.write(output_file, format='fits', overwrite=True)

print(f"✅ Catálogo salvo: {output_file}")
print(f"   Tamanho: {os.path.getsize(output_file) / (1024**2):.1f} MB\n")

# ==================== ESTATÍSTICAS FINAIS ====================

print("="*80)
print("ESTATÍSTICAS DO CATÁLOGO FINAL")
print("="*80)

print(f"\n📊 COBERTURA:")
print(f"   Total de objetos:     {len(table):,}")
print(f"   Com coordenadas:      {np.isfinite(table['ra']).sum():,} ({100*np.isfinite(table['ra']).sum()/len(table):.1f}%)")
print(f"   Com Teff:             {np.isfinite(table['teff']).sum():,} ({100*np.isfinite(table['teff']).sum()/len(table):.1f}%)")
print(f"   Com log g:            {np.isfinite(table['logg']).sum():,} ({100*np.isfinite(table['logg']).sum()/len(table):.1f}%)")
print(f"   Com massa:            {np.isfinite(table['mass']).sum():,} ({100*np.isfinite(table['mass']).sum()/len(table):.1f}%)")

if 'parallax' in table.colnames:
    print(f"   Com parallax:         {np.isfinite(table['parallax']).sum():,} ({100*np.isfinite(table['parallax']).sum()/len(table):.1f}%)")

print(f"\n🎨 FOTOMETRIA S-PLUS SINTÉTICA:")
for filt in splus_filters[:4]:  # Mostrar apenas alguns
    col = f'splus_{filt}'
    if col in table.colnames:
        n_valid = np.isfinite(table[col]).sum()
        print(f"   {filt:10s}: {n_valid:,} ({100*n_valid/len(table):.1f}%)")

print(f"\n📋 TIPOS ESPECTRAIS (top 10):")
if 'wd_type' in table.colnames:
    types, counts = np.unique(table['wd_type'], return_counts=True)
    top_idx = np.argsort(counts)[::-1][:10]
    for idx in top_idx:
        print(f"   {types[idx]:15s}: {counts[idx]:,} ({100*counts[idx]/len(table):.1f}%)")

print(f"\n✅ CATÁLOGO PRIMÁRIO CRIADO COM SUCESSO!")
print(f"📁 Arquivo: {output_file}\n")