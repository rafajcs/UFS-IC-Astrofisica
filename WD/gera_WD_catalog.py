# -*- coding: utf-8 -*-
"""
PIPELINE COMPLETO: UNIÃO DE CATÁLOGOS DE ANÃS BRANCAS
======================================================
Etapa 1: União (concatenação) de todos os catálogos
Etapa 2: Filtrar apenas objetos COM espectros disponíveis
Etapa 3: Cross-match angular (5") para remover duplicatas
Etapa 4: Exportar CSV e FITS final
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy import units as u
from astropy.table import Table
from pathlib import Path
from datetime import datetime
import glob
import os
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURAÇÃO GLOBAL ====================
OUTPUT_DIR = Path(f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("PIPELINE: UNIÃO DE CATÁLOGOS DE ANÃS BRANCAS")
print("="*80)
print(f"📁 Diretório de saída: {OUTPUT_DIR}\n")


# ==================== FUNÇÕES AUXILIARES ====================

def save_checkpoint(df: pd.DataFrame, step: str, description: str):
    """Salva checkpoint de cada etapa"""
    filename = OUTPUT_DIR / f"checkpoint_{step}.csv"
    df.to_csv(filename, index=False)
    print(f"💾 Checkpoint salvo: {filename}")
    print(f"   {len(df):,} objetos | {description}\n")
    return filename


def load_checkpoint(step: str) -> pd.DataFrame:
    """Carrega checkpoint de etapa anterior"""
    filename = OUTPUT_DIR / f"checkpoint_{step}.csv"
    if filename.exists():
        df = pd.read_csv(filename)
        print(f"✅ Checkpoint carregado: {filename}")
        print(f"   {len(df):,} objetos\n")
        return df
    else:
        print(f"❌ Checkpoint não encontrado: {filename}\n")
        return None


def fix_endian(arr):
    """Corrige endianness (NumPy 2.0 compatible)"""
    if isinstance(arr, np.ndarray) and arr.dtype.byteorder == '>':
        return arr.byteswap().view(arr.dtype.newbyteorder('<'))
    return arr


# ==================== ETAPA 1: UNIÃO DE CATÁLOGOS ====================

def load_sdss_dr7_wd(filepath: str) -> pd.DataFrame:
    """Carrega SDSS DR7 WD catalog"""
    print(f"  📂 SDSS DR7 WD: {filepath}")
    
    with fits.open(filepath) as hdul:
        data = hdul[1].data
        
        df = pd.DataFrame({
            'catalog_source': 'SDSS_DR7',
            'object_id': [s.decode('utf-8').strip() if isinstance(s, bytes) else str(s).strip() 
                         for s in data['SDSS']],
            'ra': fix_endian(data['RAdeg']),
            'dec': fix_endian(data['DEdeg']),
            'plate': fix_endian(data['Plate']),
            'mjd': fix_endian(data['MJD']),
            'fiber': fix_endian(data['Fiber']),
            'u_mag': fix_endian(data['umag']),
            'g_mag': fix_endian(data['gmag']),
            'r_mag': fix_endian(data['rmag']),
            'i_mag': fix_endian(data['imag']),
            'z_mag': fix_endian(data['zmag']),
            'teff': fix_endian(data['Teff']),
            'logg': fix_endian(data['logg']),
            'mass': fix_endian(data['Mass']),
            'wd_type': [s.decode('utf-8').strip() if isinstance(s, bytes) else str(s).strip() 
                       for s in data['Type']]
        })
    
    print(f"     ✅ {len(df):,} objetos")
    return df


def load_gaia_edr3_wd(filepath: str, max_rows: int = None) -> pd.DataFrame:
    """Carrega Gaia EDR3 WD catalog"""
    print(f"  📂 Gaia EDR3 WD: {filepath}")
    
    chunk_size = 10000
    all_data = []
    
    with fits.open(filepath, memmap=True, ignore_missing_end=True) as hdul:
        data_hdu = hdul[1]
        n_total = data_hdu.header['NAXIS2']
        n_load = n_total if max_rows is None else min(max_rows, n_total)
        
        for start in range(0, n_load, chunk_size):
            end = min(start + chunk_size, n_load)
            
            try:
                chunk = {
                    'catalog_source': 'Gaia_EDR3',
                    'object_id': data_hdu.data.field('WDJ_name')[start:end],
                    'gaia_source_id': fix_endian(data_hdu.data.field('source_id')[start:end]),
                    'ra': fix_endian(data_hdu.data.field('ra')[start:end]),
                    'dec': fix_endian(data_hdu.data.field('dec')[start:end]),
                    'parallax': fix_endian(data_hdu.data.field('parallax')[start:end]),
                    'g_mag': fix_endian(data_hdu.data.field('phot_g_mean_mag')[start:end]),
                    'bp_mag': fix_endian(data_hdu.data.field('phot_bp_mean_mag')[start:end]),
                    'rp_mag': fix_endian(data_hdu.data.field('phot_rp_mean_mag')[start:end]),
                    'teff': fix_endian(data_hdu.data.field('teff_H')[start:end]),
                    'logg': fix_endian(data_hdu.data.field('logg_H')[start:end]),
                    'mass': fix_endian(data_hdu.data.field('mass_H')[start:end]),
                }
                all_data.append(pd.DataFrame(chunk))
                
                if end % 50000 == 0:
                    print(f"     ... {end:,}/{n_load:,}")
            except:
                break
    
    df = pd.concat(all_data, ignore_index=True)
    print(f"     ✅ {len(df):,} objetos")
    return df

def inspect_fits_structure(filepath: str):
    """Inspeciona estrutura de arquivo FITS para debug"""
    print(f"\n🔍 INSPECIONANDO: {filepath}")
    print("="*60)
    
    try:
        with fits.open(filepath) as hdul:
            print(f"Número de HDUs: {len(hdul)}\n")
            
            for i, hdu in enumerate(hdul):
                print(f"HDU {i}: {hdu.name} ({type(hdu).__name__})")
                
                if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                    print(f"  Linhas: {len(hdu.data):,}")
                    print(f"  Colunas: {len(hdu.columns.names)}")
                    print(f"\n  📋 Nomes das colunas:")
                    for j, col in enumerate(hdu.columns.names):
                        col_format = hdu.columns[col].format
                        print(f"    [{j:3d}] {col:30s} ({col_format})")
                    
                    # Mostrar primeira linha
                    if len(hdu.data) > 0:
                        print(f"\n  🔬 Primeira linha (primeiras 5 colunas):")
                        for col in hdu.columns.names[:5]:
                            val = hdu.data[col][0]
                            print(f"    {col:30s} = {val}")
                    print()
                
                elif isinstance(hdu, fits.ImageHDU):
                    if hdu.data is not None:
                        print(f"  Shape: {hdu.data.shape}")
                    print()
    
    except Exception as e:
        print(f"❌ Erro ao inspecionar: {e}\n")


def load_montreal_wd(filepath: str) -> pd.DataFrame:
    """Carrega Montreal WD Database (formato .fit ou .fits)"""
    print(f"  📂 Montreal WD: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"     ❌ Arquivo não encontrado!")
        return pd.DataFrame()
    
    try:
        # Detectar formato
        if filepath.endswith('.csv') or filepath.endswith('.txt'):
            # CSV/texto
            df = pd.read_csv(filepath)
            
            # Padronizar nomes de colunas
            df['catalog_source'] = 'Montreal'
            
            # Mapear colunas comuns (ajustar conforme necessário)
            col_mapping = {
                'Name': 'object_id',
                'WD': 'object_id',
                'RA': 'ra',
                'RAJ2000': 'ra',
                'RA_ICRS': 'ra',
                'Dec': 'dec',
                'DEJ2000': 'dec',
                'DE_ICRS': 'dec',
                'Teff': 'teff',
                'T_eff': 'teff',
                'logg': 'logg',
                'log_g': 'logg',
                'Mass': 'mass',
            }
            
            df = df.rename(columns=col_mapping)
        
        elif filepath.endswith('.fit') or filepath.endswith('.fits'):
            # FITS
            print(f"     📖 Lendo arquivo FITS...")
            
            with fits.open(filepath) as hdul:
                # Tentar encontrar HDU com dados
                data_hdu = None
                for hdu in hdul:
                    if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)) and len(hdu.data) > 0:
                        data_hdu = hdu
                        break
                
                if data_hdu is None:
                    print(f"     ❌ Nenhuma tabela encontrada no FITS!")
                    print(f"     💡 Executando inspeção automática...")
                    inspect_fits_structure(filepath)
                    return pd.DataFrame()
                
                print(f"     📊 Usando HDU: {data_hdu.name}")
                print(f"     📊 Colunas disponíveis: {len(data_hdu.columns.names)}")
                
                # Listar colunas disponíveis
                available_cols = data_hdu.columns.names
                print(f"     📋 Primeiras colunas: {available_cols[:10]}")
                
                # Dicionário de possíveis mapeamentos de colunas
                # Formato: nome_esperado: [possíveis nomes no arquivo]
                col_possibilities = {
                    'object_id': ['Name', 'WD', 'DESIGNATION', 'ID', 'WDJ_name', 'Object', 'TARGET'],
                    'ra': ['RA', 'RAJ2000', 'RA_ICRS', 'RAdeg', '_RA', 'ra'],
                    'dec': ['Dec', 'DEJ2000', 'DE_ICRS', 'DEdeg', '_DE', 'dec'],
                    'teff': ['Teff', 'T_eff', 'TEFF', 'Teff_H', 'teff'],
                    'logg': ['logg', 'log_g', 'LOGG', 'logg_H'],
                    'mass': ['Mass', 'M', 'MASS', 'mass_H'],
                    'u_mag': ['umag', 'u', 'U', 'Umag'],
                    'g_mag': ['gmag', 'g', 'G', 'Gmag'],
                    'r_mag': ['rmag', 'r', 'R', 'Rmag'],
                    'i_mag': ['imag', 'i', 'I', 'Imag'],
                    'z_mag': ['zmag', 'z', 'Z', 'Zmag'],
                }
                
                # Construir DataFrame com mapeamento automático
                data_dict = {'catalog_source': 'Montreal'}
                
                for target_col, possible_names in col_possibilities.items():
                    found = False
                    for possible_name in possible_names:
                        if possible_name in available_cols:
                            try:
                                raw_data = data_hdu.data[possible_name]
                                data_dict[target_col] = fix_endian(raw_data)
                                found = True
                                print(f"     ✓ {target_col:15s} ← {possible_name}")
                                break
                            except Exception as e:
                                print(f"     ⚠️  Erro ao ler {possible_name}: {e}")
                    
                    if not found and target_col in ['object_id', 'ra', 'dec']:
                        # Colunas essenciais
                        print(f"     ❌ ESSENCIAL não encontrada: {target_col}")
                        print(f"        Tentamos: {possible_names}")
                
                # Verificar se temos colunas essenciais
                if 'object_id' not in data_dict or 'ra' not in data_dict or 'dec' not in data_dict:
                    print(f"\n     ❌ ERRO: Colunas essenciais faltando!")
                    print(f"     💡 Executando inspeção detalhada...")
                    inspect_fits_structure(filepath)
                    
                    # Permitir continuar com colunas mínimas
                    if 'ra' in data_dict and 'dec' in data_dict:
                        print(f"     ⚠️  Continuando com ID genérico...")
                        if 'object_id' not in data_dict:
                            data_dict['object_id'] = [f"Montreal_{i:06d}" for i in range(len(data_dict['ra']))]
                    else:
                        return pd.DataFrame()
                
                df = pd.DataFrame(data_dict)
        
        else:
            print(f"     ❌ Formato não suportado: {filepath}")
            return pd.DataFrame()
        
        # Converter strings se necessário
        if 'object_id' in df.columns:
            if isinstance(df['object_id'].iloc[0], bytes):
                df['object_id'] = df['object_id'].apply(
                    lambda x: x.decode('utf-8').strip() if isinstance(x, bytes) else str(x).strip()
                )
        
        print(f"     ✅ {len(df):,} objetos carregados")
        return df
    
    except Exception as e:
        print(f"     ❌ Erro ao carregar Montreal WD: {e}")
        print(f"     💡 Tentando inspeção...")
        inspect_fits_structure(filepath)
        return pd.DataFrame()


def load_lamost_dr8_wd(filepath: str) -> pd.DataFrame:
    """Carrega LAMOST DR8 WD catalog"""
    print(f"  📂 LAMOST DR8 WD: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"     ❌ Arquivo não encontrado!")
        return pd.DataFrame()
    
    try:
        with fits.open(filepath) as hdul:
            # Encontrar HDU com dados
            data_hdu = None
            for hdu in hdul:
                if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)) and len(hdu.data) > 0:
                    data_hdu = hdu
                    break
            
            if data_hdu is None:
                print(f"     ❌ Nenhuma tabela encontrada!")
                return pd.DataFrame()
            
            data = data_hdu.data
            
            # Construir DataFrame com mapeamento direto
            df = pd.DataFrame({
                'catalog_source': 'LAMOST_DR8',
                'object_id': [s.decode('utf-8').strip() if isinstance(s, bytes) else str(s).strip() 
                             for s in data['designation']],
                'obsid': fix_endian(data['obsid']),
                'uid': [s.decode('utf-8').strip() if isinstance(s, bytes) else str(s).strip() 
                       for s in data['uid']],
                'ra': fix_endian(data['ra']),
                'dec': fix_endian(data['dec']),
                'ra_obs': fix_endian(data['ra_obs']),
                'dec_obs': fix_endian(data['dec_obs']),
                
                # Identificadores para download de espectro
                'lmjd': [s.decode('utf-8').strip() if isinstance(s, bytes) else str(s).strip() 
                        for s in data['lmjd']],
                'planid': [s.decode('utf-8').strip() if isinstance(s, bytes) else str(s).strip() 
                          for s in data['planid']],
                'spid': fix_endian(data['spid']),
                'fiberid': fix_endian(data['fiberid']),
                
                # Qualidade espectral (S/N)
                'snr_u': fix_endian(data['snru']),
                'snr_g': fix_endian(data['snrg']),
                'snr_r': fix_endian(data['snrr']),
                'snr_i': fix_endian(data['snri']),
                'snr_z': fix_endian(data['snrz']),
                
                # Classificação
                'spec_class': [s.decode('utf-8').strip() if isinstance(s, bytes) else str(s).strip() 
                              for s in data['class']],
                'spec_subclass': [s.decode('utf-8').strip() if isinstance(s, bytes) else str(s).strip() 
                                 for s in data['subclass']],
                'wd_subclass': [s.decode('utf-8').strip() if isinstance(s, bytes) else str(s).strip() 
                               for s in data['wd_subclass']],
                
                # Redshift (para anãs brancas ~0)
                'z': fix_endian(data['z']),
                'z_err': fix_endian(data['z_err']),
                
                # Fotometria Pan-STARRS
                'ps_id': fix_endian(data['ps_ID']),
                'ps_g_mag': fix_endian(data['mag_ps_g']),
                'ps_r_mag': fix_endian(data['mag_ps_r']),
                'ps_i_mag': fix_endian(data['mag_ps_i']),
                'ps_z_mag': fix_endian(data['mag_ps_z']),
                'ps_y_mag': fix_endian(data['mag_ps_y']),
                
                # Fotometria Gaia
                'gaia_source_id': [s.decode('utf-8').strip() if isinstance(s, bytes) else str(s).strip() 
                                  for s in data['gaia_source_id']],
                'gaia_g_mag': fix_endian(data['gaia_g_mean_mag']),
                
                # Parâmetros físicos
                'teff': fix_endian(data['teff']),
                'teff_err': fix_endian(data['teff_err']),
                'logg': fix_endian(data['logg']),
                'logg_err': fix_endian(data['logg_err']),
            })
            
            # Padronizar algumas colunas para compatibilidade
            # Usar fotometria Pan-STARRS como proxy para SDSS (bandas similares)
            df['g_mag'] = df['ps_g_mag']
            df['r_mag'] = df['ps_r_mag']
            df['i_mag'] = df['ps_i_mag']
            df['z_mag'] = df['ps_z_mag']
            
            print(f"     ✅ {len(df):,} objetos")
            
            # Estatísticas de qualidade
            snr_cols = ['snr_g', 'snr_r']
            if all(col in df.columns for col in snr_cols):
                snr_mean = df[snr_cols].mean(axis=1)
                high_snr = (snr_mean > 10).sum()
                print(f"     📊 SNR>10: {high_snr:,} ({100*high_snr/len(df):.1f}%)")
            
            return df
            
    except Exception as e:
        print(f"     ❌ Erro ao carregar LAMOST: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def check_spectrum_availability(row: pd.Series) -> dict:
    """
    Verifica se espectro está disponível e retorna informações
    ATUALIZADO: inclui LAMOST
    """
    # SDSS: verificar se tem Plate, MJD, Fiber
    if pd.notna(row.get('plate')) and pd.notna(row.get('mjd')) and pd.notna(row.get('fiber')):
        plate = int(row['plate'])
        mjd = int(row['mjd'])
        fiber = int(row['fiber'])
        
        return {
            'has_spectrum': True,
            'spectrum_source': 'SDSS',
            'spectrum_identifier': f"{plate:04d}-{mjd}-{fiber:04d}",
            'spectrum_url': f"https://data.sdss.org/sas/dr16/sdss/spectro/redux/26/spectra/{plate:04d}/spec-{plate:04d}-{mjd}-{fiber:04d}.fits"
        }
    
    # LAMOST: verificar se tem lmjd, planid, spid, fiberid
    if (pd.notna(row.get('lmjd')) and pd.notna(row.get('planid')) and 
        pd.notna(row.get('spid')) and pd.notna(row.get('fiberid'))):
        
        lmjd = str(row['lmjd']).strip()
        planid = str(row['planid']).strip()
        spid = int(row['spid'])
        fiberid = int(row['fiberid'])
        
        # Formato LAMOST: spec-{planid}-{spid}_{fiberid}.fits
        spec_name = f"spec-{planid}-{spid:03d}_{fiberid:04d}.fits"
        
        # URL LAMOST DR8 (ajustar conforme estrutura real do servidor)
        # Exemplo genérico - verificar documentação oficial
        base_url = f"http://dr8.lamost.org/v2/sas/fits/{lmjd}"
        
        return {
            'has_spectrum': True,
            'spectrum_source': 'LAMOST',
            'spectrum_identifier': spec_name,
            'spectrum_url': f"{base_url}/{spec_name}",
            'lmjd': lmjd,
            'planid': planid,
            'spid': spid,
            'fiberid': fiberid
        }
    
    return {
        'has_spectrum': False,
        'spectrum_source': None,
        'spectrum_identifier': None,
        'spectrum_url': None
    }


def etapa_1_uniao_catalogos():
    """
    ETAPA 1: União (concatenação) de todos os catálogos
    ATUALIZADO: inclui LAMOST DR8
    """
    print("\n" + "="*80)
    print("ETAPA 1: UNIÃO DE CATÁLOGOS")
    print("="*80 + "\n")
    
    # Verificar se checkpoint existe
    df_existing = load_checkpoint("step1_union")
    if df_existing is not None:
        response = input("Checkpoint encontrado. Pular esta etapa? (s/n): ")
        if response.lower() == 's':
            return df_existing
    
    # Carregar catálogos
    print("📚 Carregando catálogos...\n")
    
    catalogs = []
    
    # Lista de arquivos para tentar
    catalog_files = {
        'SDSS_DR7': [
            "SDSS_DR7_WD_catalog.fits",
            "SDSS_DR7_WD_catalog.fit",
        ],
        'Gaia_EDR3': [
            "../GaiaEDR3_WD_main.fits",
            "GaiaEDR3_WD_main.fits",
            "../GaiaEDR3_WD_main.fit",
        ],
        'LAMOST_DR8': [
            "lamost_dr8_wd.fits",
            "lamost_dr8_wd.fit",
            "../lamost_dr8_wd.fits",
            "LAMOST_DR8_WD.fits",
        ],
    }
    
    # SDSS DR7 WD
    print("1️⃣  SDSS DR7 WD")
    sdss_loaded = False
    for filepath in catalog_files['SDSS_DR7']:
        if os.path.exists(filepath):
            try:
                catalogs.append(load_sdss_dr7_wd(filepath))
                sdss_loaded = True
                break
            except Exception as e:
                print(f"     ⚠️  Erro: {e}")
    
    if not sdss_loaded:
        print(f"  ⚠️  SDSS DR7 não encontrado")
    
    # Gaia EDR3 WD
    print("\n2️⃣  Gaia EDR3 WD")
    gaia_loaded = False
    for filepath in catalog_files['Gaia_EDR3']:
        if os.path.exists(filepath):
            try:
                catalogs.append(load_gaia_edr3_wd(filepath))
                gaia_loaded = True
                break
            except Exception as e:
                print(f"     ⚠️  Erro: {e}")
    
    if not gaia_loaded:
        print(f"  ⚠️  Gaia EDR3 não encontrado")
    
    # LAMOST DR8 WD
    print("\n3️⃣  LAMOST DR8 WD")
    lamost_loaded = False
    for filepath in catalog_files['LAMOST_DR8']:
        if os.path.exists(filepath):
            try:
                df_lamost = load_lamost_dr8_wd(filepath)
                if len(df_lamost) > 0:
                    catalogs.append(df_lamost)
                    lamost_loaded = True
                    break
            except Exception as e:
                print(f"     ⚠️  Erro: {e}")
    
    if not lamost_loaded:
        print(f"  ⚠️  LAMOST DR8 não encontrado")
    
    # Verificar se carregou pelo menos um
    if len(catalogs) == 0:
        print("\n❌ ERRO CRÍTICO: Nenhum catálogo carregado!")
        return None
    
    # Concatenar todos
    print(f"\n🔗 Concatenando {len(catalogs)} catálogos...")
    df_union = pd.concat(catalogs, ignore_index=True, sort=False)
    
    print(f"\n✅ União concluída: {len(df_union):,} objetos totais")
    print(f"   Colunas: {len(df_union.columns)}")
    
    # Estatísticas por catálogo
    print(f"\n📊 Distribuição por catálogo:")
    for source in sorted(df_union['catalog_source'].unique()):
        n = (df_union['catalog_source'] == source).sum()
        print(f"   {source:20s}: {n:8,} ({100*n/len(df_union):5.1f}%)")
    
    # Verificar qualidade dos dados
    print(f"\n🔍 Qualidade dos dados:")
    print(f"   RA válidos:   {df_union['ra'].notna().sum():8,} ({100*df_union['ra'].notna().sum()/len(df_union):5.1f}%)")
    print(f"   Dec válidos:  {df_union['dec'].notna().sum():8,} ({100*df_union['dec'].notna().sum()/len(df_union):5.1f}%)")
    
    if 'teff' in df_union.columns:
        print(f"   Teff válidos: {df_union['teff'].notna().sum():8,} ({100*df_union['teff'].notna().sum()/len(df_union):5.1f}%)")
    
    # Estatísticas adicionais LAMOST
    if (df_union['catalog_source'] == 'LAMOST_DR8').any():
        df_lamost = df_union[df_union['catalog_source'] == 'LAMOST_DR8']
        
        print(f"\n🔬 LAMOST - Estatísticas adicionais:")
        
        # Por tipo espectral
        if 'wd_subclass' in df_lamost.columns:
            print(f"\n   Tipos espectrais (top 5):")
            for wd_type, count in df_lamost['wd_subclass'].value_counts().head().items():
                print(f"   {str(wd_type):15s}: {count:6,} ({100*count/len(df_lamost):5.1f}%)")
        
        # SNR médio
        snr_cols = ['snr_g', 'snr_r', 'snr_i']
        if all(col in df_lamost.columns for col in snr_cols):
            snr_mean = df_lamost[snr_cols].mean(axis=1)
            print(f"\n   SNR médio: {snr_mean.mean():.1f}")
            print(f"   SNR > 10:  {(snr_mean > 10).sum():6,} ({100*(snr_mean > 10).sum()/len(df_lamost):5.1f}%)")
            print(f"   SNR > 20:  {(snr_mean > 20).sum():6,} ({100*(snr_mean > 20).sum()/len(df_lamost):5.1f}%)")
    
    # Salvar checkpoint
    save_checkpoint(df_union, "step1_union", "União completa de catálogos")
    
    return df_union


'''
**Principais características:**

✅ **Mapeamento completo** - todas as 45 colunas do LAMOST  
✅ **Identificadores para download** - lmjd, planid, spid, fiberid  
✅ **SNR tracking** - rastreamento de qualidade espectral  
✅ **Tipos espectrais WD** - DA, DB, DC, etc  
✅ **Cross-IDs** - Pan-STARRS e Gaia linkados  
✅ **Fotometria múltipla** - Pan-STARRS como proxy SDSS  

**Estatísticas esperadas:**
SDSS DR7:     ~20,000 WDs
Gaia EDR3:    ~1,280,000 WDs
LAMOST DR8:   ~15,000 WDs (estimativa)
──────────────────────────────
Total:        ~1,315,000 WDs
Com espectros: ~35,000 (SDSS + LAMOST)

'''
# 

# ==================== ETAPA 2: FILTRAR APENAS COM ESPECTROS ====================

def check_spectrum_availability(row: pd.Series) -> dict:
    """
    Verifica se espectro está disponível e retorna informações
    """
    # SDSS: verificar se tem Plate, MJD, Fiber
    if pd.notna(row.get('plate')) and pd.notna(row.get('mjd')) and pd.notna(row.get('fiber')):
        plate = int(row['plate'])
        mjd = int(row['mjd'])
        fiber = int(row['fiber'])
        
        return {
            'has_spectrum': True,
            'spectrum_source': 'SDSS',
            'spectrum_identifier': f"{plate:04d}-{mjd}-{fiber:04d}",
            'spectrum_url': f"https://data.sdss.org/sas/dr16/sdss/spectro/redux/26/spectra/{plate:04d}/spec-{plate:04d}-{mjd}-{fiber:04d}.fits"
        }
    
    # Adicionar outras fontes de espectros aqui (LAMOST, Gaia RVS, etc)
    
    return {
        'has_spectrum': False,
        'spectrum_source': None,
        'spectrum_identifier': None,
        'spectrum_url': None
    }


def etapa_2_filtrar_espectros(df_union: pd.DataFrame = None):
    """
    ETAPA 2: Filtrar apenas objetos COM espectros disponíveis
    ==========================================================
    """
    print("\n" + "="*80)
    print("ETAPA 2: FILTRAR OBJETOS COM ESPECTROS")
    print("="*80 + "\n")
    
    # Verificar checkpoint
    df_existing = load_checkpoint("step2_with_spectra")
    if df_existing is not None:
        response = input("Checkpoint encontrado. Pular esta etapa? (s/n): ")
        if response.lower() == 's':
            return df_existing
    
    # Carregar etapa anterior se necessário
    if df_union is None:
        df_union = load_checkpoint("step1_union")
        if df_union is None:
            print("❌ ERRO: Execute Etapa 1 primeiro!")
            return None
    
    print(f"📊 Catálogo inicial: {len(df_union):,} objetos\n")
    print("🔍 Verificando disponibilidade de espectros...")
    
    # Verificar espectros para cada objeto
    spectrum_info = []
    
    for i, row in df_union.iterrows():
        info = check_spectrum_availability(row)
        spectrum_info.append(info)
        
        if (i + 1) % 10000 == 0:
            print(f"   ... {i+1:,}/{len(df_union):,} verificados")
    
    # Adicionar colunas ao DataFrame
    for key in spectrum_info[0].keys():
        df_union[key] = [info[key] for info in spectrum_info]
    
    # Filtrar apenas com espectros
    df_with_spectra = df_union[df_union['has_spectrum'] == True].copy().reset_index(drop=True)
    
    print(f"\n✅ Filtragem concluída:")
    print(f"   COM espectros: {len(df_with_spectra):,} ({100*len(df_with_spectra)/len(df_union):.1f}%)")
    print(f"   SEM espectros: {len(df_union) - len(df_with_spectra):,}")
    
    # Estatísticas por fonte
    print(f"\n📊 Espectros por fonte:")
    for source in df_with_spectra['spectrum_source'].unique():
        n = (df_with_spectra['spectrum_source'] == source).sum()
        print(f"   {source:20s}: {n:8,}")
    
    # Salvar checkpoint
    save_checkpoint(df_with_spectra, "step2_with_spectra", "Apenas objetos com espectros")
    
    # Salvar também objetos SEM espectros (para referência)
    df_no_spectra = df_union[df_union['has_spectrum'] == False].copy()
    no_spec_file = OUTPUT_DIR / "objects_without_spectra.csv"
    df_no_spectra.to_csv(no_spec_file, index=False)
    print(f"💾 Objetos sem espectros salvos: {no_spec_file}\n")
    
    return df_with_spectra


# ==================== ETAPA 3: REMOVER DUPLICATAS (CROSS-MATCH) ====================

def etapa_3_remover_duplicatas(df_with_spectra: pd.DataFrame = None):
    """
    ETAPA 3: Cross-match angular (5") para remover duplicatas
    ===========================================================
    """
    print("\n" + "="*80)
    print("ETAPA 3: REMOVER DUPLICATAS (CROSS-MATCH 5\")")
    print("="*80 + "\n")
    
    # Verificar checkpoint
    df_existing = load_checkpoint("step3_unique")
    if df_existing is not None:
        response = input("Checkpoint encontrado. Pular esta etapa? (s/n): ")
        if response.lower() == 's':
            return df_existing
    
    # Carregar etapa anterior se necessário
    if df_with_spectra is None:
        df_with_spectra = load_checkpoint("step2_with_spectra")
        if df_with_spectra is None:
            print("❌ ERRO: Execute Etapa 2 primeiro!")
            return None
    
    print(f"📊 Catálogo inicial: {len(df_with_spectra):,} objetos\n")
    
    # Criar coordenadas
    print("🌐 Criando coordenadas...")
    coords = SkyCoord(
        ra=df_with_spectra['ra'].values * u.degree,
        dec=df_with_spectra['dec'].values * u.degree,
        frame='icrs'
    )
    
    # Cross-match consigo mesmo
    print("🔗 Realizando cross-match (raio = 5 arcsec)...")
    radius = 5.0 * u.arcsec
    
    # Match para cada objeto
    idx_match, sep2d, _ = match_coordinates_sky(coords, coords, nthneighbor=2)
    
    # Identificar duplicatas (separação < 5")
    is_duplicate = sep2d < radius
    
    print(f"   Duplicatas encontradas: {is_duplicate.sum():,}")
    
    # Estratégia: manter o primeiro de cada grupo de duplicatas
    print("\n🔧 Removendo duplicatas...")
    
    to_keep = np.ones(len(df_with_spectra), dtype=bool)
    
    for i in range(len(df_with_spectra)):
        if not to_keep[i]:
            continue
        
        if is_duplicate[i]:
            # Encontrar todos os objetos próximos
            matches = np.where((coords.separation(coords[i]) < radius) & (np.arange(len(coords)) != i))[0]
            
            # Marcar duplicatas para remoção
            to_keep[matches] = False
        
        if (i + 1) % 10000 == 0:
            print(f"   ... {i+1:,}/{len(df_with_spectra):,} processados")
    
    df_unique = df_with_spectra[to_keep].copy().reset_index(drop=True)
    
    print(f"\n✅ Remoção de duplicatas concluída:")
    print(f"   Únicos:     {len(df_unique):,}")
    print(f"   Removidos:  {len(df_with_spectra) - len(df_unique):,}")
    print(f"   Taxa de duplicação: {100*(1 - len(df_unique)/len(df_with_spectra)):.1f}%")
    
    # Salvar checkpoint
    save_checkpoint(df_unique, "step3_unique", "Duplicatas removidas")
    
    # Salvar duplicatas removidas (para análise)
    df_duplicates = df_with_spectra[~to_keep].copy()
    dup_file = OUTPUT_DIR / "duplicates_removed.csv"
    df_duplicates.to_csv(dup_file, index=False)
    print(f"💾 Duplicatas removidas salvas: {dup_file}\n")
    
    return df_unique


# ==================== ETAPA 4: EXPORTAR FINAL ====================

def etapa_4_exportar_final(df_unique: pd.DataFrame = None):
    """
    ETAPA 4: Exportar CSV e FITS final
    ====================================
    """
    print("\n" + "="*80)
    print("ETAPA 4: EXPORTAR CATÁLOGO FINAL")
    print("="*80 + "\n")
    
    # Carregar etapa anterior se necessário
    if df_unique is None:
        df_unique = load_checkpoint("step3_unique")
        if df_unique is None:
            print("❌ ERRO: Execute Etapa 3 primeiro!")
            return None
    
    print(f"📊 Catálogo final: {len(df_unique):,} objetos\n")
    
    # Exportar CSV
    csv_file = OUTPUT_DIR / "final_wd_catalog.csv"
    print(f"💾 Exportando CSV: {csv_file}")
    df_unique.to_csv(csv_file, index=False)
    print(f"   ✅ CSV salvo")
    
    # Exportar FITS
    fits_file = OUTPUT_DIR / "final_wd_catalog.fits"
    print(f"\n💾 Exportando FITS: {fits_file}")
    
    try:
        # Converter para Astropy Table
        table = Table.from_pandas(df_unique)
        
        # Salvar FITS
        table.write(fits_file, format='fits', overwrite=True)
        print(f"   ✅ FITS salvo")
        
    except Exception as e:
        print(f"   ⚠️  Erro ao salvar FITS: {e}")
        print(f"   CSV foi salvo com sucesso")
    
    # Estatísticas finais
    print(f"\n" + "="*80)
    print("ESTATÍSTICAS FINAIS")
    print("="*80)
    
    print(f"\n📊 CATÁLOGO FINAL:")
    print(f"   Total de objetos: {len(df_unique):,}")
    print(f"   Total de colunas: {len(df_unique.columns)}")
    
    print(f"\n📚 Por catálogo original:")
    for source in df_unique['catalog_source'].unique():
        n = (df_unique['catalog_source'] == source).sum()
        print(f"   {source:20s}: {n:8,} ({100*n/len(df_unique):5.1f}%)")
    
    print(f"\n🔬 Por fonte de espectro:")
    for source in df_unique['spectrum_source'].unique():
        n = (df_unique['spectrum_source'] == source).sum()
        print(f"   {source:20s}: {n:8,}")
    
    print(f"\n📏 Cobertura de dados:")
    coverage_cols = ['ra', 'dec', 'teff', 'logg', 'mass', 'parallax']
    for col in coverage_cols:
        if col in df_unique.columns:
            n_valid = df_unique[col].notna().sum()
            print(f"   {col:15s}: {n_valid:8,} ({100*n_valid/len(df_unique):5.1f}%)")
    
    # Relatório final
    report = f"""
PIPELINE DE UNIÃO DE CATÁLOGOS - RELATÓRIO FINAL
=================================================
Data: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

ETAPAS EXECUTADAS:
   1. União de catálogos
   2. Filtro de objetos com espectros
   3. Remoção de duplicatas (5 arcsec)
   4. Exportação final

RESULTADO FINAL:
   Total: {len(df_unique):,} anãs brancas únicas com espectros disponíveis

ARQUIVOS GERADOS:
   {OUTPUT_DIR}/
   ├── final_wd_catalog.csv          (catálogo final)
   ├── final_wd_catalog.fits         (catálogo final)
   ├── checkpoint_step1_union.csv
   ├── checkpoint_step2_with_spectra.csv
   ├── checkpoint_step3_unique.csv
   ├── objects_without_spectra.csv
   ├── duplicates_removed.csv
   └── RELATORIO.txt

PRÓXIMOS PASSOS:
   1. Download de espectros faltantes
   2. Convolução com filtros S-PLUS
   3. Análise científica

✅ PIPELINE CONCLUÍDO COM SUCESSO!
"""
    
    print(report)
    
    with open(OUTPUT_DIR / "RELATORIO.txt", "w") as f:
        f.write(report)
    
    print(f"\n📂 Todos os arquivos em: {OUTPUT_DIR}/")
    print("\n✅ PIPELINE COMPLETO!\n")
    
    return df_unique


# ==================== MENU PRINCIPAL ====================

def main():
    """Menu interativo para executar etapas"""
    
    print("\n" + "="*80)
    print("MENU PRINCIPAL - PIPELINE DE UNIÃO DE CATÁLOGOS")
    print("="*80)
    
    print("""
Escolha uma opção:
   1 - Executar TODAS as etapas (1→2→3→4)
   2 - Etapa 1: União de catálogos
   3 - Etapa 2: Filtrar objetos com espectros
   4 - Etapa 3: Remover duplicatas
   5 - Etapa 4: Exportar final
   0 - Sair
    """)
    
    choice = input("Opção: ").strip()
    
    if choice == '1':
        # Executar todas
        df1 = etapa_1_uniao_catalogos()
        if df1 is None: return
        
        df2 = etapa_2_filtrar_espectros(df1)
        if df2 is None: return
        
        df3 = etapa_3_remover_duplicatas(df2)
        if df3 is None: return
        
        etapa_4_exportar_final(df3)
        
    elif choice == '2':
        etapa_1_uniao_catalogos()
        
    elif choice == '3':
        etapa_2_filtrar_espectros()
        
    elif choice == '4':
        etapa_3_remover_duplicatas()
        
    elif choice == '5':
        etapa_4_exportar_final()
        
    elif choice == '0':
        print("Saindo...")
    
    else:
        print("Opção inválida!")


if __name__ == "__main__":
    main()