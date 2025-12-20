# -*- coding: utf-8 -*-
"""
MERGE CORRETO: SDSS (espectros) + S-PLUS sintético + Gaia
Garante que só mantemos objetos com espectros disponíveis
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from pathlib import Path
from datetime import datetime
import glob
import os

print("="*70)
print("MERGE TRIPLO: SDSS (espectros) + S-PLUS sintético + Gaia")
print("="*70)

# ==================== FUNÇÕES DE CARREGAMENTO ====================

def load_sdss_wd_catalog(catalog_path: str) -> pd.DataFrame:
    """Carrega catálogo SDSS completo"""
    print(f"📂 Carregando SDSS: {catalog_path}")
    
    with fits.open(catalog_path) as hdul:
        data = hdul[1].data
        
        def fix_endian(arr):
            if isinstance(arr, np.ndarray) and arr.dtype.byteorder == '>':
                return arr.byteswap().view(arr.dtype.newbyteorder('<'))
            return arr
        
        df = pd.DataFrame({
            'SDSS': data['SDSS'],
            'Plate': fix_endian(data['Plate']),
            'MJD': fix_endian(data['MJD']),
            'Fiber': fix_endian(data['Fiber']),
            'RA': fix_endian(data['RAdeg']),
            'Dec': fix_endian(data['DEdeg']),
            'umag': fix_endian(data['umag']),
            'gmag': fix_endian(data['gmag']),
            'rmag': fix_endian(data['rmag']),
            'imag': fix_endian(data['imag']),
            'zmag': fix_endian(data['zmag']),
            'Teff': fix_endian(data['Teff']),
            'logg': fix_endian(data['logg']),
            'Mass': fix_endian(data['Mass']),
            'Type': data['Type']
        })
        
        for col in ['SDSS', 'Type']:
            if col in df.columns:
                first_val = df[col].iloc[0]
                if isinstance(first_val, bytes):
                    df[col] = df[col].apply(lambda x: x.decode('utf-8').strip() if isinstance(x, bytes) else str(x).strip())
    
    print(f"✅ SDSS: {len(df):,} objetos")
    return df


def load_gaia_wd_catalog(catalog_path: str, max_rows: int = None, chunk_size: int = 10000) -> pd.DataFrame:
    """Carrega catálogo Gaia EDR3 WD"""
    print(f"📂 Carregando Gaia: {catalog_path}")
    
    with fits.open(catalog_path, memmap=True, ignore_missing_end=True) as hdul:
        data_hdu = hdul[1]
        n_header = data_hdu.header['NAXIS2']
        
        columns_to_load = [
            'WDJ_name', 'source_id', 'ra', 'dec', 'parallax',
            'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag',
            'umag', 'gmag', 'rmag', 'imag', 'zmag',
            'teff_H', 'logg_H', 'mass_H', 'Pwd'
        ]
        
        available_cols = [col for col in columns_to_load if col in data_hdu.columns.names]
        
        all_data = []
        n_limit = max_rows if max_rows else n_header
        
        for start in range(0, n_limit, chunk_size):
            end = min(start + chunk_size, n_limit)
            
            try:
                chunk_data = {}
                for col in available_cols:
                    raw_data = data_hdu.data.field(col)[start:end]
                    
                    if isinstance(raw_data, np.ndarray):
                        squeezed = raw_data.squeeze()
                        if squeezed.dtype.byteorder == '>':
                            chunk_data[col] = squeezed.byteswap().view(squeezed.dtype.newbyteorder('<'))
                        else:
                            chunk_data[col] = squeezed
                    else:
                        chunk_data[col] = raw_data
                
                all_data.append(pd.DataFrame(chunk_data))
                
                if (end % 50000) == 0:
                    print(f"   ✓ {end:,}")
            except:
                break
        
        df = pd.concat(all_data, ignore_index=True)
    
    print(f"✅ Gaia: {len(df):,} objetos")
    return df


# ==================== IDENTIFICAR ESPECTROS DISPONÍVEIS ====================

def find_available_spectra(spectra_dir: str) -> pd.DataFrame:
    """
    Lista espectros disponíveis e extrai identificadores
    
    Returns:
        DataFrame com Plate, MJD, Fiber dos espectros disponíveis
    """
    print(f"\n🔍 Procurando espectros em: {spectra_dir}")
    
    spec_files = glob.glob(os.path.join(spectra_dir, "spec-*.fits"))
    
    if len(spec_files) == 0:
        print(f"❌ Nenhum espectro encontrado!")
        return pd.DataFrame(columns=['Plate', 'MJD', 'Fiber', 'spec_file'])
    
    print(f"✅ {len(spec_files):,} espectros encontrados")
    
    # Extrair Plate, MJD, Fiber dos nomes dos arquivos
    spectra_info = []
    
    for filepath in spec_files:
        filename = os.path.basename(filepath)
        # spec-0685-52203-0225.fits
        try:
            parts = filename.replace('spec-', '').replace('.fits', '').split('-')
            plate = int(parts[0])
            mjd = int(parts[1])
            fiber = int(parts[2])
            
            spectra_info.append({
                'Plate': plate,
                'MJD': mjd,
                'Fiber': fiber,
                'spec_file': filepath
            })
        except:
            print(f"⚠️ Erro ao processar: {filename}")
    
    df_spectra = pd.DataFrame(spectra_info)
    print(f"✅ {len(df_spectra):,} identificadores extraídos")
    
    return df_spectra


# ==================== PIPELINE PRINCIPAL ====================

def main():
    print("\n" + "="*70)
    print("PASSO 1: CARREGAR DADOS")
    print("="*70)
    
    # Caminhos - AJUSTE AQUI!
    sdss_fits = "SDSS_DR7_WD_catalog.fits"
    gaia_fits = "GaiaEDR3_WD_main.fits"
    spectra_dir = "results_convolution_2/spectra"  # AJUSTE!
    
    # 1. Catálogo SDSS completo
    df_sdss_full = load_sdss_wd_catalog(sdss_fits)
    
    # 2. Espectros disponíveis
    df_spectra = find_available_spectra(spectra_dir)
    
    if len(df_spectra) == 0:
        print("\n❌ SEM ESPECTROS! Impossível continuar.")
        return
    
    # 3. Carregar fotometria S-PLUS sintética (se existe)
    splus_csv = glob.glob("results_convolution_*/data/photometry_results.csv")
    
    if splus_csv:
        print(f"\n✅ Fotometria S-PLUS encontrada: {splus_csv[-1]}")
        df_splus = pd.read_csv(splus_csv[-1])
        print(f"   {len(df_splus):,} objetos com S-PLUS sintético")
    else:
        print(f"\n⚠️ Fotometria S-PLUS não encontrada (será calculada depois)")
        df_splus = None
    
    # 4. Gaia
    df_gaia = load_gaia_wd_catalog(gaia_fits)
    
    # ==================== MERGE 1: SDSS + Espectros ====================
    print("\n" + "="*70)
    print("PASSO 2: FILTRAR SDSS - APENAS COM ESPECTROS")
    print("="*70)
    
    df_sdss_with_spectra = df_sdss_full.merge(
        df_spectra,
        on=['Plate', 'MJD', 'Fiber'],
        how='inner'
    )
    
    print(f"✅ SDSS com espectros: {len(df_sdss_with_spectra):,} objetos")
    print(f"   (de {len(df_sdss_full):,} total SDSS)")
    
    # ==================== MERGE 2: + S-PLUS sintético ====================
    if df_splus is not None:
        print("\n" + "="*70)
        print("PASSO 3: ADICIONAR FOTOMETRIA S-PLUS SINTÉTICA")
        print("="*70)
        
        df_sdss_with_spectra = df_sdss_with_spectra.merge(
            df_splus,
            left_on=['Plate', 'MJD', 'Fiber'],
            right_on=['plate', 'mjd', 'fiberid'],
            how='left',
            suffixes=('', '_splus')
        )
        
        n_with_splus = df_sdss_with_spectra['uJAVA_mag'].notna().sum() if 'uJAVA_mag' in df_sdss_with_spectra.columns else 0
        print(f"✅ Objetos com S-PLUS sintético: {n_with_splus:,}")
    
    # ==================== MERGE 3: + Gaia ====================
    print("\n" + "="*70)
    print("PASSO 4: CROSS-MATCH COM GAIA")
    print("="*70)
    
    # Coordenadas
    coords_sdss = SkyCoord(
        ra=df_sdss_with_spectra['RA'].values * u.degree,
        dec=df_sdss_with_spectra['Dec'].values * u.degree,
        frame='icrs'
    )
    
    coords_gaia = SkyCoord(
        ra=df_gaia['ra'].values * u.degree,
        dec=df_gaia['dec'].values * u.degree,
        frame='icrs'
    )
    
    # Match
    radius = 1.0 * u.arcsec
    idx_gaia, sep2d, _ = coords_sdss.match_to_catalog_sky(coords_gaia)
    match_mask = sep2d < radius
    
    print(f"   Matches: {match_mask.sum():,} ({100*match_mask.sum()/len(df_sdss_with_spectra):.1f}%)")
    print(f"   Sem match: {(~match_mask).sum():,}")
    print(f"   Separação mediana: {np.median(sep2d[match_mask].arcsec):.3f} arcsec")
    
    # Criar catálogo matched
    df_matched = df_sdss_with_spectra[match_mask].copy().reset_index(drop=True)
    df_gaia_matched = df_gaia.iloc[idx_gaia[match_mask]].copy().reset_index(drop=True)
    
    df_matched['match_separation_arcsec'] = sep2d[match_mask].arcsec
    
    # Renomear Gaia
    df_gaia_matched = df_gaia_matched.add_prefix('gaia_')
    
    # Concatenar
    df_final = pd.concat([df_matched, df_gaia_matched], axis=1)
    
    # ==================== SALVAR ====================
    print("\n" + "="*70)
    print("PASSO 5: SALVAR CATÁLOGO FINAL")
    print("="*70)
    
    output_dir = Path(f"final_catalog_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(exist_ok=True)
    
    # Catálogo final
    output_file = output_dir / "catalog_complete.csv"
    df_final.to_csv(output_file, index=False)
    
    # Objetos sem match Gaia (mas COM espectros)
    df_no_gaia = df_sdss_with_spectra[~match_mask].copy()
    no_gaia_file = output_dir / "catalog_no_gaia.csv"
    df_no_gaia.to_csv(no_gaia_file, index=False)
    
    print(f"✅ Catálogo completo: {output_file}")
    print(f"   {len(df_final):,} objetos")
    print(f"   {len(df_final.columns)} colunas")
    
    print(f"\n✅ Sem Gaia (mas COM espectros): {no_gaia_file}")
    print(f"   {len(df_no_gaia):,} objetos")
    
    # ==================== RELATÓRIO ====================
    print("\n" + "="*70)
    print("RESUMO FINAL")
    print("="*70)
    
    has_splus = 'uJAVA_mag' in df_final.columns
    n_splus = df_final['uJAVA_mag'].notna().sum() if has_splus else 0
    
    report = f"""
CATÁLOGO FINAL - ANÃS BRANCAS
==============================
Data: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

PIPELINE:
   SDSS DR7 WD completo:     {len(df_sdss_full):,}
   └─ Com espectros:         {len(df_sdss_with_spectra):,} ✅
      └─ Com S-PLUS sint.:   {n_splus:,} {'✅' if has_splus else '❌'}
         └─ Match Gaia:      {len(df_final):,} ✅

CATÁLOGO FINAL:
   Total: {len(df_final):,} objetos
   
   Cobertura:
   ✅ Espectros SDSS:        {len(df_final):,} (100%)
   {'✅' if has_splus else '❌'} S-PLUS sintético:      {n_splus:,} ({100*n_splus/len(df_final):.1f}%)
   ✅ Fotometria Gaia:       {len(df_final):,} (100%)
   ✅ Parallax Gaia:         {df_final['gaia_parallax'].notna().sum():,}
   ✅ Parâmetros físicos:    {df_final['Teff'].notna().sum():,} (SDSS)
   
ARQUIVOS:
   {output_dir}/
   ├── catalog_complete.csv      ({len(df_final):,} objetos - COM Gaia)
   ├── catalog_no_gaia.csv       ({len(df_no_gaia):,} objetos - SEM Gaia)
   └── RELATORIO.txt

{'⚠️  ATENÇÃO: Alguns objetos ainda precisam de S-PLUS sintético!' if n_splus < len(df_final) else '✅ TODOS os objetos têm S-PLUS sintético!'}

✅ PRONTO PARA CONVOLUÇÃO E ANÁLISE!
"""
    
    print(report)
    
    with open(output_dir / "RELATORIO.txt", "w") as f:
        f.write(report)
    
    print(f"\n📂 Resultados em: {output_dir}/")
    
    # Verificar espectros
    print(f"\n🔬 VERIFICAÇÃO DE ESPECTROS:")
    sample = df_final.head(3)
    for _, row in sample.iterrows():
        spec_file = row['spec_file']
        exists = "✅" if os.path.exists(spec_file) else "❌"
        print(f"   {exists} {os.path.basename(spec_file)}")
    
    print("\n✅ MERGE TRIPLO CONCLUÍDO!\n")


if __name__ == "__main__":
    main()