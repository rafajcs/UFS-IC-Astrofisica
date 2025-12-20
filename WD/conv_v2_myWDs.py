# -*- coding: utf-8 -*-
"""
PIPELINE CIENTÍFICO MODULAR: ANÁLISE DE ANÃS BRANCAS
=====================================================
Estrutura modular para processar ~18k+ objetos com espectros

Etapas:
  1. Download de espectros (uma vez, recuperável)
  2. Controle de qualidade espectral
  3. Convolução (fotometria sintética)
  4. Análise e visualização
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table
from pathlib import Path
from datetime import datetime
import glob
import os
import json
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURAÇÃO GLOBAL ====================

class Config:
    """Configuração centralizada do pipeline"""
    
    # Diretórios
    BASE_DIR = Path("pipeline_output")
    SPECTRA_DIR = BASE_DIR / "spectra"
    SDSS_DIR = SPECTRA_DIR / "SDSS"
    LAMOST_DIR = SPECTRA_DIR / "LAMOST"
    QC_DIR = BASE_DIR / "quality_control"
    PHOTOMETRY_DIR = BASE_DIR / "synthetic_photometry"
    FIGURES_DIR = BASE_DIR / "figures"
    
    # Subdiretórios de figuras
    SED_DIR = FIGURES_DIR / "SED"
    COMPARISON_DIR = FIGURES_DIR / "comparisons"
    DIAGNOSTIC_DIR = FIGURES_DIR / "diagnostics"
    
    # Logs
    LOGS_DIR = BASE_DIR / "logs"
    
    # Checkpoints
    CHECKPOINT_DIR = BASE_DIR / "checkpoints"
    
    # Filtros S-PLUS
    FILTERS_DIR = Path("../../splus_filters")  # AJUSTAR!
    
    # Parâmetros de qualidade
    MIN_SNR = 5.0
    MIN_WAVELENGTH = 3500.0  # Å
    MAX_WAVELENGTH = 9000.0  # Å
    MAX_NAN_FRACTION = 0.3
    
    # Processamento em lote
    BATCH_SIZE = 500
    
    @classmethod
    def create_structure(cls):
        """Cria estrutura de diretórios"""
        dirs = [
            cls.BASE_DIR, cls.SPECTRA_DIR, cls.SDSS_DIR, cls.LAMOST_DIR,
            cls.QC_DIR, cls.PHOTOMETRY_DIR, cls.FIGURES_DIR,
            cls.SED_DIR, cls.COMPARISON_DIR, cls.DIAGNOSTIC_DIR,
            cls.LOGS_DIR, cls.CHECKPOINT_DIR
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        
        print("📁 Estrutura de diretórios criada:")
        print(f"   {cls.BASE_DIR}/")
        print(f"   ├── spectra/")
        print(f"   │   ├── SDSS/")
        print(f"   │   └── LAMOST/")
        print(f"   ├── quality_control/")
        print(f"   ├── synthetic_photometry/")
        print(f"   ├── figures/")
        print(f"   │   ├── SED/")
        print(f"   │   ├── comparisons/")
        print(f"   │   └── diagnostics/")
        print(f"   ├── logs/")
        print(f"   └── checkpoints/\n")


# ==================== FUNÇÕES AUXILIARES ====================

def save_log(log_data: dict, filename: str):
    """Salva log em JSON"""
    filepath = Config.LOGS_DIR / f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filepath, 'w') as f:
        json.dump(log_data, f, indent=2)
    return filepath


def load_checkpoint(step: str) -> pd.DataFrame:
    """Carrega checkpoint"""
    filepath = Config.CHECKPOINT_DIR / f"{step}.csv"
    if filepath.exists():
        print(f"✅ Checkpoint encontrado: {step}")
        return pd.read_csv(filepath)
    return None


def save_checkpoint(df: pd.DataFrame, step: str):
    """Salva checkpoint"""
    filepath = Config.CHECKPOINT_DIR / f"{step}.csv"
    df.to_csv(filepath, index=False)
    print(f"💾 Checkpoint salvo: {step} ({len(df):,} objetos)\n")


# ==================== ETAPA 1: DOWNLOAD DE ESPECTROS ====================

def download_sdss_spectrum(plate: int, mjd: int, fiber: int) -> dict:
    """
    Baixa espectro SDSS individual
    
    Returns:
        dict com status: 'success', 'exists', 'not_found', 'error'
    """
    filename = f"spec-{plate:04d}-{mjd}-{fiber:04d}.fits"
    output_path = Config.SDSS_DIR / filename
    
    # Verificar se já existe
    if output_path.exists():
        return {
            'status': 'exists',
            'filepath': str(output_path),
            'plate': plate, 'mjd': mjd, 'fiber': fiber
        }
    
    url = f"https://data.sdss.org/sas/dr16/sdss/spectro/redux/26/spectra/{plate:04d}/{filename}"
    
    try:
        import urllib.request
        urllib.request.urlretrieve(url, output_path)
        return {
            'status': 'success',
            'filepath': str(output_path),
            'plate': plate, 'mjd': mjd, 'fiber': fiber
        }
    except Exception as e:
        return {
            'status': 'not_found' if '404' in str(e) else 'error',
            'error': str(e),
            'plate': plate, 'mjd': mjd, 'fiber': fiber
        }


def download_lamost_spectrum(lmjd: str, planid: str, spid: int, fiberid: int) -> dict:
    """
    Placeholder para download LAMOST
    (LAMOST requer autenticação ou acesso especial)
    """
    filename = f"spec-{planid}-{spid:03d}_{fiberid:04d}.fits"
    output_path = Config.LAMOST_DIR / filename
    
    if output_path.exists():
        return {
            'status': 'exists',
            'filepath': str(output_path),
            'lmjd': lmjd, 'planid': planid, 'spid': spid, 'fiberid': fiberid
        }
    
    # LAMOST geralmente requer acesso via navegador ou script específico
    return {
        'status': 'manual_required',
        'message': 'LAMOST requer download manual ou script específico',
        'lmjd': lmjd, 'planid': planid, 'spid': spid, 'fiberid': fiberid
    }


# Adicione esta função antes da etapa_1_download_espectros:

def find_catalog_with_spectra():
    """
    Procura catálogo processado com informações de espectros
    Busca em várias localizações possíveis
    """
    print("🔍 Procurando catálogo com espectros...\n")
    
    # Localizações possíveis (em ordem de preferência)
    possible_locations = [
        # Checkpoints do pipeline atual
        Config.CHECKPOINT_DIR / "step2_with_spectra.csv",
        
        # Pipeline de união anterior (várias pastas possíveis)
        Path("pipeline_*/checkpoints/step2_with_spectra.csv"),
        Path("unified_catalog_*/checkpoint_step2_with_spectra.csv"),
        
        # Catálogos finais do pipeline anterior
        Path("pipeline_*/final_wd_catalog.csv"),
        Path("unified_catalog_*/SDSS_Gaia_unified_catalog.csv"),
        
        # Resultados de convolução anterior
        Path("results_convolution_*/data/catalog_with_photometry.csv"),
    ]
    
    # Buscar usando glob para padrões com *
    found_files = []
    for pattern in possible_locations:
        if '*' in str(pattern):
            matches = sorted(glob.glob(str(pattern)))
            found_files.extend(matches)
        elif pattern.exists():
            found_files.append(str(pattern))
    
    if not found_files:
        return None
    
    # Usar o mais recente
    newest = max(found_files, key=lambda p: os.path.getmtime(p))
    
    print(f"✅ Encontrado: {newest}\n")
    
    df = pd.read_csv(newest)
    
    # Verificar se tem colunas essenciais
    required_cols = ['ra', 'dec']
    spectrum_cols = ['plate', 'mjd', 'fiber', 'lmjd', 'planid', 'spid', 'fiberid']
    
    has_required = all(col in df.columns for col in required_cols)
    has_spectrum_info = any(col in df.columns for col in spectrum_cols)
    
    if not has_required:
        print(f"⚠️  Catálogo sem colunas essenciais (RA, Dec)")
        return None
    
    if not has_spectrum_info:
        print(f"⚠️  Catálogo sem informações de espectros")
        print(f"   Execute o pipeline de união (etapas 1-2) primeiro!\n")
        return None
    
    print(f"📊 Catálogo carregado: {len(df):,} objetos")
    
    # Verificar quantos têm espectros SDSS ou LAMOST
    has_sdss = df[['plate', 'mjd', 'fiber']].notna().all(axis=1).sum() if all(c in df.columns for c in ['plate', 'mjd', 'fiber']) else 0
    has_lamost = df[['lmjd', 'planid', 'spid', 'fiberid']].notna().all(axis=1).sum() if all(c in df.columns for c in ['lmjd', 'planid', 'spid', 'fiberid']) else 0
    
    print(f"   SDSS:    {has_sdss:,} objetos")
    print(f"   LAMOST:  {has_lamost:,} objetos\n")
    
    return df


# Agora modifique a etapa_1_download_espectros:

def etapa_1_download_espectros(catalog_df: pd.DataFrame = None, resume: bool = True):
    """
    ETAPA 1: Download de espectros
    ================================
    """
    print("\n" + "="*80)
    print("ETAPA 1: DOWNLOAD DE ESPECTROS")
    print("="*80 + "\n")
    
    Config.create_structure()
    
    # Carregar catálogo
    if catalog_df is None:
        # Tentar carregar checkpoint local primeiro
        catalog_df = load_checkpoint("step2_with_spectra")
        
        # Se não existir, procurar em outras localizações
        if catalog_df is None:
            catalog_df = find_catalog_with_spectra()
        
        # Se ainda não encontrou, dar instruções
        if catalog_df is None:
            print("=" * 80)
            print("❌ NENHUM CATÁLOGO ENCONTRADO")
            print("=" * 80)
            print("\n💡 SOLUÇÕES:\n")
            print("1️⃣  Execute o pipeline de união primeiro:")
            print("   python pipeline_union.py")
            print("   Escolha opção 1 (União) e depois opção 2 (Filtrar espectros)\n")
            print("2️⃣  Ou forneça um catálogo manualmente:")
            print("   df = pd.read_csv('seu_catalogo.csv')")
            print("   etapa_1_download_espectros(catalog_df=df)\n")
            print("3️⃣  Ou coloque o catálogo na pasta checkpoints/:")
            print("   cp seu_catalogo.csv pipeline_output/checkpoints/step2_with_spectra.csv\n")
            return None
    
    # Verificar se tem informações necessárias para download
    if 'spectrum_source' not in catalog_df.columns:
        print("⚠️  Catálogo sem coluna 'spectrum_source'")
        print("   Tentando inferir fonte dos espectros...\n")
        
        # Inferir fonte baseado nas colunas presentes
        catalog_df['spectrum_source'] = None
        
        sdss_cols = ['plate', 'mjd', 'fiber']
        if all(c in catalog_df.columns for c in sdss_cols):
            mask_sdss = catalog_df[sdss_cols].notna().all(axis=1)
            catalog_df.loc[mask_sdss, 'spectrum_source'] = 'SDSS'
        
        lamost_cols = ['lmjd', 'planid', 'spid', 'fiberid']
        if all(c in catalog_df.columns for c in lamost_cols):
            mask_lamost = catalog_df[lamost_cols].notna().all(axis=1)
            catalog_df.loc[mask_lamost, 'spectrum_source'] = 'LAMOST'
        
        n_with_source = catalog_df['spectrum_source'].notna().sum()
        print(f"✅ Inferido fonte para {n_with_source:,} objetos\n")
    
    print(f"📊 Catálogo: {len(catalog_df):,} objetos totais")
    
    # Filtrar apenas objetos com informações de espectros
    has_spectrum_info = catalog_df['spectrum_source'].notna()
    catalog_df = catalog_df[has_spectrum_info].copy()
    
    print(f"📊 Com informações de espectros: {len(catalog_df):,}\n")
    
    if len(catalog_df) == 0:
        print("❌ Nenhum objeto com informações de espectros!")
        return None
    
    # Salvar checkpoint para uso futuro
    save_checkpoint(catalog_df, "step2_with_spectra")
    
    # [... resto do código da etapa_1 continua igual ...]
    
    # Carregar log anterior (para resume)
    log_file = Config.LOGS_DIR / "download_master.json"
    if resume and log_file.exists():
        print("🔄 Retomando download anterior...")
        with open(log_file, 'r') as f:
            download_log = json.load(f)
    else:
        download_log = {
            'sdss': {'success': [], 'exists': [], 'not_found': [], 'error': []},
            'lamost': {'success': [], 'exists': [], 'manual_required': [], 'error': []}
        }
    
    # Processar SDSS
    sdss_objects = catalog_df[catalog_df['spectrum_source'] == 'SDSS']
    print(f"📥 SDSS: {len(sdss_objects):,} objetos")
    
    if len(sdss_objects) > 0:
        for i, row in sdss_objects.iterrows():
            plate, mjd, fiber = int(row['plate']), int(row['mjd']), int(row['fiber'])
            
            # Verificar se já processado
            spec_id = f"{plate:04d}-{mjd}-{fiber:04d}"
            already_done = any(spec_id in str(item) for status_list in download_log['sdss'].values() 
                              for item in status_list)
            
            if already_done:
                continue
            
            result = download_sdss_spectrum(plate, mjd, fiber)
            download_log['sdss'][result['status']].append(result)
            
            if (i + 1) % 100 == 0:
                n_done = sum(len(v) for v in download_log['sdss'].values())
                print(f"  ✓ {n_done:,}/{len(sdss_objects):,} processados")
                
                # Salvar progresso
                with open(log_file, 'w') as f:
                    json.dump(download_log, f, indent=2)
    
    # Processar LAMOST
    lamost_objects = catalog_df[catalog_df['spectrum_source'] == 'LAMOST']
    if len(lamost_objects) > 0:
        print(f"\n📥 LAMOST: {len(lamost_objects):,} objetos")
        print("   ⚠️  LAMOST requer download manual ou script específico\n")
        
        for i, row in lamost_objects.iterrows():
            result = download_lamost_spectrum(
                row['lmjd'], row['planid'], int(row['spid']), int(row['fiberid'])
            )
            download_log['lamost'][result['status']].append(result)
    
    # Salvar log final
    with open(log_file, 'w') as f:
        json.dump(download_log, f, indent=2)
    
    # Estatísticas
    print(f"\n" + "="*80)
    print("ESTATÍSTICAS DE DOWNLOAD")
    print("="*80)
    
    print(f"\n📊 SDSS:")
    for status, items in download_log['sdss'].items():
        print(f"   {status:15s}: {len(items):6,}")
    
    if len(lamost_objects) > 0:
        print(f"\n📊 LAMOST:")
        for status, items in download_log['lamost'].items():
            print(f"   {status:15s}: {len(items):6,}")
    
    print(f"\n💾 Log salvo: {log_file}")
    print("\n✅ ETAPA 1 CONCLUÍDA!\n")
    
    return download_log

# ==================== ETAPA 2: CONTROLE DE QUALIDADE ====================

def check_spectrum_quality(filepath: str) -> dict:
    """
    Verifica qualidade de espectro individual
    
    Returns:
        dict com métricas de qualidade
    """
    try:
        with fits.open(filepath) as hdul:
            # SDSS structure
            if 'coadd' in [h.name.lower() for h in hdul]:
                data = hdul[1].data
            else:
                data = hdul[1].data
            
            wave = 10 ** data['loglam']
            flux = data['flux']
            ivar = data['ivar']
            
            # Calcular métricas
            snr = np.median(flux * np.sqrt(ivar))
            wave_min, wave_max = wave.min(), wave.max()
            nan_fraction = np.isnan(flux).sum() / len(flux)
            flux_positive = (flux > 0).sum() / len(flux)
            
            # Critérios de qualidade
            pass_snr = snr >= Config.MIN_SNR
            pass_wavelength = (wave_min <= Config.MIN_WAVELENGTH and 
                              wave_max >= Config.MAX_WAVELENGTH)
            pass_nan = nan_fraction <= Config.MAX_NAN_FRACTION
            pass_flux = flux_positive > 0.5
            
            passes_qc = all([pass_snr, pass_wavelength, pass_nan, pass_flux])
            
            return {
                'filepath': filepath,
                'passes_qc': passes_qc,
                'snr': float(snr),
                'wave_min': float(wave_min),
                'wave_max': float(wave_max),
                'nan_fraction': float(nan_fraction),
                'flux_positive_fraction': float(flux_positive),
                'pass_snr': pass_snr,
                'pass_wavelength': pass_wavelength,
                'pass_nan': pass_nan,
                'pass_flux': pass_flux,
                'error': None
            }
    
    except Exception as e:
        return {
            'filepath': filepath,
            'passes_qc': False,
            'error': str(e)
        }


def etapa_2_quality_control():
    """
    ETAPA 2: Controle de qualidade espectral
    ==========================================
    """
    print("\n" + "="*80)
    print("ETAPA 2: CONTROLE DE QUALIDADE")
    print("="*80 + "\n")
    
    print(f"📋 Critérios de qualidade:")
    print(f"   SNR mínimo:          {Config.MIN_SNR}")
    print(f"   Range wavelength:    {Config.MIN_WAVELENGTH}-{Config.MAX_WAVELENGTH} Å")
    print(f"   NaN máximo:          {Config.MAX_NAN_FRACTION*100:.0f}%")
    print(f"   Fluxo positivo mín:  50%\n")
    
    # Encontrar todos os espectros
    sdss_files = list(Config.SDSS_DIR.glob("spec-*.fits"))
    lamost_files = list(Config.LAMOST_DIR.glob("spec-*.fits"))
    all_files = sdss_files + lamost_files
    
    print(f"📂 Espectros encontrados: {len(all_files):,}")
    print(f"   SDSS:    {len(sdss_files):,}")
    print(f"   LAMOST:  {len(lamost_files):,}\n")
    
    if len(all_files) == 0:
        print("❌ Nenhum espectro encontrado! Execute Etapa 1 primeiro.\n")
        return None
    
    # Verificar qualidade
    print("🔍 Verificando qualidade...")
    qc_results = []
    
    for i, filepath in enumerate(all_files):
        result = check_spectrum_quality(str(filepath))
        qc_results.append(result)
        
        if (i + 1) % 500 == 0:
            print(f"  ✓ {i+1:,}/{len(all_files):,} verificados")
    
    # Criar DataFrames
    df_qc = pd.DataFrame(qc_results)
    
    df_good = df_qc[df_qc['passes_qc'] == True].copy()
    df_bad = df_qc[df_qc['passes_qc'] == False].copy()
    
    # Salvar resultados
    good_file = Config.QC_DIR / "spectra_good.csv"
    bad_file = Config.QC_DIR / "spectra_bad.csv"
    
    df_good.to_csv(good_file, index=False)
    df_bad.to_csv(bad_file, index=False)
    
    # Estatísticas
    print(f"\n" + "="*80)
    print("RESULTADOS DO CONTROLE DE QUALIDADE")
    print("="*80)
    
    print(f"\n✅ APROVADOS: {len(df_good):,} ({100*len(df_good)/len(df_qc):.1f}%)")
    print(f"❌ REPROVADOS: {len(df_bad):,} ({100*len(df_bad)/len(df_qc):.1f}%)")
    
    if len(df_good) > 0:
        print(f"\n📊 Estatísticas dos aprovados:")
        print(f"   SNR médio:   {df_good['snr'].mean():.1f}")
        print(f"   SNR mediano: {df_good['snr'].median():.1f}")
        print(f"   SNR mínimo:  {df_good['snr'].min():.1f}")
    
    if len(df_bad) > 0:
        print(f"\n📊 Razões de reprovação:")
        reasons = {
            'SNR baixo': (~df_bad['pass_snr']).sum(),
            'Wavelength insuficiente': (~df_bad['pass_wavelength']).sum(),
            'Muitos NaNs': (~df_bad['pass_nan']).sum(),
            'Fluxo negativo': (~df_bad['pass_flux']).sum(),
            'Erro de leitura': df_bad['error'].notna().sum()
        }
        for reason, count in reasons.items():
            if count > 0:
                print(f"   {reason:25s}: {count:6,} ({100*count/len(df_bad):5.1f}%)")
    
    print(f"\n💾 Arquivos salvos:")
    print(f"   Aprovados:  {good_file}")
    print(f"   Reprovados: {bad_file}")
    
    print("\n✅ ETAPA 2 CONCLUÍDA!\n")
    
    return df_good, df_bad


# ==================== MENU PRINCIPAL ====================

def main():
    """Menu interativo"""
    
    print("\n" + "="*80)
    print("PIPELINE CIENTÍFICO - ANÃS BRANCAS")
    print("="*80)
    
    print("""
Etapas disponíveis:
   1 - Download de espectros (SDSS + LAMOST)
   2 - Controle de qualidade espectral
   3 - Convolução (fotometria sintética) [EM DESENVOLVIMENTO]
   4 - Análise e visualização [EM DESENVOLVIMENTO]
   
   0 - Sair
    """)
    
    choice = input("Opção: ").strip()
    
    if choice == '1':
        etapa_1_download_espectros()
    
    elif choice == '2':
        etapa_2_quality_control()
    
    elif choice == '3':
        print("\n⚠️  Etapa 3 em desenvolvimento...")
    
    elif choice == '4':
        print("\n⚠️  Etapa 4 em desenvolvimento...")
    
    elif choice == '0':
        print("Saindo...")
    
    else:
        print("Opção inválida!")


if __name__ == "__main__":
    main()
'''

---

## 📂 **ORGANIZAÇÃO DE PASTAS**
```
pipeline_output/
├── spectra/                    ← Repositório local de espectros
│   ├── SDSS/
│   │   ├── spec-0685-52203-0225.fits
│   │   ├── spec-0686-52203-0340.fits
│   │   └── ... (~18,000 arquivos)
│   └── LAMOST/
│       ├── spec-planXXX-YYY_ZZZZ.fits
│       └── ...
│
├── quality_control/            ← Resultados QC
│   ├── spectra_good.csv       (aprovados para ciência)
│   └── spectra_bad.csv        (reprovados + razão)
│
├── synthetic_photometry/       ← Fotometria sintética
│   ├── splus_mags.csv         (12 filtros S-PLUS)
│   ├── sdss_mags_synthetic.csv
│   └── batch_001.csv ... batch_N.csv
│
├── figures/                    ← Visualizações
│   ├── SED/
│   │   ├── WD_0001.pdf
│   │   └── ...
│   ├── comparisons/
│   │   ├── sdss_vs_synthetic.pdf
│   │   └── color_color.pdf
│   └── diagnostics/
│       ├── snr_distribution.pdf
│       └── teff_histogram.pdf
│
├── logs/                       ← Logs JSON
│   ├── download_master.json   (status de cada download)
│   └── download_YYYYMMDD_HHMMSS.json
│
└── checkpoints/                ← Checkpoints CSV
    ├── step1_union.csv
    ├── step2_with_spectra.csv
    └── step3_unique.csv
    
'''