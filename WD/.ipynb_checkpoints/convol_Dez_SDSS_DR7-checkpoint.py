# -*- coding: utf-8 -*-
"""
Download e Análise de Espectros SDSS - Anãs Brancas DR7
Baixa espectros reais e aplica convolução com filtros S-PLUS
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import constants as const
import os
import glob
from typing import Dict, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# ==================== CONSTANTES ====================
C_CGS = const.c.to('AA/s').value

FILTER_COLORS = {
    "uJAVA": "#7F7FFF", "J0378": "#7F9FFF", "J0395": "#7FD2FF",
    "J0410": "#7FBF7F", "J0430": "#EBD27F", "gSDSS": "#DEBF7F",
    "J0515": "#E59F7F", "rSDSS": "#FF7F7F", "J0660": "#EB7F7F",
    "iSDSS": "#D27F7F", "J0861": "#AB7F7F", "zSDSS": "#9F7F7F"
}

# ==================== CLASSES ====================
@dataclass
class Spectrum:
    wave: np.ndarray
    flux: np.ndarray
    sigma: Optional[np.ndarray]
    name: str
    metadata: dict = None

@dataclass
class Filter:
    wave: np.ndarray
    throughput: np.ndarray
    name: str
    
    def lambda_eff(self) -> float:
        return np.trapezoid(self.wave * self.throughput, self.wave) / \
               np.trapezoid(self.throughput, self.wave)

@dataclass
class PhotometryResult:
    f_nu: float
    mag_ab: float
    mag_err: float
    lambda_eff: float
    filter_name: str


# ==================== LEITURA DO CATÁLOGO ====================

def load_sdss_wd_catalog(catalog_path: str) -> pd.DataFrame:
    """Carrega catálogo de anãs brancas SDSS DR7"""
    print(f"📂 Carregando catálogo: {catalog_path}")
    
    with fits.open(catalog_path) as hdul:
        data = hdul[1].data
        
        # Função auxiliar para converter endianness (NumPy 2.0 compatible)
        def fix_endian(arr):
            if isinstance(arr, np.ndarray) and arr.dtype.byteorder == '>':
                return arr.byteswap().view(arr.dtype.newbyteorder('<'))
            return arr
        
        # Converter para DataFrame com correção de endianness
        df = pd.DataFrame({
            'SDSS': data['SDSS'],
            'Plate': fix_endian(data['Plate']),
            'MJD': fix_endian(data['MJD']),
            'Fiber': fix_endian(data['Fiber']),
            'RA': fix_endian(data['RAdeg']),
            'Dec': fix_endian(data['DEdeg']),
            'SNg': fix_endian(data['SNg']),
            'umag': fix_endian(data['umag']),
            'e_umag': fix_endian(data['e_umag']),
            'gmag': fix_endian(data['gmag']),
            'e_gmag': fix_endian(data['e_gmag']),
            'rmag': fix_endian(data['rmag']),
            'e_rmag': fix_endian(data['e_rmag']),
            'imag': fix_endian(data['imag']),
            'e_imag': fix_endian(data['e_imag']),
            'zmag': fix_endian(data['zmag']),
            'e_zmag': fix_endian(data['e_zmag']),
            'Teff': fix_endian(data['Teff']),
            'e_Teff': fix_endian(data['e_Teff']),
            'logg': fix_endian(data['logg']),
            'e_logg': fix_endian(data['e_logg']),
            'Mass': fix_endian(data['Mass']),
            'e_Mass': fix_endian(data['e_Mass']),
            'Type': data['Type']
        })
        
        # Converter strings
        for col in ['SDSS', 'Type']:
            if col in df.columns:
                first_val = df[col].iloc[0]
                if isinstance(first_val, bytes):
                    df[col] = df[col].apply(lambda x: x.decode('utf-8').strip() if isinstance(x, bytes) else str(x).strip())
                elif isinstance(first_val, str):
                    df[col] = df[col].str.strip()
                else:
                    df[col] = [s.decode('utf-8').strip() if isinstance(s, bytes) else str(s).strip() 
                              for s in df[col]]
    
    print(f"✅ Catálogo carregado: {len(df):,} anãs brancas")
    print(f"\n📊 Estatísticas:")
    print(f"   Com fotometria SDSS completa: {df[['umag','gmag','rmag','imag','zmag']].notna().all(axis=1).sum():,}")
    print(f"   Com Teff: {df['Teff'].notna().sum():,}")
    print(f"   Com massa: {df['Mass'].notna().sum():,}")
    
    return df


# ==================== DOWNLOAD DE ESPECTROS ====================

def download_sdss_spectrum(plate: int, mjd: int, fiber: int, 
                          output_dir: str = "sdss_spectra") -> Optional[str]:
    """Baixa espectro SDSS individual"""
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"spec-{plate:04d}-{mjd}-{fiber:04d}.fits"
    output_path = os.path.join(output_dir, filename)
    
    if os.path.exists(output_path):
        return output_path
    
    url = f"https://data.sdss.org/sas/dr16/sdss/spectro/redux/26/spectra/{plate:04d}/{filename}"
    
    try:
        import urllib.request
        print(f"  📥 Baixando: {filename}")
        urllib.request.urlretrieve(url, output_path)
        return output_path
    except Exception as e:
        print(f"  ❌ Erro ao baixar {filename}: {e}")
        return None


def download_spectra_batch(df: pd.DataFrame, 
                          n_spectra: int = 10,
                          output_dir: str = "sdss_spectra") -> list:
    """Baixa lote de espectros SDSS"""
    print(f"🚀 Baixando {n_spectra} espectros SDSS...")
    
    downloaded = []
    
    for i in range(min(n_spectra, len(df))):
        row = df.iloc[i]
        plate = int(row['Plate'])
        mjd = int(row['MJD'])
        fiber = int(row['Fiber'])
        
        filepath = download_sdss_spectrum(plate, mjd, fiber, output_dir)
        
        if filepath:
            downloaded.append(filepath)
        
        if (i + 1) % 5 == 0:
            print(f"  ✓ Progresso: {i+1}/{n_spectra}")
    
    print(f"✅ {len(downloaded)}/{n_spectra} espectros baixados")
    return downloaded


# ==================== LEITURA DE ESPECTROS ====================

def load_sdss_spectrum_file(filepath: str) -> Spectrum:
    """Carrega espectro SDSS de arquivo FITS"""
    with fits.open(filepath) as hdul:
        data = hdul[1].data
        
        # Comprimento de onda (log-linear)
        loglam = np.asarray(data['loglam'], dtype=np.float64)
        if loglam.dtype.byteorder == '>':
            loglam = loglam.byteswap().view(loglam.dtype.newbyteorder('<'))
        wave = 10.0 ** loglam
        
        # Fluxo (1e-17 erg/s/cm²/Å)
        flux = np.asarray(data['flux'], dtype=np.float64)
        if flux.dtype.byteorder == '>':
            flux = flux.byteswap().view(flux.dtype.newbyteorder('<'))
        flux = flux * 1e-17
        
        # Erro
        ivar = np.asarray(data['ivar'], dtype=np.float64)
        if ivar.dtype.byteorder == '>':
            ivar = ivar.byteswap().view(ivar.dtype.newbyteorder('<'))
        ivar[ivar <= 0] = np.nan
        sigma = (1 / np.sqrt(ivar)) * 1e-17
        
        # Nome do objeto
        header = hdul[0].header
        obj_name = header.get('NAME', os.path.basename(filepath))
        
        # Metadados
        metadata = {
            'plate': header.get('PLATEID'),
            'mjd': header.get('MJD'),
            'fiberid': header.get('FIBERID'),
            'ra': header.get('PLUG_RA'),
            'dec': header.get('PLUG_DEC'),
            'z': header.get('Z'),
            'class': header.get('CLASS')
        }
    
    return Spectrum(wave=wave, flux=flux, sigma=sigma, 
                   name=obj_name, metadata=metadata)


# ==================== FILTROS ====================

def load_filter_csv(filter_path: str) -> Filter:
    """Carrega filtro de arquivo CSV"""
    df = pd.read_csv(filter_path, comment="#", sep=None, 
                     engine='python', header=None)
    wl = np.array(df.iloc[:, 0], dtype=float)
    thr = np.array(df.iloc[:, 1], dtype=float)
    
    if wl.max() < 2000:
        wl *= 10.0
    
    name = os.path.basename(filter_path).replace('.csv', '')
    return Filter(wave=wl, throughput=thr, name=name)


def load_all_filters(filter_dir: str = "splus_filters") -> Dict[str, Filter]:
    """Carrega todos os filtros S-PLUS"""
    filters = {}
    files = [f for f in glob.glob(f"{filter_dir}/*.csv") 
             if "central_wavelengths" not in f]
    
    for file in files:
        filt = load_filter_csv(file)
        filters[filt.name] = filt
    
    return filters


# ==================== FOTOMETRIA SINTÉTICA ====================

def synthetic_photometry(spectrum: Spectrum, filt: Filter) -> PhotometryResult:
    """Calcula fotometria sintética por convolução"""
    trans = np.interp(spectrum.wave, filt.wave, filt.throughput, 
                     left=0.0, right=0.0)
    
    denom = np.trapezoid(trans * spectrum.wave, spectrum.wave)
    
    if denom == 0 or not np.isfinite(denom):
        return PhotometryResult(
            f_nu=np.nan, mag_ab=np.nan, mag_err=np.nan,
            lambda_eff=filt.lambda_eff(), filter_name=filt.name
        )
    
    num = np.trapezoid(spectrum.flux * trans * spectrum.wave, spectrum.wave)
    f_lambda_eff = num / denom
    
    lambda_eff = filt.lambda_eff()
    f_nu_eff = f_lambda_eff * (lambda_eff ** 2) / C_CGS
    
    if f_nu_eff <= 0 or not np.isfinite(f_nu_eff):
        return PhotometryResult(
            f_nu=np.nan, mag_ab=np.nan, mag_err=np.nan,
            lambda_eff=lambda_eff, filter_name=filt.name
        )
    
    mag_ab = -2.5 * np.log10(f_nu_eff) - 48.6
    
    mag_err = np.nan
    if spectrum.sigma is not None:
        sigma_num_sq = np.trapezoid((spectrum.sigma * trans * spectrum.wave) ** 2, 
                                    spectrum.wave)
        sigma_f_lambda = np.sqrt(sigma_num_sq) / denom
        sigma_f_nu = sigma_f_lambda * (lambda_eff ** 2) / C_CGS
        if f_nu_eff != 0:
            mag_err = (2.5 / np.log(10)) * (sigma_f_nu / np.abs(f_nu_eff))
    
    return PhotometryResult(
        f_nu=f_nu_eff, mag_ab=mag_ab, mag_err=mag_err,
        lambda_eff=lambda_eff, filter_name=filt.name
    )


# ==================== PROCESSAMENTO ====================

def process_spectrum_with_filters(spectrum: Spectrum, 
                                  filters: Dict[str, Filter]) -> Dict[str, PhotometryResult]:
    """Calcula fotometria sintética para todos os filtros"""
    results = {}
    for filt_name, filt in filters.items():
        results[filt_name] = synthetic_photometry(spectrum, filt)
    return results


def process_all_spectra(spectra_dir: str, 
                       filters: Dict[str, Filter],
                       output_file: str = "convolution_results.csv") -> pd.DataFrame:
    """Processa todos os espectros baixados"""
    spec_files = glob.glob(os.path.join(spectra_dir, "spec-*.fits"))
    
    print(f"🔬 Processando {len(spec_files)} espectros...")
    
    results_list = []
    
    for i, filepath in enumerate(spec_files):
        try:
            spectrum = load_sdss_spectrum_file(filepath)
            phot_results = process_spectrum_with_filters(spectrum, filters)
            
            row = {
                'filename': os.path.basename(filepath),
                'object_name': spectrum.name,
                'plate': spectrum.metadata.get('plate'),
                'mjd': spectrum.metadata.get('mjd'),
                'fiberid': spectrum.metadata.get('fiberid'),
                'z': spectrum.metadata.get('z')
            }
            
            for filt_name, result in phot_results.items():
                row[f'{filt_name}_mag'] = result.mag_ab
                row[f'{filt_name}_err'] = result.mag_err
            
            results_list.append(row)
            
            if (i + 1) % 10 == 0:
                print(f"  ✓ {i+1}/{len(spec_files)} processados")
        
        except Exception as e:
            print(f"  ⚠️ Erro em {filepath}: {e}")
    
    df = pd.DataFrame(results_list)
    df.to_csv(output_file, index=False)
    
    print(f"✅ Processamento completo: {len(df)} espectros")
    print(f"💾 Resultados salvos: {output_file}")
    
    return df


# ==================== VISUALIZAÇÃO ====================

def plot_spectrum_convolution(spectrum: Spectrum,
                              filters: Dict[str, Filter],
                              results: Dict[str, PhotometryResult],
                              save_path: str = None):
    """Plota espectro com filtros e pontos sintéticos"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(spectrum.wave, spectrum.flux, 
            color='black', lw=0.6, alpha=0.7, label='Espectro SDSS')
    
    if spectrum.sigma is not None:
        ax.fill_between(spectrum.wave, 
                        spectrum.flux - spectrum.sigma,
                        spectrum.flux + spectrum.sigma,
                        color='gray', alpha=0.2, label='Erro (1σ)')
    
    for filt_name, filt in filters.items():
        color = FILTER_COLORS.get(filt_name, 'gray')
        result = results.get(filt_name)
        
        if result is None or not np.isfinite(result.mag_ab):
            continue
        
        scale = 0.3 * np.nanmax(spectrum.flux)
        ax.fill_between(filt.wave, filt.throughput * scale,
                       alpha=0.15, color=color)
        ax.plot(filt.wave, filt.throughput * scale,
               color=color, lw=0.5, alpha=0.5)
        
        flux_point = 10**(-0.4 * (result.mag_ab + 48.6)) * C_CGS / result.lambda_eff**2
        ax.scatter(result.lambda_eff, flux_point, 
                  c=color, s=80, edgecolor='k', linewidth=1, 
                  zorder=5, label=f'{filt_name} (AB={result.mag_ab:.2f})')
    
    ax.set_xlabel('Comprimento de onda [Å]', fontsize=13)
    ax.set_ylabel(r'Fluxo [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]', fontsize=13)
    ax.set_title(f"{spectrum.name} — Convolução Espectral", fontsize=14, pad=15)
    ax.set_yscale('log')
    ax.set_xlim(3500, 9200)
    ax.legend(ncol=3, fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

print("✅ Todas as funções carregadas!")


# PT.2

from pathlib import Path
from datetime import datetime
import glob

# ==================== CONFIGURAÇÃO ====================
print("="*70)
print("PROCESSAMENTO COMPLETO - SDSS DR7 WD CATALOG")
print("="*70)

# Caminhos
current_dir = Path.cwd()
ic_astro_dir = current_dir.parent.parent
filters_dir = ic_astro_dir / "splus_filters"
catalog_path = "SDSS_DR7_WD_catalog.fits"

# Criar pasta de resultados com timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(f"results_convolution_{timestamp}")
output_dir.mkdir(exist_ok=True)

# Subpastas
spectra_dir = output_dir / "spectra"
figures_dir = output_dir / "figures"
data_dir = output_dir / "data"

for d in [spectra_dir, figures_dir, data_dir]:
    d.mkdir(exist_ok=True)

print(f"\n📁 Estrutura criada:")
print(f"   {output_dir}/")
print(f"   ├── spectra/")
print(f"   ├── figures/")
print(f"   └── data/")

# ==================== CARREGAR CATÁLOGO ====================
print("\n" + "="*70)
print("PASSO 1: CATÁLOGO")
print("="*70)
df_catalog = load_sdss_wd_catalog(catalog_path)

df_catalog.to_csv(data_dir / "catalog_original.csv", index=False)
print(f"💾 Catálogo salvo: {data_dir / 'catalog_original.csv'}")

# ==================== USAR ESPECTROS JÁ BAIXADOS ====================
print("\n" + "="*70)
print("PASSO 2: LOCALIZAR ESPECTROS JÁ BAIXADOS")
print("="*70)

# AJUSTE AQUI: Caminho onde os espectros já estão
#existing_spectra_dir = "sdss_spectra"  # ou o caminho correto
existing_spectra_dir = "results_convolution/spectra"

# Verificar se existe
if not os.path.exists(existing_spectra_dir):
    print(f"❌ Diretório {existing_spectra_dir} não encontrado!")
    print(f"   Procurando em results_convolution_*/spectra...")
    
    # Tentar encontrar pasta de resultados anterior
    results_dirs = sorted(glob.glob("results_convolution_*/spectra"))
    if results_dirs:
        existing_spectra_dir = results_dirs[-1]  # mais recente
        print(f"   ✅ Encontrado: {existing_spectra_dir}")
    else:
        print(f"   ⚠️ Nenhuma pasta anterior encontrada")
        print(f"   Criando lista vazia...")
        spectra_files = []
else:
    print(f"✅ Diretório encontrado: {existing_spectra_dir}")

# Reconstruir lista de arquivos
if os.path.exists(existing_spectra_dir):
    spectra_files = sorted(glob.glob(os.path.join(existing_spectra_dir, "spec-*.fits")))
    print(f"✅ {len(spectra_files)} espectros já baixados encontrados")
    
    # Copiar ou criar link simbólico para nova pasta
    print(f"   Copiando referência para {spectra_dir}...")
    
    # Opção 1: Criar links simbólicos (rápido, não duplica)
    for spec_file in spectra_files:
        src = Path(spec_file).resolve()
        dst = spectra_dir / Path(spec_file).name
        if not dst.exists():
            dst.symlink_to(src)
    
    # Atualizar lista para nova localização
    spectra_files = sorted(glob.glob(str(spectra_dir / "spec-*.fits")))
    print(f"   ✅ {len(spectra_files)} espectros linkados")
    
    # Salvar lista
    with open(data_dir / "downloaded_spectra.txt", "w") as f:
        for spec in spectra_files:
            f.write(f"{spec}\n")
else:
    spectra_files = []
    print(f"⚠️ Nenhum espectro encontrado. Será necessário baixar.")

# ==================== CARREGAR FILTROS ====================
print("\n" + "="*70)
print("PASSO 3: FILTROS S-PLUS")
print("="*70)
filters = load_all_filters(str(filters_dir))
print(f"✅ {len(filters)} filtros carregados")

# ==================== PROCESSAMENTO EM LOTE ====================
if len(spectra_files) > 0:
    print("\n" + "="*70)
    print("PASSO 4: CONVOLUÇÃO ESPECTRAL")
    print("="*70)
    
    df_results = process_all_spectra(
        str(spectra_dir), 
        filters,
        output_file=str(data_dir / "photometry_results.csv")
    )
    
    print(f"✅ {len(df_results)} espectros processados")
    
    # ==================== GERAR FIGURAS ====================
    print("\n" + "="*70)
    print("PASSO 5: GERAR FIGURAS")
    print("="*70)
    
    # n_figures = len(spectra_files)
    n_figures = 1000
    
    for i, filepath in enumerate(spectra_files[:n_figures]):
        try:
            spectrum = load_sdss_spectrum_file(filepath)
            results = process_spectrum_with_filters(spectrum, filters)
            
            fig_name = f"WD_{i:04d}.pdf"
            fig_path = figures_dir / fig_name
            
            plot_spectrum_convolution(
                spectrum, filters, results,
                save_path=str(fig_path)
            )
            
            if (i + 1) % 5 == 0:
                print(f"  ✓ {i+1}/{n_figures} figuras")
        
        except Exception as e:
            print(f"  ⚠️ Erro: {e}")
    
    # ==================== ANÁLISE ====================
    print("\n" + "="*70)
    print("PASSO 6: ANÁLISE")
    print("="*70)
    
    df_merged = df_catalog.merge(
        df_results, 
        left_on=['Plate', 'MJD', 'Fiber'],
        right_on=['plate', 'mjd', 'fiberid'],
        how='inner'
    )
    
    print(f"📊 Dados combinados: {len(df_merged)} objetos")
    
    print("\n📈 Magnitudes sintéticas:")
    for filt in ['uJAVA', 'gSDSS', 'rSDSS', 'iSDSS', 'zSDSS']:
        col = f'{filt}_mag'
        if col in df_merged.columns:
            valid = df_merged[col].notna().sum()
            mean = df_merged[col].mean()
            std = df_merged[col].std()
            print(f"   {filt:10s}: {valid:6,} válidos | μ={mean:6.2f} | σ={std:5.2f}")
    
    df_merged.to_csv(data_dir / "catalog_with_photometry.csv", index=False)
    print(f"\n💾 Completo: {data_dir / 'catalog_with_photometry.csv'}")
    
    # ==================== COMPARAÇÃO ====================
    print("\n" + "="*70)
    print("PASSO 7: COMPARAÇÃO SDSS vs S-PLUS")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    sdss_filters = [('gmag', 'gSDSS'), ('rmag', 'rSDSS'), 
                    ('imag', 'iSDSS'), ('zmag', 'zSDSS')]
    
    for ax, (sdss_col, splus_filt) in zip(axes, sdss_filters):
        splus_col = f'{splus_filt}_mag'
        
        if splus_col in df_merged.columns:
            mask = df_merged[sdss_col].notna() & df_merged[splus_col].notna()
            
            ax.scatter(df_merged.loc[mask, sdss_col], 
                      df_merged.loc[mask, splus_col],
                      alpha=0.3, s=10)
            
            lim_min = min(df_merged.loc[mask, sdss_col].min(), 
                         df_merged.loc[mask, splus_col].min())
            lim_max = max(df_merged.loc[mask, sdss_col].max(), 
                         df_merged.loc[mask, splus_col].max())
            ax.plot([lim_min, lim_max], [lim_min, lim_max], 
                   'r--', lw=2, label='1:1')
            
            ax.set_xlabel(f'SDSS {sdss_col}', fontsize=11)
            ax.set_ylabel(f'S-PLUS {splus_filt} (sintético)', fontsize=11)
            ax.set_title(f'{sdss_col} vs {splus_filt}', fontsize=12)
            ax.legend()
            ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "comparison.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"💾 Comparação: {figures_dir / 'comparison.pdf'}")
    
    # ==================== RELATÓRIO ====================
    print("\n" + "="*70)
    print("RELATÓRIO FINAL")
    print("="*70)
    
    report = f"""
PROCESSAMENTO SDSS DR7 WD - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

📊 ESTATÍSTICAS:
   Catálogo: {len(df_catalog):,} anãs brancas
   Espectros usados: {len(spectra_files):,}
   Processados: {len(df_results):,} espectros
   Figuras: {n_figures}

📁 SAÍDA: {output_dir}/
   ├── spectra/ ({len(spectra_files)} FITS)
   ├── figures/ ({n_figures} PDFs + comparação)
   └── data/ (3 CSVs)

🎨 FILTROS: {', '.join(filters.keys())}

✅ CONCLUÍDO!
"""
    
    print(report)
    
    with open(output_dir / "RELATORIO.txt", "w") as f:
        f.write(report)
    
    print(f"\n💾 Relatório: {output_dir / 'RELATORIO.txt'}")
    print(f"📂 Resultados em: {output_dir}/")

else:
    print("\n❌ Nenhum espectro encontrado para processar!")
    print("   Ajuste a variável 'existing_spectra_dir' para o caminho correto")