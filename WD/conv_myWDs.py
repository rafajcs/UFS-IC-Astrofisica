# -*- coding: utf-8 -*-
"""
Análise de Espectros do meu catálogo - Anãs Brancas
Visualiza espectros reais e aplica convolução com filtros S-PLUS
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
import socket
socket.setdefaulttimeout(20)  # segundos


# ==================== CONSTANTES ====================
num_teste = 18000

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
    """
    Carrega catálogo de anãs brancas SDSS DR7
    
    Args:
        catalog_path: Caminho para o arquivo FITS
        
    Returns:
        DataFrame com dados do catálogo
    """
    print(f"📂 Carregando catálogo: {catalog_path}")
    
    with fits.open(catalog_path) as hdul:
        data = hdul[1].data
        
        # Função auxiliar para converter endianness (NumPy 2.0 compatible)
        def fix_endian(arr):
            if isinstance(arr, np.ndarray) and arr.dtype.byteorder == '>':
                # NumPy 2.0: usar view ao invés de newbyteorder()
                return arr.byteswap().view(arr.dtype.newbyteorder('<'))
            return arr
        
        # Converter para DataFrame com correção de endianness
        df = pd.DataFrame({
            'SDSS': data['object_id'],
            'Plate': fix_endian(data['Plate']),
            'MJD': fix_endian(data['MJD']),
            'Fiber': fix_endian(data['Fiber']),
            'RA': fix_endian(data['ra']),
            'Dec': fix_endian(data['dec']),
            #'SNg': fix_endian(data['SNg']),
            'umag': fix_endian(data['u_mag']),
            #'e_umag': fix_endian(data['e_umag']),
            'gmag': fix_endian(data['g_mag']),
            #'e_gmag': fix_endian(data['e_gmag']),
            'rmag': fix_endian(data['r_mag']),
            #'e_rmag': fix_endian(data['e_rmag']),
            'imag': fix_endian(data['i_mag']),
           # 'e_imag': fix_endian(data['e_imag']),
            'zmag': fix_endian(data['z_mag']),
           # 'e_zmag': fix_endian(data['e_zmag']),
            'Teff': fix_endian(data['teff']),
           # 'e_Teff': fix_endian(data['e_Teff']),
            'logg': fix_endian(data['logg']),
            #'e_logg': fix_endian(data['e_logg']),
            'Mass': fix_endian(data['mass']),
            #'e_Mass': fix_endian(data['e_Mass']),
            'Type': data['wd_type']
        })
        
        # Converter strings (tratar bytes ou strings)
        for col in ['SDSS', 'Type']:
            if col in df.columns:
                # Verificar se é bytes
                first_val = df[col].iloc[0]
                if isinstance(first_val, bytes):
                    df[col] = df[col].apply(lambda x: x.decode('utf-8').strip() if isinstance(x, bytes) else str(x).strip())
                elif isinstance(first_val, str):
                    df[col] = df[col].str.strip()
                else:
                    # Array de bytes numpy
                    df[col] = [s.decode('utf-8').strip() if isinstance(s, bytes) else str(s).strip() 
                              for s in df[col]]
    
    print(f"✅ Catálogo carregado: {len(df):,} anãs brancas")
    
    # Estatísticas
    print(f"\n📊 Estatísticas:")
    print(f"   Com fotometria SDSS completa: {df[['umag','gmag','rmag','imag','zmag']].notna().all(axis=1).sum():,}")
    print(f"   Com Teff: {df['Teff'].notna().sum():,}")
    print(f"   Com massa: {df['Mass'].notna().sum():,}")
    
    return df

# ==================== DEFINE LOG DOWNL. ====================


log_file = open("download_sdss_spectra.log", "a", encoding="utf-8")

def log(msg):
    #print(msg)
    log_file.write(f"{datetime.now()} | {msg}\n")
    log_file.flush()
    
# ==================== DOWNLOAD DE ESPECTROS ====================

def download_sdss_spectrum(plate: int, mjd: int, fiber: int, 
                          output_dir: str = "sdss_spectra") -> Optional[str]:
    """
    Baixa espectro SDSS individual
    
    Args:
        plate: Número da placa
        mjd: Modified Julian Date
        fiber: Número da fibra
        output_dir: Diretório para salvar espectros
        
    Returns:
        Caminho do arquivo baixado ou None se falhar
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Nome do arquivo
    filename = f"spec-{plate:04d}-{mjd}-{fiber:04d}.fits"
    output_path = os.path.join(output_dir, filename)
    
    # Se já existe, retornar
    if os.path.exists(output_path):
        return output_path
    
    # URL do SDSS
    url = f"https://data.sdss.org/sas/dr16/sdss/spectro/redux/26/spectra/{plate:04d}/{filename}"
    
    try:
        import urllib.request
        log(f"  📥 Baixando: {filename}")
        urllib.request.urlretrieve(url, output_path)
        return output_path
    except Exception as e:
        log(f"  ❌ Erro ao baixar {filename}: {e}")
        return None


def download_spectra_batch(df: pd.DataFrame, 
                          n_spectra: int = num_teste,
                          output_dir: str = "sdss_spectra") -> list:
    """
    Baixa lote de espectros SDSS
    
    Args:
        df: DataFrame com catálogo
        n_spectra: Número de espectros a baixar
        output_dir: Diretório de saída
        
    Returns:
        Lista de caminhos dos arquivos baixados
    """
    print(f"🚀 Baixando {n_spectra} espectros SDSS...")
    
    downloaded = []
    
    for i in range(max(n_spectra, len(df))):
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
    """
    Carrega espectro SDSS de arquivo FITS
    
    Args:
        filepath: Caminho para o arquivo spec-*.fits
        
    Returns:
        Objeto Spectrum
    """
    with fits.open(filepath) as hdul:
        # Dados espectrais estão no HDU 1 (coadd)
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
            'z': header.get('Z'),  # redshift
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
    """
    Calcula fotometria sintética para todos os filtros
    
    Args:
        spectrum: Espectro
        filters: Dicionário de filtros
        
    Returns:
        Dicionário com resultados por filtro
    """
    results = {}
    for filt_name, filt in filters.items():
        results[filt_name] = synthetic_photometry(spectrum, filt)
    return results


def process_all_spectra(spectra_dir: str, 
                       filters: Dict[str, Filter],
                       output_file: str = "convolution_results.csv") -> pd.DataFrame:
    """
    Processa todos os espectros baixados
    
    Args:
        spectra_dir: Diretório com espectros
        filters: Dicionário de filtros
        output_file: Arquivo de saída CSV
        
    Returns:
        DataFrame com resultados
    """
    spec_files = glob.glob(os.path.join(spectra_dir, "spec-*.fits"))
    
    print(f"🔬 Processando {len(spec_files)} espectros...")
    
    results_list = []
    
    #
    #for i, filepath in enumerate(spec_files[start:end], start=9600):
    for i, filepath in enumerate(spec_files):
        try:
            # Carregar espectro
            spectrum = load_sdss_spectrum_file(filepath)
            
            # Calcular fotometria
            phot_results = process_spectrum_with_filters(spectrum, filters)
            
            # Organizar resultados
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
    
    # Espectro
    ax.plot(spectrum.wave, spectrum.flux, 
            color='black', lw=0.6, alpha=0.7, label='Espectro SDSS')
    
    if spectrum.sigma is not None:
        ax.fill_between(spectrum.wave, 
                        spectrum.flux - spectrum.sigma,
                        spectrum.flux + spectrum.sigma,
                        color='gray', alpha=0.2, label='Erro (1σ)')
    
    # Filtros e pontos
    for filt_name, filt in filters.items():
        color = FILTER_COLORS.get(filt_name, 'gray')
        result = results.get(filt_name)
        
        if result is None or not np.isfinite(result.mag_ab):
            continue
        
        # Filtro escalado
        scale = 0.3 * np.nanmax(spectrum.flux)
        ax.fill_between(filt.wave, filt.throughput * scale,
                       alpha=0.15, color=color)
        ax.plot(filt.wave, filt.throughput * scale,
               color=color, lw=0.5, alpha=0.5)
        
        # Ponto fotométrico
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

    # Limites dinâmicos
    f_positive = spectrum.flux[spectrum.flux > 0]
    if len(f_positive) > 0:
        fmin, fmax = np.percentile(f_positive, [5, 99.5])
        plt.ylim(fmin * 0.1, fmax * 2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.close()
    
    #plt.show()


# ==================== PIPELINE COMPLETO ====================
from pathlib import Path
from datetime import datetime
import traceback

# Configurar caminhos
current_dir = Path.cwd()
ic_astro_dir = current_dir.parent.parent
filters_dir = ic_astro_dir / "splus_filters"

print(f"📂 Diretório atual: {current_dir}")
print(f"📂 Filtros em: {filters_dir}")


if __name__ == "__main__":
    
    # 1. CARREGAR CATÁLOGO
    print("=" * 70)
    print("PASSO 1: CATÁLOGO")
    print("=" * 70)
    catalog_path = "final_wd_catalog.fits"
    df_catalog = load_sdss_wd_catalog(catalog_path)
    
    # 2. BAIXAR ESPECTROS
    
    print("\n" + "=" * 70)
    print("PASSO 2: DOWNLOAD DE ESPECTROS")
    print("=" * 70)
    
    start = 9600
    end = 11000
    df_slice = df_catalog.iloc[start:end]
    #spectra_files = download_spectra_batch(df_catalog, n_spectra=len(df_slice))
    spectra_files = download_spectra_batch(df_catalog, n_spectra=num_teste)
    log_file.close()

    
    # 3. CARREGAR FILTROS
    print("\n" + "=" * 70)
    print("PASSO 3: FILTROS S-PLUS")
    print("=" * 70)
    filters = load_all_filters(str(filters_dir))
    print(f"✅ {len(filters)} filtros carregados")
    
    # 4. PROCESSAR ESPECTROS
    print("\n" + "=" * 70)
    print("PASSO 4: CONVOLUÇÃO ESPECTRAL")
    print("=" * 70)
    df_results = process_all_spectra("sdss_spectra", filters)
    
    # 5. VISUALIZAR EXEMPLOS
    print("\n" + "=" * 70)
    print("PASSO 5: VISUALIZAÇÃO")
    print("=" * 70)

    # Criar pasta de resultados com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"convolucao_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    
    #elimina arquivos corrompidos:
    '''
    def is_valid_sdss_spectrum(filepath):
        try:
            with fits.open(filepath, memmap=False) as hdul:
                _ = hdul[1].data['flux']
            return True
        except Exception:
            return False
            
    spectra_files = [f for f in spectra_files if is_valid_sdss_spectrum(f)]
    '''
    
    
    #start, end
    for i, filepath in enumerate(spectra_files):
        
        spec_id = Path(filepath).stem  # spec-PLATE-MJD-FIBER
        
        try:
            spectrum = load_sdss_spectrum_file(filepath)

            spec_name = f"spec-{spectrum.metadata['plate']:04d}-{spectrum.metadata['mjd']}-{spectrum.metadata['fiberid']:04d}"

            with open("log_convolucao.txt", "a") as f:
                f.write(f"INICIANDO OBJETO: {spec_name}\n")

            print(f"🔄 Plotando [{i}]: {spec_name}\n")
            
            results = process_spectrum_with_filters(spectrum, filters)
            
            plot_spectrum_convolution(
                spectrum, filters, results,
                save_path = output_dir / f"SDSS_{spec_name}_convolution.pdf"
            )

        except Exception as e:
            with open("log_convolucao.txt", "a") as f:
                f.write(f"❌ ERRO NO ARQUIVO {spec_id}\n")
                f.write(f"Índice no loop: {i}\n")
                f.write(f"Erro: {str(e)}\n")
                f.write(traceback.format_exc())
                f.write("\n-------------------------\n")

            print(f"❌ Erro ao processar [{i}] {spec_id}, pulando...")
            continue
    
    
    
    print("\n✅ PIPELINE COMPLETO!")
    print(f"   Espectros processados: {len(df_results)}")
    print(f"   Figuras salvas: {(len(spectra_files))} PDFs")
    print(f"   Resultados: convolution_results.csv")