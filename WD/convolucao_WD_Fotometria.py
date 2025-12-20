def load_photometry_catalog_alternative(catalog_path: str, max_rows: int = None) -> pd.DataFrame:
    """
    Método alternativo usando astropy.Table com correção manual de endianness
    
    Args:
        catalog_path: Caminho para o arquivo FITS
        max_rows: Número máximo de linhas
        
    Returns:
        DataFrame com dados fotométricos
    """
    from astropy.table import Table
    
    print(f"📂 Carregando catálogo (método alternativo): {catalog_path}")
    
    try:
        # Ler com astropy.Table
        print(f"🔄 Lendo com astropy.Table...")
        t = Table.read(catalog_path, hdu=1)
        
        print(f"✅ Tabela carregada: {len(t):,} objetos")
        
        # Selecionar colunas de interesse
        columns_to_keep = [
            'WDJ_name', 'source_id', 'ra', 'dec', 'parallax',
            'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag',
            'umag', 'gmag', 'rmag', 'imag', 'zmag',
            'e_umag', 'e_gmag', 'e_rmag', 'e_imag', 'e_zmag',
            'teff_H', 'logg_H', 'mass_H', 'Pwd'
        ]
        
        # Filtrar apenas colunas existentes
        available_cols = [col for col in columns_to_keep if col in t.colnames]
        print(f"📋 Colunas selecionadas: {len(available_cols)}")
        
        # Limitar linhas se necessário
        if max_rows and max_rows < len(t):
            t = t[:max_rows]
            print(f"📊 Limitado a {len(t):,} linhas")
        
        # Converter para pandas manualmente com correção de endianness
        print(f"🔄 Convertendo para DataFrame com correção de endianness...")
        
        data_dict = {}
        for col in available_cols:
            col_data = t[col]
            
            # Converter para numpy array
            if hasattr(col_data, 'data'):
                arr = col_data.data
            else:
                arr = np.array(col_data)
            
            # Squeeze para remover dimensões extras
            squeezed = arr.squeeze()
            
            # Corrigir big-endian usando byteswap (NumPy 2.0 compatible)
            if hasattr(squeezed, 'dtype') and squeezed.dtype.byteorder == '>':
                print(f"  🔧 Convertendo {col} de big-endian para little-endian")
                data_dict[col] = squeezed.byteswap().view(squeezed.dtype.newbyteorder('<'))
            else:
                data_dict[col] = squeezed
        
        # Criar DataFrame
        df = pd.DataFrame(data_dict)
        
        print(f"✅ DataFrame criado: {len(df):,} objetos, {len(df.columns)} colunas")
        
        # Estatísticas
        print(f"\n📈 Estatísticas de completude:")
        for col in ['phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 
                   'umag', 'gmag', 'rmag', 'imag', 'zmag']:
            if col in df.columns:
                n_valid = df[col].notna().sum()
                pct_valid = 100 * n_valid / len(df)
                print(f"   {col:20s}: {n_valid:7,} ({pct_valid:5.1f}%)")
        
        return df
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        raise


def load_photometry_catalog(catalog_path: str, max_rows: int = None, 
                           chunk_size: int = 10000, 
                           use_alternative: bool = False) -> pd.DataFrame:
    """
    Carrega catálogo fotométrico do FITS para DataFrame (com proteção contra truncamento e endianness)
    
    Args:
        catalog_path: Caminho para o arquivo FITS
        max_rows: Número máximo de linhas (None = todas disponíveis)
        chunk_size: Tamanho do chunk para leitura segura
        use_alternative: Se True, usa método via astropy.Table
    
    Returns:
        DataFrame com dados fotométricos
    """
    
    # Se solicitado método alternativo, usar direto
    if use_alternative:
        return load_photometry_catalog_alternative(catalog_path, max_rows)
    
    print(f"📂 Carregando catálogo: {catalog_path}")
    
    # Verificar tamanho do arquivo
    file_size_mb = os.path.getsize(catalog_path) / (1024**2)
    print(f"📦 Tamanho do arquivo: {file_size_mb:.1f} MB")
    
    try:
        with fits.open(catalog_path, memmap=True, ignore_missing_end=True) as hdul:
            # Pegar HDU com dados (geralmente índice 1)
            data_hdu = hdul[1]
            
            n_header = data_hdu.header['NAXIS2']
            print(f"📊 Objetos esperados (header): {n_header:,}")
            
            # Colunas de interesse
            columns_to_load = [
                'WDJ_name', 'source_id', 'ra', 'dec', 'parallax',
                'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag',
                'umag', 'gmag', 'rmag', 'imag', 'zmag',
                'e_umag', 'e_gmag', 'e_rmag', 'e_imag', 'e_zmag',
                'teff_H', 'logg_H', 'mass_H', 'Pwd'
            ]
            
            # Verificar quais colunas existem
            available_cols = [col for col in columns_to_load 
                            if col in data_hdu.columns.names]
            print(f"📋 Colunas disponíveis: {len(available_cols)}/{len(columns_to_load)}")
            
            # Ler em chunks para evitar buffer overflow
            print(f"🔍 Carregando dados em chunks...")
            all_data = []
            n_loaded = 0
            
            try:
                # Determinar limite real
                n_limit = max_rows if max_rows else n_header
                
                for start in range(0, n_limit, chunk_size):
                    end = min(start + chunk_size, n_limit)
                    
                    try:
                        # Tentar ler chunk
                        chunk_data = {}
                        for col in available_cols:
                            # Ler dados
                            raw_data = data_hdu.data.field(col)[start:end]
                            
                            # CONVERSÃO COMPLETA de endianness usando byteswap
                            if isinstance(raw_data, np.ndarray):
                                # Fazer squeeze para garantir formato correto
                                squeezed = raw_data.squeeze()
                                
                                # Verificar e corrigir big-endian (NumPy 2.0 compatible)
                                if squeezed.dtype.byteorder == '>':
                                    # Big-endian detectado: usar byteswap + view
                                    chunk_data[col] = squeezed.byteswap().view(squeezed.dtype.newbyteorder('<'))
                                elif squeezed.dtype.byteorder == '|':
                                    # Byte order não aplicável (strings, bytes)
                                    chunk_data[col] = squeezed
                                else:
                                    # Já é little-endian ou nativo
                                    chunk_data[col] = squeezed
                            else:
                                # Não é array numpy (listas, scalars, etc)
                                chunk_data[col] = raw_data
                        
                        # Criar DataFrame do chunk
                        chunk_df = pd.DataFrame(chunk_data)
                        all_data.append(chunk_df)
                        n_loaded = end
                        
                        if (end % 50000) == 0 or end < 1000:
                            print(f"  ✓ Carregado: {end:,} objetos")
                        
                    except Exception as e:
                        print(f"⚠️ Chunk {start}-{end} falhou: {e}")
                        print(f"  Tipo do erro: {type(e).__name__}")
                        print(f"  Parando na linha {n_loaded:,}")
                        break
                
            except Exception as e:
                print(f"⚠️ Erro durante leitura: {e}")
            
            if not all_data:
                raise ValueError("Nenhum dado pôde ser carregado")
            
            # Combinar chunks
            print(f"🔧 Combinando chunks...")
            df = pd.concat(all_data, ignore_index=True)
            
            print(f"✅ Catálogo carregado: {len(df):,} objetos (de {n_header:,} esperados)")
            print(f"📊 Colunas: {len(df.columns)}")
            
            # Estatísticas de completude
            if len(df) < n_header:
                pct = 100 * len(df) / n_header
                print(f"⚠️ Arquivo truncado: {pct:.1f}% dos dados disponíveis")
            
            # Estatísticas de dados válidos
            print(f"\n📈 Estatísticas de completude:")
            for col in ['phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 
                       'umag', 'gmag', 'rmag', 'imag', 'zmag']:
                if col in df.columns:
                    n_valid = df[col].notna().sum()
                    pct_valid = 100 * n_valid / len(df)
                    print(f"   {col:20s}: {n_valid:7,} ({pct_valid:5.1f}%)")
            
            return df
            
    except Exception as e:
        print(f"❌ Erro ao abrir arquivo: {e}")
        print(f"\n💡 SOLUÇÕES:")
        print(f"   1. Usar método alternativo (via astropy.Table)")
        print(f"   2. Re-baixar o arquivo completo")
        print(f"   3. Converter para CSV primeiro")
        raise# -*- coding: utf-8 -*-
"""
Análise de Catálogo Fotométrico - Gaia EDR3 WD + SDSS
Extrai e visualiza SEDs fotométricos (sem convolução espectral)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import os

# ==================== MAPEAMENTO DE FILTROS ====================
# Comprimentos de onda efetivos (Å) para cada filtro
FILTER_WAVELENGTHS = {
    'phot_g_mean_mag': 6231,           # Gaia G
    'phot_bp_mean_mag': 5110,          # Gaia BP
    'phot_rp_mean_mag': 7770,          # Gaia RP
    'umag': 3543,                      # SDSS u
    'gmag': 4770,                      # SDSS g
    'rmag': 6231,                      # SDSS r
    'imag': 7625,                      # SDSS i
    'zmag': 9134,                      # SDSS z
}

FILTER_COLORS = {
    'phot_bp_mean_mag': '#4477FF',     # Gaia BP - azul
    'phot_g_mean_mag': '#44DD44',      # Gaia G - verde
    'phot_rp_mean_mag': '#DD4444',     # Gaia RP - vermelho
    'umag': '#7F7FFF',                 # SDSS u
    'gmag': '#44DD44',                 # SDSS g
    'rmag': '#FF7F7F',                 # SDSS r
    'imag': '#DD7777',                 # SDSS i
    'zmag': '#AA5555',                 # SDSS z
}

FILTER_NAMES = {
    'phot_g_mean_mag': 'Gaia G',
    'phot_bp_mean_mag': 'Gaia BP',
    'phot_rp_mean_mag': 'Gaia RP',
    'umag': 'SDSS u',
    'gmag': 'SDSS g',
    'rmag': 'SDSS r',
    'imag': 'SDSS i',
    'zmag': 'SDSS z',
}


# ==================== FUNÇÕES DE LEITURA ====================

def diagnose_fits_file(catalog_path: str):
    """
    Diagnostica problemas no arquivo FITS
    """
    print("🔍 DIAGNÓSTICO DO ARQUIVO FITS")
    print("=" * 60)
    
    import os
    file_size = os.path.getsize(catalog_path)
    print(f"📦 Tamanho real: {file_size / (1024**2):.1f} MB ({file_size:,} bytes)")
    
    try:
        with fits.open(catalog_path, memmap=True, ignore_missing_end=True) as hdul:
            print(f"✅ Arquivo abriu com sucesso")
            print(f"📋 HDUs: {len(hdul)}")
            
            for i, hdu in enumerate(hdul):
                print(f"\nHDU {i}: {hdu.name}")
                if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                    n_rows = hdu.header.get('NAXIS2', 0)
                    n_cols = len(hdu.columns.names)
                    row_bytes = hdu.header.get('NAXIS1', 0)
                    
                    expected_size = n_rows * row_bytes
                    
                    print(f"   Linhas esperadas: {n_rows:,}")
                    print(f"   Colunas: {n_cols}")
                    print(f"   Bytes por linha: {row_bytes:,}")
                    print(f"   Tamanho esperado da tabela: {expected_size / (1024**2):.1f} MB")
                    
                    # Testar leitura da primeira linha
                    try:
                        first_row = hdu.data[0]
                        print(f"   ✅ Primeira linha acessível")
                        
                        # Verificar endianness
                        for col_name in hdu.columns.names[:5]:
                            col_data = hdu.data.field(col_name)[0]
                            if hasattr(col_data, 'dtype'):
                                endian = col_data.dtype.byteorder
                                if endian == '>':
                                    print(f"   ⚠️ Coluna '{col_name}': Big-endian detectado")
                                elif endian == '<':
                                    print(f"   ✓ Coluna '{col_name}': Little-endian (nativo)")
                                
                    except Exception as e:
                        print(f"   ❌ Erro ao acessar primeira linha: {e}")
                    
                    # Tentar determinar quantas linhas são acessíveis
                    print(f"   🔍 Testando acessibilidade...")
                    accessible = 0
                    test_points = [100, 1000, 10000, 50000, 100000]
                    for test in test_points:
                        if test >= n_rows:
                            break
                        try:
                            _ = hdu.data[test]
                            accessible = test + 1
                        except:
                            break
                    
                    if accessible > 0:
                        print(f"   ✅ Pelo menos {accessible:,} linhas acessíveis")
                    else:
                        print(f"   ⚠️ Apenas primeira linha acessível")
    
    except Exception as e:
        print(f"❌ Erro: {e}")


def load_photometry_catalog(catalog_path: str, max_rows: int = None, 
                           chunk_size: int = 10000) -> pd.DataFrame:
    """
    Carrega catálogo fotométrico do FITS para DataFrame (com proteção contra truncamento)
    
    Args:
        catalog_path: Caminho para o arquivo FITS
        max_rows: Número máximo de linhas (None = todas disponíveis)
        chunk_size: Tamanho do chunk para leitura segura
    
    Returns:
        DataFrame com dados fotométricos
    """
    print(f"📂 Carregando catálogo: {catalog_path}")
    
    # Verificar tamanho do arquivo
    file_size_mb = os.path.getsize(catalog_path) / (1024**2)
    print(f"📦 Tamanho do arquivo: {file_size_mb:.1f} MB")
    
    try:
        with fits.open(catalog_path, memmap=True, ignore_missing_end=True) as hdul:
            # Pegar HDU com dados (geralmente índice 1)
            data_hdu = hdul[1]
            
            n_header = data_hdu.header['NAXIS2']
            print(f"📊 Objetos esperados (header): {n_header}")
            
            # Colunas de interesse
            columns_to_load = [
                'WDJ_name', 'source_id', 'ra', 'dec', 'parallax',
                'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag',
                'umag', 'gmag', 'rmag', 'imag', 'zmag',
                'e_umag', 'e_gmag', 'e_rmag', 'e_imag', 'e_zmag',
                'teff_H', 'logg_H', 'mass_H', 'Pwd'
            ]
            
            # Verificar quais colunas existem
            available_cols = [col for col in columns_to_load 
                            if col in data_hdu.columns.names]
            print(f"📋 Colunas disponíveis: {len(available_cols)}/{len(columns_to_load)}")
            
            # Tentar determinar quantas linhas realmente existem
            print(f"🔍 Detectando linhas realmente disponíveis...")
            
            # Ler em chunks para evitar buffer overflow
            all_data = []
            n_loaded = 0
            
            try:
                # Determinar limite real
                n_limit = max_rows if max_rows else n_header
                
                for start in range(0, n_limit, chunk_size):
                    end = min(start + chunk_size, n_limit)
                    
                    try:
                        # Tentar ler chunk
                        chunk_data = {}
                        for col in available_cols:
                            chunk_data[col] = data_hdu.data.field(col)[start:end]
                        
                        chunk_df = pd.DataFrame(chunk_data)
                        all_data.append(chunk_df)
                        n_loaded = end
                        
                        if (end % 50000) == 0:
                            print(f"  ✓ Carregado: {end:,} objetos")
                        
                    except Exception as e:
                        print(f"⚠️ Chunk {start}-{end} falhou: {e}")
                        print(f"  Parando na linha {n_loaded}")
                        break
                
            except Exception as e:
                print(f"⚠️ Erro durante leitura: {e}")
            
            if not all_data:
                raise ValueError("Nenhum dado pôde ser carregado")
            
            # Combinar chunks
            df = pd.concat(all_data, ignore_index=True)
            
            print(f"✅ Catálogo carregado: {len(df):,} objetos (de {n_header:,} esperados)")
            print(f"📊 Colunas: {len(df.columns)}")
            
            # Estatísticas de completude
            if len(df) < n_header:
                pct = 100 * len(df) / n_header
                print(f"⚠️ Arquivo truncado: {pct:.1f}% dos dados disponíveis")
            
            return df
            
    except Exception as e:
        print(f"❌ Erro ao abrir arquivo: {e}")
        print(f"\n💡 SOLUÇÕES:")
        print(f"   1. Re-baixe o arquivo (pode estar corrompido)")
        print(f"   2. Verifique a integridade com: md5sum {catalog_path}")
        print(f"   3. Tente baixar versão comprimida (.fits.gz)")
        raise


def extract_sed(df: pd.DataFrame, index: int) -> dict:
    """
    Extrai SED fotométrico de um objeto
    
    Args:
        df: DataFrame com dados fotométricos
        index: Índice do objeto
    
    Returns:
        Dicionário com wavelengths, magnitudes e erros
    """
    row = df.iloc[index]
    
    sed = {
        'wavelengths': [],
        'magnitudes': [],
        'errors': [],
        'filters': [],
        'obj_name': row.get('WDJ_name', f"WD_{index}")
    }
    
    # Gaia (sem erros no catálogo)
    gaia_filters = ['phot_bp_mean_mag', 'phot_g_mean_mag', 'phot_rp_mean_mag']
    for filt in gaia_filters:
        if filt in row and pd.notna(row[filt]):
            sed['wavelengths'].append(FILTER_WAVELENGTHS[filt])
            sed['magnitudes'].append(row[filt])
            sed['errors'].append(np.nan)
            sed['filters'].append(filt)
    
    # SDSS (com erros)
    sdss_filters = ['umag', 'gmag', 'rmag', 'imag', 'zmag']
    for filt in sdss_filters:
        if filt in row and pd.notna(row[filt]) and row[filt] > 0:
            sed['wavelengths'].append(FILTER_WAVELENGTHS[filt])
            sed['magnitudes'].append(row[filt])
            
            # Erro
            err_col = f'e_{filt}'
            err = row[err_col] if err_col in row and pd.notna(row[err_col]) else np.nan
            sed['errors'].append(err)
            sed['filters'].append(filt)
    
    # Ordenar por comprimento de onda
    if len(sed['wavelengths']) > 0:
        idx = np.argsort(sed['wavelengths'])
        sed['wavelengths'] = [sed['wavelengths'][i] for i in idx]
        sed['magnitudes'] = [sed['magnitudes'][i] for i in idx]
        sed['errors'] = [sed['errors'][i] for i in idx]
        sed['filters'] = [sed['filters'][i] for i in idx]
    
    return sed


# ==================== VISUALIZAÇÃO ====================

def plot_sed(sed: dict, save_path: str = None, show_info: bool = True, df: pd.DataFrame = None, index: int = None):
    """
    Plota SED fotométrico
    
    Args:
        sed: Dicionário com dados do SED
        save_path: Caminho para salvar figura
        show_info: Se True, mostra informações do objeto
        df: DataFrame original (para mostrar parâmetros)
        index: Índice do objeto no DataFrame
    """
    if len(sed['wavelengths']) == 0:
        print(f"⚠️ Sem dados fotométricos para {sed['obj_name']}")
        return
    
    # Figura com espaço extra para título
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Cores dos pontos
    colors = [FILTER_COLORS[f] for f in sed['filters']]
    
    # Linha conectando pontos
    ax.plot(sed['wavelengths'], sed['magnitudes'], 
            color='lightgray', linewidth=2, zorder=1, alpha=0.5)
    
    # Pontos com barras de erro
    valid_errors = [e if not np.isnan(e) else 0 for e in sed['errors']]
    ax.errorbar(sed['wavelengths'], sed['magnitudes'], yerr=valid_errors,
                fmt='o', markersize=10, elinewidth=2, capsize=4,
                c='none', ecolor='gray', alpha=0.5, zorder=2)
    
    # Pontos coloridos
    ax.scatter(sed['wavelengths'], sed['magnitudes'], 
              c=colors, s=150, edgecolor='black', linewidth=1.5, 
              alpha=1.0, zorder=3)
    
    # Labels dos filtros (ajustados para não sobrepor)
    for i, (wave, mag, filt) in enumerate(zip(sed['wavelengths'], 
                                               sed['magnitudes'], 
                                               sed['filters'])):
        label = FILTER_NAMES[filt]
        # Alternar posição para evitar sobreposição
        offset = -0.4 if i % 2 == 0 else 0.4
        ax.text(wave, mag + offset, label, ha='center', 
               va='bottom' if offset > 0 else 'top',
               fontsize=8, style='italic', alpha=0.7)
    
    # Configurações do gráfico
    ax.invert_yaxis()
    ax.set_xlabel('Comprimento de onda [Å]', fontsize=13)
    ax.set_ylabel('Magnitude [AB]', fontsize=13)
    
    # Título principal
    title_main = f"{sed['obj_name']}"
    
    # Informações físicas em texto separado
    info_text = None
    if show_info and df is not None and index is not None:
        row = df.iloc[index]
        info_parts = []
        if 'teff_H' in row and pd.notna(row['teff_H']):
            info_parts.append(f"$T_{{eff}}$ = {row['teff_H']:.0f} K")
        if 'logg_H' in row and pd.notna(row['logg_H']):
            info_parts.append(f"log g = {row['logg_H']:.2f}")
        if 'mass_H' in row and pd.notna(row['mass_H']):
            info_parts.append(f"M = {row['mass_H']:.2f} $M_\\odot$")
        
        if info_parts:
            info_text = " | ".join(info_parts)
    
    # Título com suptitle para melhor controle
    fig.suptitle(title_main, fontsize=15, fontweight='bold', y=0.98)
    if info_text:
        ax.set_title(info_text, fontsize=11, pad=10, style='italic')
    else:
        ax.set_title("SED Fotométrico", fontsize=11, pad=10)
    
    ax.grid(alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=11)
    
    # Ajustar layout manualmente
    plt.subplots_adjust(top=0.92, bottom=0.10, left=0.10, right=0.95)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Figura salva: {save_path}")
    
    plt.show()
    plt.close(fig)


def plot_multiple_seds(df: pd.DataFrame, indices: list, save_path: str = None):
    """
    Plota múltiplos SEDs em um único gráfico
    
    Args:
        df: DataFrame com dados fotométricos
        indices: Lista de índices dos objetos
        save_path: Caminho para salvar figura
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Usar colormap
    colors_map = plt.cm.viridis(np.linspace(0, 1, len(indices)))
    
    for i, idx in enumerate(indices):
        sed = extract_sed(df, idx)
        
        if len(sed['wavelengths']) == 0:
            continue
        
        # Pegar nome curto para legenda
        obj_name = sed['obj_name']
        if len(obj_name) > 20:
            obj_name = obj_name[:17] + "..."
        
        ax.plot(sed['wavelengths'], sed['magnitudes'], 
               'o-', color=colors_map[i], markersize=7, linewidth=2,
               label=obj_name, alpha=0.8)
    
    ax.invert_yaxis()
    ax.set_xlabel('Comprimento de onda [Å]', fontsize=13)
    ax.set_ylabel('Magnitude [AB]', fontsize=13)
    ax.set_title('Comparação de SEDs Fotométricos — Anãs Brancas', fontsize=15, pad=15)
    ax.legend(fontsize=9, ncol=2, loc='best', framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=11)
    
    plt.subplots_adjust(top=0.95, bottom=0.08, left=0.08, right=0.96)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Figura salva: {save_path}")
    
    plt.show()
    plt.close(fig)


def create_sed_catalog(df: pd.DataFrame, output_file: str = "sed_catalog.csv"):
    """
    Cria catálogo CSV com todos os SEDs
    
    Args:
        df: DataFrame com dados fotométricos
        output_file: Nome do arquivo de saída
    """
    print(f"\n📊 Criando catálogo de SEDs...")
    
    results = []
    
    for i in range(len(df)):
        if (i + 1) % 1000 == 0:
            print(f"  Processado: {i+1}/{len(df)} ({100*(i+1)/len(df):.1f}%)")
        
        sed = extract_sed(df, i)
        
        if len(sed['wavelengths']) == 0:
            continue
        
        row_data = {
            'index': i,
            'name': sed['obj_name'],
            'n_filters': len(sed['wavelengths'])
        }
        
        # Adicionar magnitudes
        for filt, mag in zip(sed['filters'], sed['magnitudes']):
            row_data[f'mag_{filt}'] = mag
        
        results.append(row_data)
    
    df_sed = pd.DataFrame(results)
    df_sed.to_csv(output_file, index=False)
    
    print(f"✅ Catálogo salvo: {output_file}")
    print(f"📊 Total de objetos com fotometria: {len(df_sed)}")
    
    return df_sed


# ==================== ANÁLISE ESTATÍSTICA ====================

def plot_color_magnitude_diagram(df: pd.DataFrame, save_path: str = None):
    """
    Cria diagrama cor-magnitude (HR diagram simplificado)
    """
    # Filtrar objetos com Gaia G e BP-RP
    mask = df['phot_g_mean_mag'].notna() & df['phot_bp_mean_mag'].notna() & df['phot_rp_mean_mag'].notna()
    df_clean = df[mask].copy()
    
    # Calcular cor BP-RP
    df_clean['bp_rp_calc'] = df_clean['phot_bp_mean_mag'] - df_clean['phot_rp_mean_mag']
    
    # Calcular magnitude absoluta se tiver paralaxe
    if 'parallax' in df_clean.columns:
        mask_plx = df_clean['parallax'] > 0
        df_clean.loc[mask_plx, 'abs_G'] = df_clean.loc[mask_plx, 'phot_g_mean_mag'] + 5 * np.log10(df_clean.loc[mask_plx, 'parallax'] / 100)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # CMD aparente
    axes[0].scatter(df_clean['bp_rp_calc'], df_clean['phot_g_mean_mag'], 
                   s=1, alpha=0.3, c='royalblue')
    axes[0].invert_yaxis()
    axes[0].set_xlabel('BP - RP [mag]', fontsize=12)
    axes[0].set_ylabel('G [mag]', fontsize=12)
    axes[0].set_title('Diagrama Cor-Magnitude (aparente)', fontsize=14)
    axes[0].grid(alpha=0.3)
    
    # CMD absoluto (se tiver paralaxe)
    if 'abs_G' in df_clean.columns:
        mask_abs = df_clean['abs_G'].notna()
        axes[1].scatter(df_clean.loc[mask_abs, 'bp_rp_calc'], 
                       df_clean.loc[mask_abs, 'abs_G'],
                       s=1, alpha=0.3, c='darkred')
        axes[1].invert_yaxis()
        axes[1].set_xlabel('BP - RP [mag]', fontsize=12)
        axes[1].set_ylabel('M_G [mag]', fontsize=12)
        axes[1].set_title('Diagrama Cor-Magnitude (absoluto)', fontsize=14)
        axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


# ==================== EXEMPLO DE USO ====================

if __name__ == "__main__":
    
    catalog_path = "UFS-IC-Astrofisica/WD/GaiaEDR3_WD_main.fits"
    
    # 1. Carregar catálogo (começar com subset para teste)
    print("=" * 60)
    print("CARREGANDO CATÁLOGO")
    print("=" * 60)
    df = load_photometry_catalog(catalog_path, max_rows=1000)  # teste com 1000
    
    print(f"\n📊 Primeiras linhas do catálogo:")
    print(df[['WDJ_name', 'phot_g_mean_mag', 'umag', 'gmag', 'rmag']].head())
    
    # 2. Extrair e plotar alguns SEDs
    print("\n" + "=" * 60)
    print("PLOTANDO SEDs INDIVIDUAIS")
    print("=" * 60)
    
    # Plotar primeiros 5 objetos
    for i in range(5):
        sed = extract_sed(df, i)
        plot_sed(sed, save_path=f"SED_WD_{i:03d}.pdf", 
                show_info=True, df=df, index=i)
    
    # 3. Comparação múltipla
    print("\n" + "=" * 60)
    print("COMPARANDO MÚLTIPLOS SEDs")
    print("=" * 60)
    plot_multiple_seds(df, indices=list(range(10)), 
                      save_path="SED_comparison.pdf")
    
    # 4. Diagrama cor-magnitude
    print("\n" + "=" * 60)
    print("DIAGRAMA COR-MAGNITUDE")
    print("=" * 60)
    plot_color_magnitude_diagram(df, save_path="CMD.pdf")
    
    # 5. Criar catálogo completo (opcional)
    # df_full = load_photometry_catalog(catalog_path, max_rows=None)
    # create_sed_catalog(df_full, "WD_sed_catalog.csv")
    
    print("\n✅ Análise completa!")