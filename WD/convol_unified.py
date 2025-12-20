import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import constants as const
import os
import glob

# -------------------------------------------------------------
# CONSTANTES
# -------------------------------------------------------------
C_CGS = const.c.to('AA/s').value

# -------------------------------------------------------------
# FUNÇÃO PARA LER FILTRO
# -------------------------------------------------------------
def load_filter_csv(path):
    df = pd.read_csv(path, comment="#", header=None)
    wl = df.iloc[:,0].values.astype(float)
    thr = df.iloc[:,1].values.astype(float)

    if wl.max() < 2000:
        wl *= 10.0

    name = os.path.basename(path).replace(".csv","")
    return name, wl, thr

# -------------------------------------------------------------
# CARREGAR TODOS OS FILTROS S-PLUS
# -------------------------------------------------------------
def load_all_filters(filter_dir="splus_filters"):
    filters = {}
    files = glob.glob(os.path.join(filter_dir, "*.csv"))
    files = [f for f in files if "central_wavelength" not in f]

    for f in files:
        name, wl, thr = load_filter_csv(f)
        filters[name] = (wl, thr)

    return filters

# -------------------------------------------------------------
# CONVERTE MAG AB → fν → fλ
# -------------------------------------------------------------
def mag_to_flux_lambda(mag, lambda_eff):
    """Converte magnitude AB em f_lambda."""
    fnu = 10**(-0.4 * (mag + 48.6))   # erg/s/cm2/Hz
    return fnu * (C_CGS / lambda_eff**2)

# -------------------------------------------------------------
# CALCULA A SED A PARTIR DE FOTOMETRIA SDSS
# -------------------------------------------------------------
def build_sed_from_sdss(row):
    bands = ["u","g","r","i","z"]
    mags  = [row["umag"], row["gmag"], row["rmag"], row["imag"], row["zmag"]]

    # λ efetivo SDSS (fonte: documentação SDSS)
    lambda_eff = np.array([3543, 4770, 6231, 7625, 9134])

    mask = np.isfinite(mags)
    if mask.sum() < 3:
        return None, None

    lam = lambda_eff[mask]
    mags = np.array(mags)[mask]

    flux = np.zeros_like(mags, dtype=float)
    for i in range(len(mags)):
        flux[i] = mag_to_flux_lambda(mags[i], lam[i])

    return lam, flux

# -------------------------------------------------------------
# CONVOLUÇÃO DA SED COM UM FILTRO
# -------------------------------------------------------------
def convolve_photometry(lam_sed, flux_sed, filt_w, filt_t):
    # interpolar SED na grade do filtro
    flux_interp = np.interp(filt_w, lam_sed, flux_sed, left=0, right=0)

    denom = np.trapezoid(filt_t * filt_w, filt_w)
    if denom == 0:
        return np.nan

    num = np.trapezoid(flux_interp * filt_t * filt_w, filt_w)
    f_lambda_eff = num / denom

    # λ_eff real do filtro
    lambda_eff = np.trapezoid(filt_w * filt_t, filt_w) / np.trapezoid(filt_t, filt_w)

    # converter para mag AB
    fnu = f_lambda_eff * (lambda_eff**2) / C_CGS

    if fnu <= 0:
        return np.nan

    return -2.5 * np.log10(fnu) - 48.6

# -------------------------------------------------------------
# LOOP PRINCIPAL
# -------------------------------------------------------------
def run_convolution(csv_path, filters_dir="splus_filters", out_csv="synthetic_photometry.csv"):
    print("📂 Lendo catálogo unificado...")
    df = pd.read_csv(csv_path)

    print("📥 Carregando filtros S-PLUS...")
    filters = load_all_filters(filters_dir)

    results = []

    for idx, row in df.iterrows():
        lam, flux = build_sed_from_sdss(row)
        if lam is None:
            continue

        entry = {"SDSS": row["SDSS"], "RA": row["RA"], "Dec": row["Dec"]}

        for name, (fw, ft) in filters.items():
            mag_syn = convolve_photometry(lam, flux, fw, ft)
            entry[name+"_mag"] = mag_syn

        results.append(entry)

        if (idx+1) % 200 == 0:
            print(f"  ✓ {idx+1} / {len(df)} processados...")

    out = pd.DataFrame(results)
    out.to_csv(out_csv, index=False)

    print(f"\n✅ COMPLETO! Arquivo salvo em {out_csv}")
    return out

# -------------------------------------------------------------
# EXECUÇÃO DIRETA
# -------------------------------------------------------------
if __name__ == "__main__":
    run_convolution(
        "unified_catalog/SDSS_Gaia_unified_catalog.csv",
        filters_dir="splus_filters",
        out_csv="synthetic_photometry_unified.csv"
    )
