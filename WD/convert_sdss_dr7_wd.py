import pandas as pd
from astropy.table import Table


# ------------------------------------------------------
# 1) Definir colunas e posições conforme README
#    Byte positions -> 0-indexed slices para pandas
# ------------------------------------------------------
colspecs = [
    (0, 18),   # SDSS name
    (19, 23),  # Plate
    (24, 29),  # MJD
    (30, 33),  # Fiber
    (34, 46),  # RAdeg
    (47, 59),  # DEdeg
    (60, 69),  # SNg
    (70, 75),  # umag
    (76, 80),  # e_umag
    (81, 96),  # f_umag
    (97,102),  # gmag
    (103,108), # e_gmag
    (109,124), # f_gmag
    (125,130), # rmag
    (131,135), # e_rmag
    (136,151), # f_rmag
    (152,157), # imag
    (158,164), # e_imag
    (165,180), # f_imag
    (181,186), # zmag
    (187,192), # e_zmag
    (193,208), # f_zmag
    (209,219), # pm
    (220,230), # pmPA
    (231,232), # f_pm
    (236,244), # Ag
    (245,271), # GMT
    (272,286), # Atype
    (287,293), # Teff
    (294,299), # e_Teff
    (300,305), # logg
    (306,312), # e_logg
    (313,319), # chi2
    (320,324), # m_Nsp
    (324,328), # Nsp
    (328,329), # q_Nsp
    (330,335), # Mass
    (336,341), # e_Mass
    (342,361)  # Type
]

colnames = [
    "SDSS", "Plate", "MJD", "Fiber",
    "RAdeg", "DEdeg",
    "SNg",
    "umag", "e_umag", "f_umag",
    "gmag", "e_gmag", "f_gmag",
    "rmag", "e_rmag", "f_rmag",
    "imag", "e_imag", "f_imag",
    "zmag", "e_zmag", "f_zmag",
    "pm", "pmPA", "f_pm",
    "Ag", "GMT",
    "Atype", "Teff", "e_Teff",
    "logg", "e_logg",
    "chi2", "m_Nsp", "Nsp", "q_Nsp",
    "Mass", "e_Mass",
    "Type"
]

# ------------------------------------------------------
# 2) Ler usando colunas de largura fixa
# ------------------------------------------------------
df = pd.read_fwf("sdss_dr7_wd_catalog.dat", colspecs=colspecs, names=colnames)

# ------------------------------------------------------
# 3) Converter tipos numéricos (pandas lê tudo como string inicialmente)
# ------------------------------------------------------
float_cols = [
    "RAdeg","DEdeg","SNg","umag","e_umag","gmag","e_gmag","rmag","e_rmag",
    "imag","e_imag","zmag","e_zmag","pm","pmPA","Ag","Teff","e_Teff",
    "logg","e_logg","chi2","Mass","e_Mass"
]

int_cols = ["Plate","MJD","Fiber","f_umag","f_gmag","f_rmag","f_imag","f_zmag","Nsp"]

# converter
for c in float_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

for c in int_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

# ------------------------------------------------------
# 4) Preview
# ------------------------------------------------------
print(df.head())
print(df.info())

# ---
# Salvar como CSV (UTF-8, separador padrão)
df.to_csv("SDSS_DR7_WD_catalog.csv", index=False)
print("Arquivo CSV salvo como SDSS_DR7_WD_catalog.csv")



# Converter DataFrame → Astropy Table
tab = Table.from_pandas(df)

# Salvar como arquivo FITS
tab.write("SDSS_DR7_WD_catalog.fits", overwrite=True)
print("Arquivo FITS salvo como SDSS_DR7_WD_catalog.fits")


# ----
# 6) Test
# ----
check = Table.read("SDSS_DR7_WD_catalog.fits")
print(check)

check_csv = pd.read_csv("SDSS_DR7_WD_catalog.csv")
print(check_csv.head())

print(df.shape, check.to_pandas().shape)

