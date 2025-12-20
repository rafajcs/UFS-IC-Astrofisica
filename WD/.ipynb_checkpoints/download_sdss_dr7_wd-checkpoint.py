import requests
from astropy.table import Table
import pandas as pd

# URL DO CATÁLOGO — arquivo ASCII .dat no VizieR
url = "https://cdsarc.u-strasbg.fr/ftp/catv/J/other/ApJS/204/5/table2.dat"

local_file = "sdss_dr7_wd_catalog.dat"

# --- 1. Download do arquivo ---
print("📥 Baixando catálogo SDSS DR7 WD...")
r = requests.get(url)

if r.status_code != 200:
    raise RuntimeError(f"Erro ao baixar arquivo: HTTP {r.status_code}")

with open(local_file, "wb") as f:
    f.write(r.content)

print("✔️ Download concluído:", local_file)


# --- 2. Leitura do arquivo ASCII ---
print("\n📄 Lendo arquivo .dat... isso pode demorar alguns segundos...")
try:
    tbl = Table.read(local_file, format="ascii")
except Exception as e:
    print("Falha ao ler como 'ascii'. Tentando 'ascii.fixed_width_no_header'...")
    tbl = Table.read(local_file, format="ascii.fixed_width_no_header")

print("✔️ Arquivo lido com sucesso!")
print(f"Total de linhas: {len(tbl)}")
print("Colunas encontradas:")
print(tbl.colnames)


# --- 3. Converter para pandas (opcional) ---
df = tbl.to_pandas()
print("\nPrimeiras linhas do DataFrame:")
print(df.head())


# --- 4. Salvar em formatos úteis ---
df.to_csv("sdss_dr7_wd_catalog.csv", index=False)
tbl.write("sdss_dr7_wd_catalog.fits", overwrite=True)

print("\n📂 Arquivos gerados:")
print(" - sdss_dr7_wd_catalog.csv")
print(" - sdss_dr7_wd_catalog.fits")

print("\n🎉 Pronto! Catálogo carregado com sucesso.")
