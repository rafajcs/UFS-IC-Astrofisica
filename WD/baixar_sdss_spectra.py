import os
import requests
import pandas as pd
from time import sleep

# ============================================================
# CONFIGURAÇÕES
# ============================================================

CATALOGO = "unified_catalog/SDSS_Gaia_unified_catalog.csv"
SAIDA = "sdss_spectra/"
MAX_TENTATIVAS = 3

# Endpoints conhecidos do SDSS:
URLS_SDSS = [
    # DR7 (mais comum para o catálogo de WD Kleinman+2013)
    "https://dr7.sdss.org/spectro/1d_26/{plate}/spec-{plate}-{mjd}-{fiber:04d}.fits",

    # DR12 (caso alguns objetos tenham espectro reprocessado lá)
    "https://dr12.sdss.org/sas/dr12/sdss/spectro/redux/{plate}/spectra/lite/spec-{plate}-{mjd}-{fiber:04d}.fits",

    # DR16 (último para SDS SPECTROGRAPH I)
    "https://data.sdss.org/sas/dr16/sdss/spectro/redux/{plate}/spectra/lite/spec-{plate}-{mjd}-{fiber:04d}.fits",
]

# ============================================================
# FUNÇÃO PARA TENTAR DOWNLOAD
# ============================================================

def tentar_baixar(plate, mjd, fiber, destino):
    """
    Tenta baixar um espectro de diferentes servidores SDSS.
    Retorna True se baixado com sucesso.
    """
    fiber_int = int(fiber)

    for url in URLS_SDSS:
        link = url.format(plate=int(plate), mjd=int(mjd), fiber=fiber_int)

        for tentativa in range(MAX_TENTATIVAS):
            try:
                r = requests.get(link, timeout=10)
                if r.status_code == 200:
                    with open(destino, "wb") as f:
                        f.write(r.content)
                    print(f"  ✔ Baixado: {destino}")
                    return True
                else:
                    print(f"  ✖ Falhou ({r.status_code}): {link}")
            except Exception as e:
                print(f"  ⚠ Erro: {e}")
            sleep(1)

    return False

# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    if not os.path.exists(SAIDA):
        os.makedirs(SAIDA)

    df = pd.read_csv(CATALOGO)

    # Verificando as colunas
    col_plate = "SDSS_plate" if "SDSS_plate" in df else "Plate"
    col_mjd   = "SDSS_mjd"   if "SDSS_mjd"   in df else "MJD"
    col_fiber = "SDSS_fiber" if "SDSS_fiber" in df else "Fiber"

    print("\n📌 Colunas detectadas:",
          col_plate, col_mjd, col_fiber)

    total = len(df)
    sucesso = 0
    falhas = 0

    for i, row in df.iterrows():
        plate = row[col_plate]
        mjd = row[col_mjd]
        fiber = row[col_fiber]

        destino = f"{SAIDA}/spec-{plate}-{mjd}-{int(fiber):04d}.fits"

        if os.path.exists(destino):
            print(f"[{i+1}/{total}] ✔ Já existe: {destino}")
            sucesso += 1
            continue

        print(f"[{i+1}/{total}] Tentando baixar Plate={plate}, MJD={mjd}, Fiber={fiber}")

        ok = tentar_baixar(plate, mjd, fiber, destino)

        if ok:
            sucesso += 1
        else:
            falhas += 1
            print(f"  ❌ Não encontrado em nenhum servidor.")

    print("\n================= RELATÓRIO FINAL =================")
    print(f"Total solicitado:  {total}")
    print(f"Baixados:          {sucesso}")
    print(f"Falharam:          {falhas}")
    print("Arquivos salvos em:", SAIDA)
    print("===================================================")

# ============================================================
if __name__ == "__main__":
    main()
