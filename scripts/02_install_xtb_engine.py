import zipfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
XTB_DIR = ROOT / "data" / "external" / "xtb"
XTB_ZIP = XTB_DIR / "xtb_win.zip"
XTB_URL = "https://github.com/grimme-lab/xtb/releases/download/v6.6.1/xtb-6.6.1-win64.zip"

def download_and_extract():
    XTB_DIR.mkdir(parents=True, exist_ok=True)
    xtb_exe = XTB_DIR / "bin" / "xtb.exe"
    
    found_exes = list(XTB_DIR.glob("**/xtb.exe"))
    if found_exes:
        print(f" GFN2-xTB binary ready at: {found_exes[0]}")
        return found_exes[0]

    print("Downloading precompiled GFN2-xTB Quantum Chemistry binary...")
    urllib.request.urlretrieve(XTB_URL, XTB_ZIP)
    print("Download complete. Extracting archive...")
    
    with zipfile.ZipFile(XTB_ZIP, "r") as zip_ref:
        zip_ref.extractall(XTB_DIR)
        
    found_exes = list(XTB_DIR.glob("**/xtb.exe"))
    if found_exes:
        print(f" GFN2-xTB binary ready: {found_exes[0]}")
        return found_exes[0]
    else:
        raise FileNotFoundError("xtb.exe not found after extraction.")

if __name__ == "__main__":
    download_and_extract()
