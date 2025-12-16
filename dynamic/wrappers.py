import subprocess
from pathlib import Path


def run_shelxt(hkl_file: str, shelxt_path: str = "shelxt"):
    """
    Run SHELXT on an .ins file and return the subprocess result.

    Parameters
    ----------
    hkl_file : str or Path
        Path to the input .hkl file.
    shelxt_path : str
        Path to the SHELXT executable (default assumes it is on PATH).

    Returns
    -------
    result : subprocess.CompletedProcess
        Contains stdout, stderr, return code, etc.
    """

    hkl_file = str(hkl_file)

    filename = hkl_file.replace('.hkl', '')
    ins_file = hkl_file.replace('.hkl', '.ins')

    ins_file = Path(ins_file)
    hkl_file = Path(hkl_file)
    if not ins_file.exists() or not hkl_file.exists():
        print('hkl or ins file not present. Ignoring')
        return None

    # SHELXT must run in the directory where the .ins file is
    cmd = f"{shelxt_path} {filename}"

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=False,   # we'll check manually
        )
    except Exception:
        print('Shelex failed')

    return result
