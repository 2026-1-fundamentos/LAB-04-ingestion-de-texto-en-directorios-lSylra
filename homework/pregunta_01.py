# pylint: disable=import-outside-toplevel
# pylint: disable=line-too-long
# flake8: noqa
"""
Escriba el codigo que ejecute la accion solicitada en cada pregunta.
"""


def pregunta_01():
    """
    La información requerida para este laboratio esta almacenada en el
    archivo "files/input.zip" ubicado en la carpeta raíz.
    Descomprima este archivo.

    Como resultado se creara la carpeta "input" en la raiz del
    repositorio, la cual contiene la siguiente estructura de archivos:


    ```
    train/
        negative/
            0000.txt
            0001.txt
            ...
        positive/
            0000.txt
            0001.txt
            ...
        neutral/
            0000.txt
            0001.txt
            ...
    test/
        negative/
            0000.txt
            0001.txt
            ...
        positive/
            0000.txt
            0001.txt
            ...
        neutral/
            0000.txt
            0001.txt
            ...
    ```

    A partir de esta informacion escriba el código que permita generar
    dos archivos llamados "train_dataset.csv" y "test_dataset.csv". Estos
    archivos deben estar ubicados en la carpeta "output" ubicada en la raiz
    del repositorio.

    Estos archivos deben tener la siguiente estructura:

    * phrase: Texto de la frase. hay una frase por cada archivo de texto.
    * sentiment: Sentimiento de la frase. Puede ser "positive", "negative"
      o "neutral". Este corresponde al nombre del directorio donde se
      encuentra ubicado el archivo.

    Cada archivo tendria una estructura similar a la siguiente:

    ```
    |    | phrase                                                                                                                                                                 | target   |
    |---:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------|
    |  0 | Cardona slowed her vehicle , turned around and returned to the intersection , where she called 911                                                                     | neutral  |
    |  1 | Market data and analytics are derived from primary and secondary research                                                                                              | neutral  |
    |  2 | Exel is headquartered in Mantyharju in Finland                                                                                                                         | neutral  |
    |  3 | Both operating profit and net sales for the three-month period increased , respectively from EUR16 .0 m and EUR139m , as compared to the corresponding quarter in 2006 | positive |
    |  4 | Tampere Science Parks is a Finnish company that owns , leases and builds office properties and it specialises in facilities for technology-oriented businesses         | neutral  |
    ```


    """
    import zipfile
    from pathlib import Path
    from typing import List, Tuple

    import pandas as pd


    def extract_input_zip(zip_path: Path, dest: Path) -> None:
        if dest.exists():
            return
        if not zip_path.exists():
            raise FileNotFoundError(f"Zip file not found: {zip_path}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)


    def read_split(input_root: Path, split: str) -> pd.DataFrame:
        rows: List[Tuple[str, str]] = []
        split_dir = input_root / split
        if not split_dir.exists():
            return pd.DataFrame(columns=["phrase", "target"])

        for target_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
            target = target_dir.name
            for txt_file in sorted(target_dir.glob("*.txt")):
                try:
                    text = txt_file.read_text(encoding="utf-8").strip()
                except Exception:
                    # fallback a latin-1 si utf-8 falla
                    text = txt_file.read_text(encoding="latin-1").strip()
                rows.append((text, target))

        df = pd.DataFrame(rows, columns=["phrase", "target"])
        return df


    def save_dataframe(df: pd.DataFrame, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)


    # Rutas
    repo_root = Path(".")
    zip_path = repo_root / "files" / "input.zip"
    input_dest = repo_root / "files" / "input"
    output_dir = repo_root / "files" / "output"

    # Extraer si es necesario
    extract_input_zip(zip_path, input_dest)

    # Determinar la carpeta que contiene directamente 'train' y 'test'.
    if (input_dest / "train").exists():
        actual_input_root = input_dest
    else:
        # buscar un subdirectorio que contenga 'train'
        actual_input_root = None
        for child in sorted([p for p in input_dest.iterdir() if p.is_dir()]):
            if (child / "train").exists():
                actual_input_root = child
                break
        if actual_input_root is None:
            actual_input_root = input_dest

    # Generar datasets
    train_df = read_split(actual_input_root, "train")
    test_df = read_split(actual_input_root, "test")

    save_dataframe(train_df, output_dir / "train_dataset.csv")
    save_dataframe(test_df, output_dir / "test_dataset.csv")