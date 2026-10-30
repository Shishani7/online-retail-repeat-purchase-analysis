"""Download the original UCI workbook and convert it to the expected local CSV."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

SOURCE_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
ROOT = Path(__file__).resolve().parents[2]


def download_data(output=ROOT / "data/raw/online_retail.csv", force=False):
    output = Path(output)
    if output.exists() and not force:
        return output
    with urlopen(SOURCE_URL, timeout=120) as response:
        payload = response.read()
    with ZipFile(io.BytesIO(payload)) as archive:
        workbooks = [name for name in archive.namelist() if name.lower().endswith(".xlsx")]
        if len(workbooks) != 1:
            raise ValueError("Expected one UCI workbook")
        frame = pd.read_excel(io.BytesIO(archive.read(workbooks[0])), engine="openpyxl")
    required = {"InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"}
    if set(frame.columns) != required or len(frame) != 541909:
        raise ValueError("Unexpected UCI data schema or row count")
    frame["InvoiceDate"] = frame["InvoiceDate"].dt.strftime("%m/%d/%Y %H:%M")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, lineterminator="\n")
    metadata = {
        "source": SOURCE_URL,
        "citation": "Chen, D. (2015). Online Retail. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33",
        "license": "CC BY 4.0",
        "rows": len(frame),
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
        "csv_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    (output.parent / "source.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(download_data(force=args.force))
