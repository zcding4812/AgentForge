from __future__ import annotations

from llama_index.readers.file import PandasCSVReader, PandasExcelReader

from app.knowledge.adapters.document_text._llama_util import join_llama_documents
from app.knowledge.adapters.document_text._tempfile import bytes_as_temp_file


class ExcelHandler:
    name = "excel"
    priority = 20

    def matches(self, mime: str, filename: str) -> bool:
        fn = filename.lower()
        return mime in (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel",
        ) or fn.endswith((".xlsx", ".xls"))

    def extract(self, raw: bytes, mime: str, filename: str) -> str:
        suf = ".xlsx" if filename.lower().endswith(".xlsx") else ".xls"
        with bytes_as_temp_file(raw, suf) as path:
            docs = PandasExcelReader(concat_rows=True).load_data(path)
            return join_llama_documents(docs)


class CsvHandler:
    name = "csv"
    priority = 22

    def matches(self, mime: str, filename: str) -> bool:
        fn = filename.lower()
        return mime in ("text/csv", "application/csv") or fn.endswith(".csv")

    def extract(self, raw: bytes, mime: str, filename: str) -> str:
        with bytes_as_temp_file(raw, ".csv") as path:
            for enc in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
                try:
                    docs = PandasCSVReader(
                        concat_rows=True,
                        pandas_config={"encoding": enc},
                    ).load_data(path)
                    return join_llama_documents(docs)
                except UnicodeDecodeError:
                    continue
            docs = PandasCSVReader(concat_rows=True, pandas_config={}).load_data(path)
            return join_llama_documents(docs)
