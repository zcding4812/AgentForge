"""处理器注册表：新增格式时在此追加实例。"""

from __future__ import annotations

from typing import Final

from app.knowledge.adapters.document_text.handlers.csv_excel import CsvHandler, ExcelHandler
from app.knowledge.adapters.document_text.handlers.docx import DocxHandler
from app.knowledge.adapters.document_text.handlers.epub import EpubHandler
from app.knowledge.adapters.document_text.handlers.html import HtmlHandler
from app.knowledge.adapters.document_text.handlers.ipynb import IpynbHandler
from app.knowledge.adapters.document_text.handlers.pdf import PdfHandler
from app.knowledge.adapters.document_text.handlers.plain_text import PlainTextHandler
from app.knowledge.adapters.document_text.handlers.pptx import PptxHandler
from app.knowledge.adapters.document_text.handlers.rtf import RtfHandler
from app.knowledge.adapters.document_text.handlers.xml_reader import XmlHandler
from app.knowledge.adapters.document_text.protocol import DocumentTextHandler

HANDLERS: Final[tuple[DocumentTextHandler, ...]] = (
    PdfHandler(),
    DocxHandler(),
    PptxHandler(),
    ExcelHandler(),
    CsvHandler(),
    HtmlHandler(),
    RtfHandler(),
    EpubHandler(),
    IpynbHandler(),
    XmlHandler(),
    PlainTextHandler(),
)
