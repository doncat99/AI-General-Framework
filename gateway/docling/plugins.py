# gateway/docling/plugins.py
from __future__ import annotations

import json
import math
import re
import textwrap
from pathlib import Path
from statistics import median
from typing import Dict, List, Protocol

import numpy as np
from spellchecker import SpellChecker
from tabulate import tabulate

from docling.datamodel.document import ConversionResult
from docling_core.types.doc import (
    DoclingDocument,
    DocItemLabel,
    BoundingBox,
    ImageRefMode,
    GroupLabel,
)
from docling_core.types.doc.document import (
    DocItem,
    GroupItem,
    ListItem,
    TableItem,
    PictureItem,
    SectionHeaderItem,
    TextItem,
    ContentLayer,
)

from .types import Artifact, ArtifactType


DEFAULT_EXPORT_LABELS = {
    DocItemLabel.TITLE,
    DocItemLabel.DOCUMENT_INDEX,
    DocItemLabel.SECTION_HEADER,
    DocItemLabel.PARAGRAPH,
    DocItemLabel.TABLE,
    DocItemLabel.PICTURE,
    DocItemLabel.FORMULA,
    DocItemLabel.CHECKBOX_UNSELECTED,
    DocItemLabel.CHECKBOX_SELECTED,
    DocItemLabel.TEXT,
    DocItemLabel.LIST_ITEM,
    DocItemLabel.CODE,
    DocItemLabel.REFERENCE,
    DocItemLabel.PAGE_HEADER,
    DocItemLabel.PAGE_FOOTER,
    DocItemLabel.KEY_VALUE_REGION,
}


# --------------------------
# PostProcessor interface
# --------------------------
class PostProcessor(Protocol):
    name: str

    def process(self, base_dir: str, conv: ConversionResult) -> List[Artifact] | None: ...


# --------------------------
# Header / Footer detection
# --------------------------
class HeaderFooterDetector:
    @staticmethod
    def normalize_text(text: str) -> str:
        return " ".join(text.strip().lower().split())

    def __init__(self):
        # scoring weights
        self.weight_pos = 0.6
        self.weight_freq = 0.15
        self.weight_label = 0.05
        self.weight_consistency = 0.15
        self.weight_horiz_align = 0.05

        # tolerances
        self.t_tolerance = 20.0
        self.x_tolerance = 50.0
        self.height_tolerance = 5.0

    def detect(self, document: DoclingDocument) -> DoclingDocument:
        if not document.texts or not document.pages:
            return document

        (
            page_texts, all_texts,
            header_t_values, footer_t_values,
            l_values, r_values, heights,
        ) = self._collect_candidates(document)
        if not page_texts:
            return document

        ref_page_id = min(document.pages.keys())
        ref_page_width = document.pages[ref_page_id].size.width
        ref_page_height = document.pages[ref_page_id].size.height

        (
            header_t_min, header_t_max,
            footer_t_min, footer_t_max,
            l_min, l_max, r_min, r_max, avg_height,
        ) = self._define_regions(
            header_t_values, footer_t_values,
            l_values, r_values, heights,
            ref_page_height, ref_page_width,
        )

        text_t_values, text_l_values, text_spacing = self._collect_text_data(
            document.texts, page_texts
        )
        t_std = {t: np.std(vals) if len(vals) > 1 else 0.0 for t, vals in text_t_values.items()}
        l_std = {t: np.std(vals) if len(vals) > 1 else 0.0 for t, vals in text_l_values.items()}
        spacing_std = {t: np.std(vals) if len(vals) > 1 else 0.0 for t, vals in text_spacing.items()}

        from collections import Counter
        merged_counts = Counter(
            self.normalize_text(t.text) for t in all_texts if t.text.strip() and len(t.text) < 100
        )
        min_occurrence_threshold = 4

        for idx, item in enumerate(document.texts):
            if not getattr(item, "prov", None) or not item.prov or not getattr(item, "text", "").strip():
                continue
            if item.label == DocItemLabel.LIST_ITEM:
                continue

            prov = item.prov[0]
            bbox = prov.bbox
            page_h = (
                document.pages.get(prov.page_no).size.height
                if prov.page_no in document.pages
                else 783.0
            )
            if bbox.t < 0 or bbox.t > page_h:
                continue

            txt = self.normalize_text(item.text)
            is_labeled_candidate = item.label in [DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER]
            header_score, footer_score = self._compute_scores(
                bbox, txt, is_labeled_candidate, merged_counts.get(txt, 0),
                min_occurrence_threshold, t_std, l_std, spacing_std, page_h,
                header_t_min, header_t_max, footer_t_min, footer_t_max,
            )

            score_header_threshold = 0.6
            score_footer_threshold = 0.85
            if header_score >= score_header_threshold or footer_score >= score_footer_threshold:
                new_label = (
                    DocItemLabel.PAGE_HEADER if header_score > footer_score else DocItemLabel.PAGE_FOOTER
                )
                if item.label != new_label:
                    document.texts[idx].label = new_label
            else:
                if item.label in [DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER]:
                    document.texts[idx].label = DocItemLabel.TEXT
                    document.texts[idx].content_layer = ContentLayer.BODY

        return document

    def _collect_candidates(self, document: DoclingDocument):
        page_texts: Dict[int, List[TextItem]] = {}
        all_texts: List[TextItem] = []
        header_t_values: List[float] = []
        footer_t_values: List[float] = []
        l_values: List[float] = []
        r_values: List[float] = []
        heights: List[float] = []

        for text in document.texts:
            if not getattr(text, "prov", None) or not text.prov or not text.text.strip():
                continue
            prov = text.prov[0]
            if not hasattr(prov, "page_no") or not hasattr(prov, "bbox"):
                continue
            page_no = prov.page_no
            if page_no not in document.pages:
                continue
            bbox = prov.bbox
            if not all(hasattr(bbox, attr) for attr in ["t", "b", "l", "r"]):
                continue
            page_texts.setdefault(page_no, []).append(text)
            all_texts.append(text)
            if text.label in [DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER]:
                (header_t_values if text.label == DocItemLabel.PAGE_HEADER else footer_t_values).append(bbox.t)
                l_values.append(bbox.l)
                r_values.append(bbox.r)
                heights.append(bbox.t - bbox.b)

        return page_texts, all_texts, header_t_values, footer_t_values, l_values, r_values, heights

    def _define_regions(self, header_t_values, footer_t_values, l_values, r_values, heights, ref_h, ref_w):
        if header_t_values or footer_t_values:
            t_values = sorted(header_t_values + footer_t_values, reverse=True)
            if t_values:
                header_t_min = (median(t_values[: len(t_values) // 2]) - self.t_tolerance) if len(t_values) > 1 else t_values[0] - self.t_tolerance
                header_t_max = t_values[0] + self.t_tolerance
                footer_t_min = t_values[-1] - self.t_tolerance
                footer_t_max = (median(t_values[len(t_values) // 2 :]) + self.t_tolerance) if len(t_values) > 1 else t_values[-1] + self.t_tolerance
            else:
                header_t_min, header_t_max = ref_h * 0.9, ref_h
                footer_t_min, footer_t_max = 0, ref_h * 0.1
            l_min = max(0, min(l_values) - self.x_tolerance) if l_values else 0
            l_max = min(ref_w, max(l_values) + self.x_tolerance) if l_values else ref_w
            r_min = max(0, min(r_values) - self.x_tolerance) if r_values else 0
            r_max = min(ref_w, max(r_values) + self.x_tolerance) if r_values else ref_w
            avg_h = sum(heights) / len(heights) if heights else 10.0
        else:
            header_t_min, header_t_max = ref_h * 0.9, ref_h
            footer_t_min, footer_t_max = 0, ref_h * 0.1
            l_min, l_max, r_min, r_max, avg_h = 0, ref_w, 0, ref_w, 10.0

        return header_t_min, header_t_max, footer_t_min, footer_t_max, l_min, l_max, r_min, r_max, avg_h

    def _collect_text_data(self, texts, page_texts):
        text_t_values: Dict[str, List[float]] = {}
        text_l_values: Dict[str, List[float]] = {}
        text_spacing: Dict[str, List[float]] = {}

        for text in texts:
            if not getattr(text, "prov", None) or not text.prov or not text.text.strip():
                continue
            prov = text.prov[0]
            if not hasattr(prov, "page_no") or not hasattr(prov, "bbox"):
                continue
            bbox = prov.bbox
            if not all(hasattr(bbox, a) for a in ["t", "b", "l", "r"]):
                continue
            s = self.normalize_text(text.text)
            text_t_values.setdefault(s, []).append(bbox.t)
            text_l_values.setdefault(s, []).append(bbox.l)

        for _page_no, items in page_texts.items():
            sorted_items = sorted(items, key=lambda x: x.prov[0].bbox.t, reverse=True)
            for i, t in enumerate(sorted_items):
                s = self.normalize_text(t.text)
                if i + 1 < len(sorted_items):
                    nxt = sorted_items[i + 1]
                    spacing = t.prov[0].bbox.t - nxt.prov[0].bbox.b
                    text_spacing.setdefault(s, []).append(spacing)
                else:
                    text_spacing.setdefault(s, []).append(0.0)

        return text_t_values, text_l_values, text_spacing

    def _compute_scores(
        self,
        bbox: BoundingBox,
        text_str: str,
        is_labeled_candidate: bool,
        count: int,
        min_occ: int,
        t_std: Dict[str, float],
        l_std: Dict[str, float],
        spacing_std: Dict[str, float],
        page_h: float,
        header_t_min: float,
        header_t_max: float,
        footer_t_min: float,
        footer_t_max: float,
    ):
        t_normalized = bbox.t / page_h
        t_range = (header_t_max - footer_t_min) if header_t_max is not None and footer_t_min is not None else 0.2 * page_h
        sigma = t_range / 10
        header_pos_score = math.exp(-(abs(header_t_max - bbox.t) / sigma) ** 2) * self.weight_pos
        footer_pos_score = math.exp(-(abs(bbox.t - footer_t_min) / sigma) ** 2) * self.weight_pos
        if header_t_max is None and footer_t_min is None:
            header_pos_score = max(0, 1 - (1 - t_normalized) / 0.2) * self.weight_pos
            footer_pos_score = max(0, 1 - t_normalized / 0.2) * self.weight_pos

        freq_score = min(1, count / max(1, min_occ)) * self.weight_freq
        label_score = self.weight_label if is_labeled_candidate else 0.0
        consistency_score = max(0, 1 - t_std.get(text_str, 0.0) / 10.0) * self.weight_consistency
        horiz_align_score = max(0, 1 - l_std.get(text_str, 0.0) / 50.0) * self.weight_horiz_align

        header_score = header_pos_score + freq_score + label_score + consistency_score + horiz_align_score
        footer_score = footer_pos_score + freq_score + label_score + consistency_score + horiz_align_score
        return header_score, footer_score


class HeaderFooterDetectorProcessor:
    name = "header-footer-detector"

    def __init__(self):
        self._detector = HeaderFooterDetector()

    def process(self, base_dir: str, conv: ConversionResult):
        # do not emit artifacts—mutates document in-place
        conv.document = self._detector.detect(conv.document)
        return []


# --------------------------
# Label Correction
# --------------------------
class LabelCorrectionProcessor:
    name = "label-correction"

    def __init__(self, min_score: float = 4.5):
        self.spell = SpellChecker()
        self.min_score = min_score

    def _filter_illegal_labels(
        self,
        document: DoclingDocument,
        target_labels=(DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER),
    ) -> DoclingDocument:
        WEIGHTS = {
            "length_check": 2.0,
            "spell_check": 1.0,
            "word_count_check": 0.5,
            "alpha_check": 0.5,
            "no_end_punctuation": 0.5,
            "capitalized_word": 1.0,
            "uppercase_ratio": 0.5,
        }

        def is_meaningful_header(text: str, min_score: float) -> bool:
            txt = text.strip()
            score = 0.0

            if 3 <= len(txt) <= 100:
                score += WEIGHTS["length_check"]

            words = txt.split()
            cleaned = [re.sub(r"[^\w\s-]", "", w).lower() for w in words if w]
            if cleaned:
                unknown = self.spell.unknown(cleaned)
                spelled_ok = len(cleaned) - len(unknown)
                if len(cleaned) > 0 and spelled_ok / len(cleaned) >= 0.5:
                    score += WEIGHTS["spell_check"]

            if 1 <= len(words) <= 12:
                score += WEIGHTS["word_count_check"]

            alpha_chars = sum(c.isalpha() for c in txt)
            if alpha_chars > 0:
                score += WEIGHTS["alpha_check"]

            if txt and txt[-1] not in {".", "?", "!", ":"}:
                score += WEIGHTS["no_end_punctuation"]

            if txt and txt[0].isupper():
                score += WEIGHTS["capitalized_word"]

            if alpha_chars > 0:
                uppercase_chars = sum(c.isupper() for c in txt)
                uppercase_ratio = uppercase_chars / alpha_chars
                if uppercase_ratio <= 0.7:
                    score += WEIGHTS["uppercase_ratio"]

            return score >= min_score

        for idx, item in enumerate(document.texts):
            if (isinstance(item, TextItem) and item.label in target_labels) or isinstance(item, SectionHeaderItem):
                if not is_meaningful_header(item.text, self.min_score):
                    document.texts[idx].label = DocItemLabel.TEXT
        return document

    def process(self, base_dir: str, conv: ConversionResult):
        conv.document = self._filter_illegal_labels(conv.document)
        return []


# --------------------------
# Markdown Exporter
# --------------------------
# --------------------------
# Markdown Exporter (version-tolerant, no hard-coding)
# --------------------------
class MarkdownExporter:
    """Writes markdown produced by doc.export_to_markdown() to an artifact."""
    name = "markdown"

    def __init__(self, domtree_artifacts_folder: str = "artifacts"):
        # Keep the name configurable (used by other exporters / callers),
        # but do NOT pass it to export_to_markdown since this Docling build
        # doesn't accept a folder argument.
        self.domtree_artifacts_folder = domtree_artifacts_folder

    def process(self, base_dir: str, conv: ConversionResult) -> List[Artifact]:
        # Call with the signature supported by your current Docling version.
        md = conv.document.export_to_markdown(
            image_mode=ImageRefMode.REFERENCED
        )

        return [
            Artifact(
                type=ArtifactType.MARKDOWN,
                name=f"{conv.input.file.stem}.md",
                payload=md.encode("utf-8"),
            )
        ]


# --------------------------
# Doc JSON
# --------------------------
class JsonExporter:
    """Dumps the *raw* DoclingDocument dict (one-to-one with doc model)."""
    name = "json"

    def process(self, base_dir: str, conv: ConversionResult) -> List[Artifact]:
        doc_json = conv.document.export_to_dict()
        payload = json.dumps(doc_json, ensure_ascii=False, indent=2).encode("utf-8")
        return [
            Artifact(
                type=ArtifactType.DOC_JSON,
                name=f"{conv.input.file.stem}.doc.json",
                payload=payload,
            )
        ]


# --------------------------
# DomTree (structured content buckets)
# --------------------------
class DomTreeExporter:
    name = "domtree-exporter"

    def __init__(
        self,
        *,
        strict_text: bool = False,
        indent: int = 4,
        text_width: int = -1,
        page_no: int | None = None,
        domtree_artifacts_folder: str = "artifacts",
        merge_headers: bool = True,
        merge_distance: float = 50.0,
    ):
        self.strict_text = strict_text
        self.indent = indent
        self.text_width = text_width
        self.page_no = page_no
        self.domtree_artifacts_folder = domtree_artifacts_folder
        self.merge_headers = merge_headers
        self.merge_distance = merge_distance

    @staticmethod
    def _bbox_distance_bottom_to_top(item1, item2):
        try:
            b1 = item1.prov[0].bbox.b
            t2 = item2.prov[0].bbox.t
            d = t2 - b1
            return d if d > 0 else float("inf")
        except Exception:
            return float("inf")

    def _flush_mdtexts(self, bucket, mdtexts, page_no):
        if mdtexts:
            if not bucket:
                bucket.append(["Default", [["title", "Default\n", page_no]]])
            bucket[-1][1].append(["list_item", "".join(mdtexts), page_no])
            mdtexts.clear()

    def _collect_picture_children(self, picture_dict, full_doc_dict):
        out = []
        for ch in picture_dict.get("children", []):
            if isinstance(ch, dict) and "$ref" in ch:
                ref = ch["$ref"]
                ref_type, ref_num = ref.split("/")[-2:]
                if ref_type == "texts":
                    idx = int(ref_num)
                    t = full_doc_dict["texts"][idx]
                    entry = {"text": t.get("text", ""), "type": t.get("label", ""), "text_id": idx}
                    if t.get("orig") and t.get("orig") != t.get("text"):
                        entry["orig"] = t["orig"]
                    if "enumerated" in t:
                        entry["enumerated"] = t["enumerated"]
                    if "marker" in t:
                        entry["marker"] = t["marker"]
                    out.append(entry)
        return out

    def process(self, base_dir: str, conv: ConversionResult) -> List[Artifact]:
        d = conv.document
        labels = DEFAULT_EXPORT_LABELS

        mdtexts: List[str] = []
        list_nesting_level = 0
        previous_level = 0
        in_list = False
        in_list_page = 0
        document_content: List = []
        previous_header_buffer = None

        art_dir = Path(base_dir) / self.domtree_artifacts_folder
        art_dir.mkdir(parents=True, exist_ok=True)

        for ix, (item, level) in enumerate(d.iterate_items(d.body, with_groups=True, page_no=self.page_no)):
            def flush():
                self._flush_mdtexts(document_content, mdtexts, in_list_page)

            # level tracking for nested groups/lists
            if level < previous_level:
                list_nesting_level = max(0, list_nesting_level - (previous_level - level))
            previous_level = level

            # skip if not in export labels
            if isinstance(item, DocItem) and item.label not in labels:
                continue

            # close an open list when next item is not a list item/group
            if mdtexts and not isinstance(item, (ListItem, GroupItem)) and in_list:
                mdtexts[-1] += "\n"
                in_list = False

            if isinstance(item, GroupItem) and item.label in [GroupLabel.LIST, GroupLabel.ORDERED_LIST]:
                if list_nesting_level == 0:
                    mdtexts.append("\n")
                list_nesting_level += 1
                in_list = True
                continue
            elif isinstance(item, GroupItem):
                continue

            if isinstance(item, TextItem) and item.label == DocItemLabel.TITLE:
                in_list = False
                flush()
                document_content.append([item.text, [["title", f"{item.text}\n", item.prov[0].page_no]]])
                previous_header_buffer = None
                continue

            if (isinstance(item, TextItem) and item.label == DocItemLabel.SECTION_HEADER) or isinstance(item, SectionHeaderItem):
                if self.merge_headers and document_content:
                    if previous_header_buffer is None:
                        previous_header_buffer = item
                        in_list = False
                        flush()
                        document_content.append([item.text, [["section_header", f"{item.text}\n", item.prov[0].page_no]]])
                    else:
                        dpx = self._bbox_distance_bottom_to_top(previous_header_buffer, item)
                        if 0 < dpx < self.merge_distance:
                            merged = document_content[-1][0] + " " + item.text
                            document_content[-1][0] = merged
                            document_content[-1][1][-1][1] = f"{merged}\n"
                            previous_header_buffer = item
                        else:
                            previous_header_buffer = item
                            in_list = False
                            flush()
                            document_content.append([item.text, [["section_header", f"{item.text}\n", item.prov[0].page_no]]])
                else:
                    in_list = False
                    flush()
                    document_content.append([item.text, [["section_header", f"{item.text}\n", item.prov[0].page_no]]])
                continue

            if isinstance(item, TextItem) and item.label == DocItemLabel.CODE:
                in_list = False
                flush()
                code = f"```\n{item.text}\n```\n"
                if not document_content:
                    document_content.append(["Default", [["title", "Default\n", item.prov[0].page_no]]])
                document_content[-1][1].append(["code", code, item.prov[0].page_no])
                previous_header_buffer = None
                continue

            if isinstance(item, ListItem) and item.label == DocItemLabel.LIST_ITEM:
                in_list = True
                in_list_page = item.prov[0].page_no
                marker = item.marker if item.enumerated else "-"
                list_indent = " " * (self.indent * max(0, list_nesting_level - 1))
                mdtexts.append(f"{list_indent}{marker} {item.text}")
                previous_header_buffer = None
                continue

            if isinstance(item, TextItem) and item.label in labels:
                in_list = False
                flush()
                txt = (
                    textwrap.fill(item.text, width=self.text_width)
                    if (item.text and self.text_width > 0)
                    else f"{item.text}\n"
                )
                if not document_content:
                    document_content.append(["Default", [["title", "Default\n", item.prov[0].page_no]]])
                document_content[-1][1].append(["text", txt, item.prov[0].page_no])
                previous_header_buffer = None
                continue

            if isinstance(item, TableItem) and not self.strict_text:
                in_list = False
                flush()

                table_df = item.export_to_dataframe(d)
                csv_path = art_dir / f"table_{ix}.csv"
                png_path = art_dir / f"table_{ix}.png"
                try:
                    table_df.to_csv(csv_path, index=False)
                except Exception:
                    pass

                image = item.get_image(d)
                if image is not None:
                    with open(png_path, "wb") as fp:
                        image.save(fp, "PNG")

                table_json = item.model_dump()
                grid = table_json["data"]["grid"]
                rows = [[cell["text"] for cell in row] for row in grid]
                if rows and len(rows) > 1 and len(rows[0]) > 0:
                    try:
                        md = tabulate(rows[1:], headers=rows[0], tablefmt="github")
                    except Exception:
                        md = tabulate(rows[1:], headers=rows[0], tablefmt="github", disable_numparse=True)
                else:
                    md = tabulate(rows, tablefmt="github")

                html = item.export_to_html(doc=d)
                prov = table_json["prov"][0]
                bbox = prov["bbox"]
                payload = {
                    "table_id": int(table_json["self_ref"].split("/")[-1]),
                    "page": prov["page_no"],
                    "bbox": [bbox["l"], bbox["t"], bbox["r"], bbox["b"]],
                    "rows": table_json["data"]["num_rows"],
                    "cols": table_json["data"]["num_cols"],
                    "markdown": md,
                    "html": html,
                    "json": table_json,
                }
                if not document_content:
                    document_content.append(["Default", [["title", "Default\n", prov["page_no"]]]])
                document_content[-1][1].append(["table", [str(png_path), json.dumps(payload, indent=2, ensure_ascii=False)], prov["page_no"]])
                previous_header_buffer = None
                continue

            if isinstance(item, PictureItem) and not self.strict_text:
                in_list = False
                flush()

                prov = item.prov[0]
                img_path = art_dir / f"picture_{ix}.png"
                image = item.get_image(d)
                if image is not None:
                    with open(img_path, "wb") as fp:
                        image.save(fp, "PNG")

                picture_data = item.model_dump()
                bbox = picture_data["prov"][0]["bbox"]
                payload = {
                    "picture_id": int(picture_data["self_ref"].split("/")[-1]),
                    "page": picture_data["prov"][0]["page_no"],
                    "bbox": [bbox["l"], bbox["t"], bbox["r"], bbox["b"]],
                    "children": self._collect_picture_children(picture_data, d.export_to_dict()),
                }
                if not document_content:
                    document_content.append(["Default", [["title", "Default\n", prov.page_no]]])
                document_content[-1][1].append(["picture", [str(img_path), json.dumps(payload, indent=2, ensure_ascii=False)], prov.page_no])
                previous_header_buffer = None
                continue

        self._flush_mdtexts(document_content, mdtexts, in_list_page)
        payload = json.dumps(document_content, ensure_ascii=False, indent=2).encode("utf-8")
        return [Artifact(type=ArtifactType.DOMTREE_JSON, name=f"{conv.input.file.stem}.domtree.json", payload=payload)]


# --------------------------
# Deep JSON Report (rich, structured)
# --------------------------
class DeepJsonReportExporter:
    """
    Rich report:
      - metainfo
      - per-page content refs
      - full tables (md/html/json)
      - pictures (with child text refs)
    """
    name = "deep-json-report"

    def __init__(self, debug_dump_dir: str | None = None, domtree_artifacts_folder: str = "artifacts"):
        self.debug_dump_dir = Path(debug_dump_dir) if debug_dump_dir else None
        self.domtree_artifacts_folder = domtree_artifacts_folder

    def _table_to_md(self, table_json) -> str:
        grid = table_json["data"]["grid"]
        rows = [[cell["text"] for cell in row] for row in grid]
        if rows and len(rows) > 1 and len(rows[0]) > 0:
            try:
                return tabulate(rows[1:], headers=rows[0], tablefmt="github")
            except Exception:
                return tabulate(rows[1:], headers=rows[0], tablefmt="github", disable_numparse=True)
        return tabulate(rows, tablefmt="github")

    def _assemble_metainfo(self, data: dict) -> dict:
        origin = (data.get("origin") or {}).get("filename", "")
        sha1_name = origin.rsplit(".", 1)[0] if origin else ""
        return {
            "sha1_name": sha1_name,
            "pages_amount": len(data.get("pages", [])),
            "text_blocks_amount": len(data.get("texts", [])),
            "tables_amount": len(data.get("tables", [])),
            "pictures_amount": len(data.get("pictures", [])),
            "equations_amount": len(data.get("equations", [])),
            "footnotes_amount": len([t for t in data.get("texts", []) if t.get("label") == "footnote"]),
        }

    def _expand_groups(self, body_children, groups):
        expanded = []
        for item in body_children:
            if isinstance(item, dict) and "$ref" in item:
                ref = item["$ref"]
                ref_type, ref_num = ref.split("/")[-2:]
                ref_num = int(ref_num)
                if ref_type == "groups":
                    group = groups[ref_num]
                    gid = ref_num
                    gname = group.get("name", "")
                    glabel = group.get("label", "")
                    for ch in group["children"]:
                        cc = ch.copy()
                        cc["group_id"] = gid
                        cc["group_name"] = gname
                        cc["group_label"] = glabel
                        expanded.append(cc)
                else:
                    expanded.append(item)
            else:
                expanded.append(item)
        return expanded

    def _process_text_ref(self, idx: int, data: dict) -> dict:
        t = data["texts"][idx]
        out = {"text": t.get("text", ""), "type": t.get("label", ""), "text_id": idx}
        if t.get("orig") and t.get("orig") != t.get("text"):
            out["orig"] = t["orig"]
        if "enumerated" in t:
            out["enumerated"] = t["enumerated"]
        if "marker" in t:
            out["marker"] = t["marker"]
        return out

    def _assemble_content(self, data: dict) -> List[dict]:
        pages: Dict[int, dict] = {}
        body_children = (data.get("body") or {}).get("children", [])
        groups = data.get("groups", [])
        expanded = self._expand_groups(body_children, groups)

        for item in expanded:
            if not (isinstance(item, dict) and "$ref" in item):
                continue

            ref = item["$ref"]
            ref_type, ref_num = ref.split("/")[-2:]
            idx = int(ref_num)

            if ref_type == "texts":
                t = data["texts"][idx]
                c = self._process_text_ref(idx, data)
                if "group_id" in item:
                    c["group_id"] = item["group_id"]
                    c["group_name"] = item["group_name"]
                    c["group_label"] = item["group_label"]
                if "prov" in t and t["prov"]:
                    pno = t["prov"][0]["page_no"]
                    pages.setdefault(pno, {"page": pno, "content": [], "page_dimensions": t["prov"][0].get("bbox", {})})
                    pages[pno]["content"].append(c)

            elif ref_type == "tables":
                tb = data["tables"][idx]
                if "prov" in tb and tb["prov"]:
                    pno = tb["prov"][0]["page_no"]
                    pages.setdefault(pno, {"page": pno, "content": [], "page_dimensions": tb["prov"][0].get("bbox", {})})
                    pages[pno]["content"].append({"type": "table", "table_id": idx})

            elif ref_type == "pictures":
                pic = data["pictures"][idx]
                if "prov" in pic and pic["prov"]:
                    pno = pic["prov"][0]["page_no"]
                    pages.setdefault(pno, {"page": pno, "content": [], "page_dimensions": pic["prov"][0].get("bbox", {})})
                    pages[pno]["content"].append({"type": "picture", "picture_id": idx})

        return [pages[k] for k in sorted(pages.keys())]

    def _assemble_tables(self, conv_doc: DoclingDocument, data: dict) -> List[dict]:
        out: List[dict] = []
        for i, table in enumerate(conv_doc.tables):
            table_json = table.model_dump()
            md = self._table_to_md(table_json)
            html = table.export_to_html(doc=conv_doc)
            tb_data = data["tables"][i]
            prov = tb_data["prov"][0]
            bbox = prov["bbox"]
            out.append({
                "table_id": int(tb_data["self_ref"].split("/")[-1]),
                "page": prov["page_no"],
                "bbox": [bbox["l"], bbox["t"], bbox["r"], bbox["b"]],
                "#-rows": tb_data["data"]["num_rows"],
                "#-cols": tb_data["data"]["num_cols"],
                "markdown": md,
                "html": html,
                "json": table_json
            })
        return out

    def _assemble_pictures(self, data: dict) -> List[dict]:
        out: List[dict] = []
        for i, picture in enumerate(data.get("pictures", [])):
            prov = picture["prov"][0]
            bbox = prov["bbox"]
            children = []
            for ch in picture.get("children", []):
                if isinstance(ch, dict) and "$ref" in ch:
                    ref = ch["$ref"]
                    ref_type, ref_num = ref.split("/")[-2:]
                    if ref_type == "texts":
                        idx = int(ref_num)
                        children.append(self._process_text_ref(idx, data))
            out.append({
                "picture_id": int(picture["self_ref"].split("/")[-1]),
                "page": prov["page_no"],
                "bbox": [bbox["l"], bbox["t"], bbox["r"], bbox["b"]],
                "children": children
            })
        return out

    # --- main entry ---
    def process(self, base_dir: str, conv: ConversionResult) -> List[Artifact]:
        data = conv.document.export_to_dict()

        # optional raw dump for debugging
        if self.debug_dump_dir:
            self.debug_dump_dir.mkdir(parents=True, exist_ok=True)
            (self.debug_dump_dir / f"{conv.input.file.stem}.raw.json").write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        report = {
            "metainfo": self._assemble_metainfo(data),
            "content": self._assemble_content(data),
            "tables": self._assemble_tables(conv.document, data),
            "pictures": self._assemble_pictures(data),
        }
        payload = json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")
        return [
            Artifact(
                type=ArtifactType.JSON,
                name=f"{conv.input.file.stem}.deep_report.json",
                payload=payload,
            )
        ]
