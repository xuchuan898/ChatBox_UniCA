import argparse
import difflib
import os
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    from docling.document_converter import DocumentConverter
except ImportError:
    DocumentConverter = None

try:
    import opendataloader_pdf
except ImportError:
    opendataloader_pdf = None


def convert_with_docling(pdf_path: str) -> str:
    if DocumentConverter is None:
        raise ImportError("docling is not installed. Install with: pip install docling")
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    return result.document.export_to_markdown()


def _find_generated_markdown(output_dir: Path) -> Path:
    candidates = sorted(
        [
            p
            for p in output_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".md", ".markdown"}
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No markdown output found under: {output_dir}")
    return candidates[0]


def convert_with_opendataloader(pdf_path: str, use_struct_tree: bool = False) -> str:
    if opendataloader_pdf is None:
        raise ImportError(
            "opendataloader-pdf is not installed. Install with: pip install -U opendataloader-pdf"
        )

    with TemporaryDirectory(prefix="opendataloader_pdf_") as tmp_dir:
        opendataloader_pdf.convert(
            input_path=[pdf_path],
            output_dir=tmp_dir,
            format="markdown",
            use_struct_tree=use_struct_tree,
        )
        md_file = _find_generated_markdown(Path(tmp_dir))
        return md_file.read_text(encoding="utf-8", errors="ignore")


def save_output(pdf_path: str, suffix: str, markdown_text: str) -> str:
    output_path = pdf_path.replace(".pdf", suffix)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(markdown_text)
    return output_path


def print_diff(original_md_path: str, converted_md_text: str, tofile_name: str) -> None:
    if not os.path.exists(original_md_path):
        print(f"Skip diff: original markdown not found: {original_md_path}")
        return

    with open(original_md_path, "r", encoding="utf-8") as handle:
        original_text = handle.readlines()

    diff = difflib.unified_diff(
        original_text,
        converted_md_text.splitlines(keepends=True),
        fromfile="Original.md",
        tofile=tofile_name,
    )
    print("\n--- Diff Result ---")
    for line in diff:
        print(line.rstrip("\n"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PDF to Markdown and compare with original markdown.")
    parser.add_argument("--pdf", type=str, default="docs/chroma/master.pdf", help="Input PDF path.")
    parser.add_argument("--original-md", type=str, default="docs/chroma/master.md", help="Reference markdown path.")
    parser.add_argument(
        "--tool",
        type=str,
        choices=["docling", "opendataloader", "both"],
        default="both",
        help="Which converter to run.",
    )
    parser.add_argument(
        "--use-struct-tree",
        action="store_true",
        help="Enable OpenDataLoader struct tree mode.",
    )
    parser.add_argument(
        "--show-diff",
        action="store_true",
        help="Print unified diff against --original-md.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.pdf):
        raise FileNotFoundError(f"PDF not found: {args.pdf}")

    if args.tool in {"docling", "both"}:
        print(f"Converting with Docling: {args.pdf}")
        try:
            docling_md = convert_with_docling(args.pdf)
            docling_out = save_output(args.pdf, "_docling_converted.md", docling_md)
            print(f"Docling output saved: {docling_out}")
            if args.show_diff:
                print_diff(args.original_md, docling_md, "Docling.md")
        except Exception as exc:
            print(f"Docling conversion failed: {exc}")

    if args.tool in {"opendataloader", "both"}:
        print(f"Converting with OpenDataLoader: {args.pdf}")
        try:
            odl_md = convert_with_opendataloader(args.pdf, use_struct_tree=args.use_struct_tree)
            odl_out = save_output(args.pdf, "_opendataloader_converted.md", odl_md)
            print(f"OpenDataLoader output saved: {odl_out}")
            if args.show_diff:
                print_diff(args.original_md, odl_md, "OpenDataLoader.md")
        except Exception as exc:
            print(f"OpenDataLoader conversion failed: {exc}")


if __name__ == "__main__":
    main()
