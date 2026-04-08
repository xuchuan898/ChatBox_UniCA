import os
import difflib
from docling.document_converter import DocumentConverter


def process_and_compare_with_docling(pdf_path, original_md_path):
    # 1. 初始化转换器
    converter = DocumentConverter()

    print(f"正在使用 IBM Docling 转换 {pdf_path}...")
    try:
        # 2. 执行转换
        result = converter.convert(pdf_path)
        # 3. 导出为 Markdown 格式
        converted_md_text = result.document.export_to_markdown()
    except Exception as e:
        print(f"转换失败: {e}")
        return

    # 4. 保存结果
    output_path = pdf_path.replace(".pdf", "_docling_converted.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(converted_md_text)

    print(f"转换完成！已保存至: {output_path}")

    # 5. 对比 (保持不变)
    if os.path.exists(original_md_path):
        with open(original_md_path, "r", encoding="utf-8") as f:
            original_text = f.readlines()

        diff = difflib.unified_diff(
            original_text,
            converted_md_text.splitlines(keepends=True),
            fromfile="Original.md",
            tofile="Docling.md"
        )
        print("\n--- 差异对比结果 ---")
        for line in diff:
            print(line.strip())


if __name__ == "__main__":
    process_and_compare_with_docling("docs/chroma/master.pdf", "docs/chroma/master.md")