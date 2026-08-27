from collections.abc import Sequence
from textwrap import wrap

PAGE_WIDTH = 88


def build_sample_pdf(pages: Sequence[Sequence[str]]) -> bytes:
    if not pages or any(not page for page in pages):
        raise ValueError("sample PDF requires at least one non-empty page")

    page_count = len(pages)
    font_object_number = 3 + (page_count * 2)
    page_object_numbers = [3 + (index * 2) for index in range(page_count)]
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        (
            f"<< /Type /Pages /Kids [{' '.join(f'{number} 0 R' for number in page_object_numbers)}] "
            f"/Count {page_count} >>"
        ).encode("ascii"),
    ]

    for index, page in enumerate(pages):
        page_object_number = page_object_numbers[index]
        content_object_number = page_object_number + 1
        stream = _page_stream(page)
        objects.extend(
            [
                (
                    f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    f"/Resources << /Font << /F1 {font_object_number} 0 R >> >> "
                    f"/Contents {content_object_number} 0 R >>"
                ).encode("ascii"),
                b"<< /Length "
                + str(len(stream)).encode("ascii")
                + b" >>\nstream\n"
                + stream
                + b"\nendstream",
            ]
        )

    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    return _serialize_pdf(objects)


def _page_stream(paragraphs: Sequence[str]) -> bytes:
    lines = [line for paragraph in paragraphs for line in wrap(paragraph, PAGE_WIDTH)]
    commands = [b"BT", b"/F1 11 Tf", b"72 740 Td", b"15 TL"]
    for line in lines:
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        commands.append(f"({escaped}) Tj".encode("latin-1"))
        commands.append(b"T*")
    commands.append(b"ET")
    return b"\n".join(commands)


def _serialize_pdf(objects: Sequence[bytes]) -> bytes:
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode("ascii"))
        output.extend(body)
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(output)
