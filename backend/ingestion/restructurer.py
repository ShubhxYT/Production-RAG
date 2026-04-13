"""Restructurer: parse markdown into typed Element objects."""

import logging

from markdown_it import MarkdownIt

from ingestion.models import Element, ElementType

logger = logging.getLogger(__name__)


def restructure(markdown_text: str) -> list[Element]:
    """Parse markdown text into a list of typed Element objects.

    Uses markdown-it-py to tokenize the markdown, then walks the flat
    token stream to classify each block-level element.

    Args:
        markdown_text: Raw markdown content from any loader.

    Returns:
        Ordered list of Element objects.
    """
    if not markdown_text.strip():
        return []

    md = MarkdownIt("commonmark").enable("table")
    tokens = md.parse(markdown_text)
    lines = markdown_text.split("\n")
    elements: list[Element] = []

    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token.type == "heading_open":
            level = int(token.tag[1])  # h1 -> 1, h2 -> 2, ...
            inline_token = tokens[i + 1] if i + 1 < len(tokens) else None
            content = inline_token.content if inline_token else ""
            elements.append(
                Element(
                    type=ElementType.HEADING, content=content, level=level
                )
            )
            i += 3  # heading_open, inline, heading_close

        elif token.type == "paragraph_open":
            inline_token = tokens[i + 1] if i + 1 < len(tokens) else None
            if inline_token and _is_standalone_image(inline_token):
                img = _get_image_token(inline_token)
                if img:
                    src = img.attrs.get("src", "") if img.attrs else ""
                    alt_text = _get_image_alt(img)
                    elements.append(
                        Element(
                            type=ElementType.IMAGE,
                            content=src,
                            metadata={"alt_text": alt_text},
                        )
                    )
            elif inline_token:
                elements.append(
                    Element(
                        type=ElementType.PARAGRAPH,
                        content=inline_token.content,
                    )
                )
            i += 3  # paragraph_open, inline, paragraph_close

        elif token.type == "fence":
            language = token.info.strip() if token.info else ""
            meta = {"language": language} if language else {}
            elements.append(
                Element(
                    type=ElementType.CODE_BLOCK,
                    content=token.content.rstrip("\n"),
                    metadata=meta,
                )
            )
            i += 1

        elif token.type == "table_open":
            start_line, end_line = (
                token.map if token.map else (0, 0)
            )
            table_md = "\n".join(lines[start_line:end_line])
            rows, cols = _count_table_dims(tokens, i)
            elements.append(
                Element(
                    type=ElementType.TABLE,
                    content=table_md,
                    metadata={"rows": rows, "cols": cols},
                )
            )
            i = _skip_to_close(tokens, i, "table_open", "table_close")

        elif token.type in ("bullet_list_open", "ordered_list_open"):
            start_line, end_line = (
                token.map if token.map else (0, 0)
            )
            list_md = "\n".join(lines[start_line:end_line])
            list_type = (
                "ordered" if "ordered" in token.type else "unordered"
            )
            elements.append(
                Element(
                    type=ElementType.LIST,
                    content=list_md,
                    metadata={"list_type": list_type},
                )
            )
            close_type = token.type.replace("_open", "_close")
            i = _skip_to_close(tokens, i, token.type, close_type)

        else:
            # Skip tokens we don't explicitly handle (hr, blockquote
            # wrappers, html_block, etc.)
            i += 1

    logger.info("Restructured markdown into %d elements", len(elements))
    return elements


def _is_standalone_image(inline_token) -> bool:
    """Check if an inline token contains only a single image."""
    if not inline_token.children:
        return False
    meaningful = [
        c
        for c in inline_token.children
        if not (c.type == "text" and not c.content.strip())
        and c.type != "softbreak"
    ]
    return len(meaningful) == 1 and meaningful[0].type == "image"


def _get_image_token(inline_token):
    """Return the first image token from inline children, or None."""
    if not inline_token.children:
        return None
    for child in inline_token.children:
        if child.type == "image":
            return child
    return None


def _get_image_alt(image_token) -> str:
    """Extract alt text from an image token."""
    if image_token.children:
        return "".join(
            c.content for c in image_token.children if c.type == "text"
        )
    if image_token.attrs:
        return image_token.attrs.get("alt", "")
    return ""


def _count_table_dims(tokens, table_open_idx: int) -> tuple[int, int]:
    """Count rows and columns in a table token sequence."""
    rows = 0
    cols = 0
    current_cols = 0
    i = table_open_idx + 1

    while i < len(tokens) and tokens[i].type != "table_close":
        if tokens[i].type == "tr_open":
            rows += 1
            current_cols = 0
        elif tokens[i].type in ("th_open", "td_open"):
            current_cols += 1
            cols = max(cols, current_cols)
        i += 1

    return rows, cols


def _skip_to_close(
    tokens, start_idx: int, open_type: str, close_type: str
) -> int:
    """Skip tokens until the matching close token, handling nesting."""
    nesting = 1
    i = start_idx + 1
    while i < len(tokens) and nesting > 0:
        if tokens[i].type == open_type:
            nesting += 1
        elif tokens[i].type == close_type:
            nesting -= 1
        i += 1
    return i
