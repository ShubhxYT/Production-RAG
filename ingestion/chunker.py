"""Structure-aware document chunker.

Splits a Document's Element list into Chunk objects that respect
structural boundaries, stay within a target token range, and track
provenance back to the source document and section.
"""

import logging
import re
import uuid

import tiktoken

from ingestion.models import (
    Chunk,
    ChunkingConfig,
    Document,
    Element,
    ElementType,
)

logger = logging.getLogger(__name__)

# Module-level singleton for the tiktoken encoder
_encoder: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    """Return a cached tiktoken encoder (cl100k_base)."""
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string.

    Uses the cl100k_base encoding (compatible with OpenAI models).

    Args:
        text: The text to tokenize.

    Returns:
        Number of tokens.
    """
    if not text:
        return 0
    return len(_get_encoder().encode(text))


def split_text_by_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation and newline boundaries.

    This is a fallback splitter for paragraphs that exceed the maximum
    token limit. It splits on sentence-ending punctuation followed by
    whitespace, or on double newlines.

    Args:
        text: The text to split.

    Returns:
        List of sentence strings (non-empty, stripped).
    """
    if not text.strip():
        return []
    # Split on sentence-ending punctuation followed by space, or double newline
    parts = re.split(r'(?<=[.!?])\s+|\n\n+', text)
    return [p.strip() for p in parts if p.strip()]


def _get_overlap_text(text: str, max_tokens: int) -> str:
    """Extract the last ~max_tokens tokens of text as overlap for the next chunk.

    Args:
        text: The full chunk text.
        max_tokens: Target number of overlap tokens.

    Returns:
        The trailing portion of text approximating max_tokens tokens.
    """
    if max_tokens <= 0 or not text:
        return ""
    encoder = _get_encoder()
    tokens = encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text
    overlap_tokens = tokens[-max_tokens:]
    return encoder.decode(overlap_tokens)


def _split_table_by_rows(
    table_content: str, max_tokens: int
) -> list[str]:
    """Split a markdown table into sub-tables that fit within max_tokens.

    Each sub-table preserves the header row and separator row from the
    original table. Splits at row boundaries.

    Args:
        table_content: The full markdown table string.
        max_tokens: Maximum tokens per sub-table.

    Returns:
        List of markdown table strings, each within the token limit.
    """
    lines = table_content.strip().split("\n")
    if len(lines) < 3:
        # Not a proper table (need header + separator + at least one data row)
        return [table_content]

    header_line = lines[0]
    separator_line = lines[1]
    data_lines = lines[2:]

    header_block = f"{header_line}\n{separator_line}"
    header_tokens = count_tokens(header_block)

    if not data_lines:
        return [table_content]

    sub_tables: list[str] = []
    current_rows: list[str] = []
    current_tokens = header_tokens

    for row in data_lines:
        row_tokens = count_tokens(row)
        # If adding this row would exceed max, flush current batch
        if current_rows and (current_tokens + row_tokens + 1) > max_tokens:
            sub_table = header_block + "\n" + "\n".join(current_rows)
            sub_tables.append(sub_table)
            current_rows = []
            current_tokens = header_tokens

        current_rows.append(row)
        current_tokens += row_tokens + 1  # +1 for newline

    # Flush remaining rows
    if current_rows:
        sub_table = header_block + "\n" + "\n".join(current_rows)
        sub_tables.append(sub_table)

    return sub_tables if sub_tables else [table_content]


def _split_list_by_items(
    list_content: str, max_tokens: int
) -> list[str]:
    """Split a markdown list into sub-lists that fit within max_tokens.

    Splits at list item boundaries (lines starting with '- ', '* ',
    or numbered items like '1. ').

    Args:
        list_content: The full markdown list string.
        max_tokens: Maximum tokens per sub-list.

    Returns:
        List of markdown list strings, each within the token limit.
    """
    lines = list_content.strip().split("\n")
    if not lines:
        return [list_content]

    # Group lines into list items (an item may span multiple lines
    # if continuation lines are indented)
    items: list[str] = []
    current_item_lines: list[str] = []

    for line in lines:
        if re.match(r'^(\s*[-*+]|\s*\d+\.)\s', line) and current_item_lines:
            items.append("\n".join(current_item_lines))
            current_item_lines = [line]
        else:
            current_item_lines.append(line)

    if current_item_lines:
        items.append("\n".join(current_item_lines))

    if not items:
        return [list_content]

    sub_lists: list[str] = []
    current_items: list[str] = []
    current_tokens = 0

    for item in items:
        item_tokens = count_tokens(item)
        if current_items and (current_tokens + item_tokens + 1) > max_tokens:
            sub_lists.append("\n".join(current_items))
            current_items = []
            current_tokens = 0

        current_items.append(item)
        current_tokens += item_tokens + 1

    if current_items:
        sub_lists.append("\n".join(current_items))

    return sub_lists if sub_lists else [list_content]


def chunk_document(
    document: Document, config: ChunkingConfig | None = None
) -> list[Chunk]:
    """Split a Document into Chunks that respect structural boundaries.

    The algorithm walks the document's Element list, accumulating
    elements into a buffer. When adding the next element would exceed
    the target_max_tokens limit, the buffer is flushed as a Chunk.

    Structural rules:
    - Tables are emitted as their own chunk (split by row groups if oversized)
    - Code blocks are kept whole (emitted oversized if necessary)
    - Headings are never the last element in a chunk
    - Lists are kept whole when possible (split at item boundaries if oversized)
    - Paragraphs exceeding max are split at sentence boundaries
    - Images are attached to the current buffer

    Args:
        document: The Document with populated elements.
        config: Chunking configuration. Uses defaults if None.

    Returns:
        Ordered list of Chunk objects.
    """
    if config is None:
        config = ChunkingConfig()

    if not document.elements:
        return []

    chunks: list[Chunk] = []
    position = 0
    overlap_text = ""

    # Buffer state
    buffer_texts: list[str] = []
    buffer_tokens: int = 0
    buffer_element_types: set[ElementType] = set()
    current_section_path: list[str] = []

    def _flush() -> str:
        """Flush the buffer into a Chunk and return overlap text."""
        nonlocal position, overlap_text, buffer_texts, buffer_tokens
        nonlocal buffer_element_types

        if not buffer_texts:
            return overlap_text

        combined = "\n\n".join(buffer_texts)
        if overlap_text:
            full_text = overlap_text + "\n\n" + combined
        else:
            full_text = combined

        token_count = count_tokens(full_text)

        chunk = Chunk(
            text=full_text,
            token_count=token_count,
            document_id=document.id,
            section_path=list(current_section_path),
            element_types=sorted(set(buffer_element_types), key=lambda x: x.value),
            position=position,
            overlap_before=overlap_text,
        )
        chunks.append(chunk)
        position += 1

        new_overlap = _get_overlap_text(
            combined, config.overlap_tokens
        )

        # Reset buffer
        buffer_texts = []
        buffer_tokens = 0
        buffer_element_types = set()

        return new_overlap

    def _add_to_buffer(text: str, element_type: ElementType) -> None:
        """Add text to the buffer, flushing first if needed."""
        nonlocal overlap_text, buffer_tokens
        text_tokens = count_tokens(text)

        if buffer_texts and (buffer_tokens + text_tokens) > config.target_max_tokens:
            overlap_text = _flush()

        buffer_texts.append(text)
        buffer_tokens += text_tokens
        buffer_element_types.add(element_type)

    i = 0
    while i < len(document.elements):
        element = document.elements[i]

        if element.type == ElementType.HEADING:
            # Update section path based on heading level
            level = element.level or 1
            # Trim section_path to parent level, then append
            current_section_path = current_section_path[: level - 1]
            current_section_path.append(element.content)

            heading_text = f"{'#' * level} {element.content}"
            heading_tokens = count_tokens(heading_text)

            # Check if there's a next element to attach
            has_next = (i + 1) < len(document.elements)

            if has_next:
                next_el = document.elements[i + 1]
                next_text = next_el.content
                next_tokens = count_tokens(next_text)

                # If heading + next element would exceed max and buffer
                # is non-empty, flush first
                if buffer_texts and (
                    buffer_tokens + heading_tokens + next_tokens
                ) > config.target_max_tokens:
                    overlap_text = _flush()

                # Add the heading to the buffer
                buffer_texts.append(heading_text)
                buffer_tokens += heading_tokens
                buffer_element_types.add(ElementType.HEADING)
                # The next iteration will handle the following element normally
            else:
                # Heading is the last element — just add it
                _add_to_buffer(heading_text, ElementType.HEADING)

            i += 1

        elif element.type == ElementType.TABLE:
            table_tokens = count_tokens(element.content)

            if table_tokens <= config.max_table_tokens:
                # Table fits as a single chunk — flush buffer, emit table
                if buffer_texts:
                    overlap_text = _flush()

                buffer_texts.append(element.content)
                buffer_tokens = table_tokens
                buffer_element_types.add(ElementType.TABLE)
                overlap_text = _flush()
            else:
                # Table is oversized — split by row groups
                if buffer_texts:
                    overlap_text = _flush()

                sub_tables = _split_table_by_rows(
                    element.content, config.max_table_tokens
                )
                for sub_table in sub_tables:
                    buffer_texts.append(sub_table)
                    buffer_tokens = count_tokens(sub_table)
                    buffer_element_types.add(ElementType.TABLE)
                    overlap_text = _flush()

            i += 1

        elif element.type == ElementType.CODE_BLOCK:
            code_tokens = count_tokens(element.content)

            if buffer_texts and (buffer_tokens + code_tokens) > config.target_max_tokens:
                overlap_text = _flush()

            # Add code block to buffer — never split it
            buffer_texts.append(element.content)
            buffer_tokens += code_tokens
            buffer_element_types.add(ElementType.CODE_BLOCK)

            # If the code block alone exceeds max, flush it as oversized
            if buffer_tokens > config.target_max_tokens:
                overlap_text = _flush()

            i += 1

        elif element.type == ElementType.LIST:
            list_tokens = count_tokens(element.content)

            if list_tokens <= config.target_max_tokens:
                _add_to_buffer(element.content, ElementType.LIST)
            else:
                # List is oversized — split by items
                if buffer_texts:
                    overlap_text = _flush()

                sub_lists = _split_list_by_items(
                    element.content, config.target_max_tokens
                )
                for sub_list in sub_lists:
                    buffer_texts.append(sub_list)
                    buffer_tokens = count_tokens(sub_list)
                    buffer_element_types.add(ElementType.LIST)
                    overlap_text = _flush()

            i += 1

        elif element.type == ElementType.PARAGRAPH:
            para_tokens = count_tokens(element.content)

            if para_tokens <= config.target_max_tokens:
                _add_to_buffer(element.content, ElementType.PARAGRAPH)
            else:
                # Paragraph exceeds max — split by sentences
                if buffer_texts:
                    overlap_text = _flush()

                sentences = split_text_by_sentences(element.content)
                for sentence in sentences:
                    _add_to_buffer(sentence, ElementType.PARAGRAPH)

            i += 1

        elif element.type == ElementType.IMAGE:
            image_text = element.content
            if element.metadata.get("alt_text"):
                image_text = f"![{element.metadata['alt_text']}]({element.content})"
            else:
                image_text = f"![]({element.content})"
            _add_to_buffer(image_text, ElementType.IMAGE)
            i += 1

        else:
            # Unknown element type — treat as paragraph
            _add_to_buffer(element.content, ElementType.PARAGRAPH)
            i += 1

    # Flush remaining buffer
    if buffer_texts:
        _flush()

    logger.info(
        "Chunked document %s into %d chunks", document.id, len(chunks)
    )
    return chunks
