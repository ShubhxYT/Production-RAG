"""Unit tests for the structure-aware chunker."""

from ingestion.chunker import (
    chunk_document,
    count_tokens,
    split_text_by_sentences,
    _split_table_by_rows,
    _split_list_by_items,
)
from ingestion.models import (
    Chunk,
    ChunkingConfig,
    Document,
    Element,
    ElementType,
)


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


def test_count_tokens_empty():
    assert count_tokens("") == 0


def test_count_tokens_nonempty():
    n = count_tokens("hello world")
    assert n > 0


def test_count_tokens_deterministic():
    text = "The quick brown fox jumps over the lazy dog."
    assert count_tokens(text) == count_tokens(text)


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------


def test_split_text_by_sentences_empty():
    assert split_text_by_sentences("") == []


def test_split_text_by_sentences_basic():
    sentences = split_text_by_sentences("Hello world. How are you? Fine!")
    assert len(sentences) == 3


def test_split_text_by_sentences_double_newline():
    sentences = split_text_by_sentences("First.\n\nSecond.")
    assert len(sentences) == 2


# ---------------------------------------------------------------------------
# Table splitting
# ---------------------------------------------------------------------------


def test_split_table_small():
    table = "| A | B |\n| --- | --- |\n| 1 | 2 |"
    result = _split_table_by_rows(table, max_tokens=1000)
    assert len(result) == 1
    assert result[0] == table


def test_split_table_preserves_header():
    rows = ["| A | B |", "| --- | --- |"]
    for i in range(20):
        rows.append(f"| {i} | data{i} |")
    table = "\n".join(rows)
    # Use a very small max to force splitting
    result = _split_table_by_rows(table, max_tokens=30)
    assert len(result) > 1
    for sub in result:
        lines = sub.strip().split("\n")
        assert lines[0] == "| A | B |"
        assert lines[1] == "| --- | --- |"


# ---------------------------------------------------------------------------
# List splitting
# ---------------------------------------------------------------------------


def test_split_list_small():
    items = "- Item 1\n- Item 2\n- Item 3"
    result = _split_list_by_items(items, max_tokens=1000)
    assert len(result) == 1


def test_split_list_large():
    items = "\n".join([f"- {'Item ' * 20}{i}" for i in range(20)])
    result = _split_list_by_items(items, max_tokens=50)
    assert len(result) > 1


# ---------------------------------------------------------------------------
# chunk_document — basic cases
# ---------------------------------------------------------------------------


def _make_doc(**kwargs) -> Document:
    defaults = {"source_path": "test.md", "format": "md"}
    defaults.update(kwargs)
    return Document(**defaults)


def test_empty_document():
    doc = _make_doc(elements=[])
    chunks = chunk_document(doc)
    assert chunks == []


def test_single_short_paragraph():
    doc = _make_doc(
        elements=[
            Element(type=ElementType.PARAGRAPH, content="Hello world."),
        ]
    )
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].token_count > 0
    assert ElementType.PARAGRAPH in chunks[0].element_types


def test_very_short_document_single_chunk():
    doc = _make_doc(
        elements=[
            Element(type=ElementType.HEADING, content="Title", level=1),
            Element(type=ElementType.PARAGRAPH, content="Short text."),
        ]
    )
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert ElementType.HEADING in chunks[0].element_types
    assert ElementType.PARAGRAPH in chunks[0].element_types


# ---------------------------------------------------------------------------
# chunk_document — structural rules
# ---------------------------------------------------------------------------


def test_heading_never_orphaned():
    """A heading should never be the last (and only) content in a chunk,
    unless it is the very last element in the document."""
    doc = _make_doc(
        elements=[
            Element(
                type=ElementType.PARAGRAPH,
                content="First paragraph. " * 80,
            ),
            Element(type=ElementType.HEADING, content="Section Two", level=2),
            Element(
                type=ElementType.PARAGRAPH,
                content="Second paragraph content here.",
            ),
        ]
    )
    config = ChunkingConfig(target_max_tokens=100, overlap_tokens=10)
    chunks = chunk_document(doc, config)

    for chunk in chunks[:-1]:  # Exclude the very last chunk
        lines = chunk.text.strip().split("\n")
        last_line = lines[-1].strip()
        # If the chunk contains a heading, it should not be the only content
        if ElementType.HEADING in chunk.element_types:
            assert ElementType.PARAGRAPH in chunk.element_types or len(
                chunk.element_types
            ) > 1, f"Heading orphaned in chunk {chunk.position}"


def test_table_kept_whole():
    table_content = "| A | B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |"
    doc = _make_doc(
        elements=[
            Element(type=ElementType.PARAGRAPH, content="Before table."),
            Element(
                type=ElementType.TABLE,
                content=table_content,
                metadata={"rows": 3, "cols": 2},
            ),
            Element(type=ElementType.PARAGRAPH, content="After table."),
        ]
    )
    chunks = chunk_document(doc)
    # Find the chunk containing the table
    table_chunks = [c for c in chunks if ElementType.TABLE in c.element_types]
    assert len(table_chunks) >= 1
    assert table_content in table_chunks[0].text


def test_code_block_kept_whole():
    code = "def hello():\n    print('hello world')\n    return 42"
    doc = _make_doc(
        elements=[
            Element(
                type=ElementType.CODE_BLOCK,
                content=code,
                metadata={"language": "python"},
            ),
        ]
    )
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert code in chunks[0].text
    assert ElementType.CODE_BLOCK in chunks[0].element_types


def test_list_kept_whole_when_small():
    list_content = "- Item 1\n- Item 2\n- Item 3"
    doc = _make_doc(
        elements=[
            Element(
                type=ElementType.LIST,
                content=list_content,
                metadata={"list_type": "unordered"},
            ),
        ]
    )
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert list_content in chunks[0].text


# ---------------------------------------------------------------------------
# chunk_document — overlap
# ---------------------------------------------------------------------------


def test_overlap_present_in_subsequent_chunks():
    doc = _make_doc(
        elements=[
            Element(
                type=ElementType.PARAGRAPH,
                content="First paragraph with enough content. " * 40,
            ),
            Element(
                type=ElementType.PARAGRAPH,
                content="Second paragraph with different content. " * 40,
            ),
        ]
    )
    config = ChunkingConfig(
        target_max_tokens=100, overlap_tokens=20
    )
    chunks = chunk_document(doc, config)
    assert len(chunks) >= 2
    # Second chunk onwards should have non-empty overlap_before
    for chunk in chunks[1:]:
        assert chunk.overlap_before != "", (
            f"Chunk {chunk.position} missing overlap"
        )


# ---------------------------------------------------------------------------
# chunk_document — provenance
# ---------------------------------------------------------------------------


def test_section_path_tracking():
    doc = _make_doc(
        elements=[
            Element(type=ElementType.HEADING, content="Chapter 1", level=1),
            Element(type=ElementType.PARAGRAPH, content="Intro."),
            Element(type=ElementType.HEADING, content="Section 1.1", level=2),
            Element(type=ElementType.PARAGRAPH, content="Details."),
        ]
    )
    chunks = chunk_document(doc)
    # All content fits in one chunk with default config
    # The section_path should reflect the last heading seen
    last_chunk = chunks[-1]
    assert "Chapter 1" in last_chunk.section_path or "Section 1.1" in last_chunk.section_path


def test_chunk_positions_sequential():
    doc = _make_doc(
        elements=[
            Element(
                type=ElementType.PARAGRAPH,
                content="Text. " * 100,
            ),
            Element(
                type=ElementType.PARAGRAPH,
                content="More text. " * 100,
            ),
        ]
    )
    config = ChunkingConfig(target_max_tokens=80, overlap_tokens=10)
    chunks = chunk_document(doc, config)
    for i, chunk in enumerate(chunks):
        assert chunk.position == i, (
            f"Expected position {i}, got {chunk.position}"
        )


def test_document_id_propagated():
    doc = _make_doc(
        elements=[
            Element(type=ElementType.PARAGRAPH, content="Hello."),
        ]
    )
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].document_id == doc.id


# ---------------------------------------------------------------------------
# chunk_document — edge cases
# ---------------------------------------------------------------------------


def test_oversized_paragraph_split_by_sentences():
    long_para = ". ".join([f"Sentence number {i}" for i in range(100)]) + "."
    doc = _make_doc(
        elements=[
            Element(type=ElementType.PARAGRAPH, content=long_para),
        ]
    )
    config = ChunkingConfig(target_max_tokens=50, overlap_tokens=10)
    chunks = chunk_document(doc, config)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count > 0


def test_image_element_included():
    doc = _make_doc(
        elements=[
            Element(type=ElementType.PARAGRAPH, content="Some text."),
            Element(
                type=ElementType.IMAGE,
                content="images/photo.png",
                metadata={"alt_text": "A photo"},
            ),
        ]
    )
    chunks = chunk_document(doc)
    assert len(chunks) >= 1
    combined_text = " ".join(c.text for c in chunks)
    assert "photo.png" in combined_text
