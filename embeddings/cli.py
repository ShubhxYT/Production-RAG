"""CLI for batch embedding of staged document chunks."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

from embeddings.cache import CachedEmbeddingService
from embeddings.models import EmbeddingConfig
from embeddings.service import EmbeddingService
from ingestion.staging import load_staged_document


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="embeddings",
        description="Generate embeddings for staged document chunks using local GPU.",
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to staging directory with Document JSON files.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=".embedding_output",
        help="Output directory for embedding results (default: .embedding_output).",
    )
    parser.add_argument(
        "--cache-dir",
        default=".embedding_cache",
        help="Cache directory for dev caching (default: .embedding_cache).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable embedding cache.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of texts per batch (default: 100).",
    )
    parser.add_argument(
        "--model",
        default="BAAI/bge-base-en-v1.5",
        help="HuggingFace model name (default: BAAI/bge-base-en-v1.5).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    input_path = Path(args.input)
    if not input_path.is_dir():
        print(
            f"Error: staging directory does not exist: {args.input}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load staged documents
    json_files = sorted(input_path.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {args.input}", file=sys.stderr)
        sys.exit(1)

    documents = []
    for jf in json_files:
        try:
            doc = load_staged_document(jf)
            documents.append(doc)
        except Exception:
            logging.getLogger(__name__).exception(
                "Failed to load: %s", jf.name
            )

    # Collect all chunks
    chunk_ids: list[str] = []
    chunk_texts: list[str] = []
    for doc in documents:
        for chunk in doc.chunks:
            chunk_ids.append(chunk.id)
            chunk_texts.append(chunk.text)

    if not chunk_texts:
        print("No chunks found in staged documents.", file=sys.stderr)
        sys.exit(1)

    # Set up embedding service
    config = EmbeddingConfig(
        model_name=args.model,
        batch_size=args.batch_size,
    )
    service = EmbeddingService(config=config)

    if args.no_cache:
        embed_service = service
    else:
        cache_dir = Path(args.cache_dir)
        embed_service = CachedEmbeddingService(
            service=service, cache_dir=cache_dir
        )

    # Embed all chunks
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_time = time.perf_counter()
    result = embed_service.embed(chunk_texts)
    elapsed = time.perf_counter() - start_time

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_data = {
        "model": result.model,
        "dimensions": result.dimensions,
        "device": device,
        "embeddings": [
            {"chunk_id": cid, "vector": vec}
            for cid, vec in zip(chunk_ids, result.vectors)
        ],
    }

    output_file = output_dir / "embeddings.json"
    output_file.write_text(
        json.dumps(output_data, indent=2), encoding="utf-8"
    )

    # Print summary - local model, no API cost
    print(f"\n{'=' * 50}")
    print("Embedding Summary")
    print(f"{'=' * 50}")
    print(f"Documents loaded:  {len(documents)}")
    print(f"Chunks embedded:   {len(chunk_ids)}")
    print(f"Model:             {result.model}")
    print(f"Dimensions:        {result.dimensions}")
    print(f"Device:            {device}")
    print(f"API cost:          $0.00 (local model)")
    print(f"Elapsed time:      {elapsed:.2f}s")
    print(f"Output file:       {output_file}")


if __name__ == "__main__":
    main()
