"""Allow running the database module with python -m database."""

import argparse
import logging
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="database",
        description="Database management commands for FullRag.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # seed subcommand
    seed_parser = subparsers.add_parser(
        "seed", help="Seed the database from staged JSON documents."
    )
    seed_parser.add_argument(
        "--staging-dir",
        "-s",
        default="staging",
        help="Path to staging directory with Document JSON files (default: staging).",
    )
    seed_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )

    # clear subcommand
    clear_parser = subparsers.add_parser(
        "clear", help="Clear all tables from the database."
    )
    clear_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm the operation without prompting.",
    )
    clear_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.command == "seed":
        from database.seed import seed_from_staging

        stats = seed_from_staging(staging_dir=args.staging_dir)

        print(f"\n{'=' * 50}")
        print("Database Seed Summary")
        print(f"{'=' * 50}")
        print(f"Documents inserted: {stats['documents']}")
        print(f"Chunks inserted:    {stats['chunks']}")
        print(f"Embeddings inserted:{stats['embeddings']}")
        print(f"Skipped (existing): {stats['skipped']}")
        print(f"Failed:             {stats.get('failed', 0)}")
        print(f"Elapsed time:       {stats.get('elapsed_seconds', 0):.2f}s")

    elif args.command == "clear":
        from database.models import Base
        from database.connection import get_engine

        if not args.confirm:
            response = input(
                "\nWARNING: This will delete all tables from the database!\n"
                "Are you sure you want to proceed? (yes/no): "
            )
            if response.lower() != "yes":
                print("Operation cancelled.")
                sys.exit(0)

        engine = get_engine()
        Base.metadata.drop_all(engine)
        print("\nDatabase cleared successfully. All tables dropped.")


if __name__ == "__main__":
    main()
