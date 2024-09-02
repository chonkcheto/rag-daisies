# flake8: noqa: E402
from dotenv import load_dotenv

load_dotenv()

import logging
import os
import argparse

from app.engine.loaders import get_documents
from app.settings import init_settings
from llama_index.core.indices import (
    VectorStoreIndex,
)


parser = argparse.ArgumentParser(description="Generate.py argument parsing")
parser.add_argument("--logging", type=int, help="Name of the user")
args = parser.parse_args()

log_level = logging.INFO
logging.basicConfig(level=log_level)
logger = logging.getLogger()
if args.logging is not None and args.logging < 10:
    match args.logging:
        case 1:
            logger.setLevel(logging.DEBUG)
        case 2:
            logger.setLevel(logging.INFO)
        case 3:
            logger.setLevel(logging.WARNING)
        case 4:
            logger.setLevel(logging.ERROR)
        case 5:
            logger.setLevel(logging.CRITICAL)
        case _:
            logger.setLevel(logging.INFO)
    logger.info(f"Logging level set to {logger.getEffectiveLevel()/10}")
else:
    logger.warning(f"Logging value incorrect {args.logging}")

print(f"{args}")



def generate_datasource():
    init_settings()
    logger.info("Creating new index")
    storage_dir = os.environ.get("STORAGE_DIR", "storage")
    # load the documents and create the index
    documents = get_documents()
    # Set private=false to mark the document as public (required for filtering)
    for doc in documents:
        doc.metadata["private"] = "false"
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True,
    )
    # store it for later
    index.storage_context.persist(storage_dir)
    logger.info(f"Finished creating new index. Stored in {storage_dir}")


if __name__ == "__main__":
    generate_datasource()
