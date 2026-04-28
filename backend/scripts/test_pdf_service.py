"""
Simple test for pdf_service.py
Run from the backend/ folder with:  python scripts/test_pdf_service.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.pdf_service import process_pdf

# Run the full pipeline on our test PDF
chunks = process_pdf("test.pdf", "test.pdf")

# Print results
print(f"Total chunks: {len(chunks)}")

if len(chunks) == 0:
    print("⚠️  No chunks produced! Check that data/uploads/test.pdf exists and has text.")
else:
    print(f"First chunk metadata: {chunks[0]['metadata']}")
    print()
    print("First chunk text (first 300 chars):")
    print(chunks[0]["text"][:300])