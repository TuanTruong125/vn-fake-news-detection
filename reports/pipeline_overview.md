# Pipeline Overview

raw -> ingest -> validate -> normalize -> map_labels -> quality_filter -> deduplicate -> build_master -> split -> reporting

## Core Rules
- Label convention: 0=real, 1=fake.
- Raw files are immutable.
- Deduplicate before split.
- Leakage checks use hash_text intersection across train/val/test.
