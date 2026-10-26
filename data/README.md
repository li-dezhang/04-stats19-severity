# Data provenance and acquisition

Only DfT's public, non-sensitive **Collisions 2024**, **Collisions 2025**, and official **Open Dataset Data Guide** are used. No vehicle, casualty, sensitive-factor, or full historical download.

Official source: https://www.gov.uk/government/statistical-data-sets/road-safety-open-data  
Licence: https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/

Contains public sector information licensed under the Open Government Licence v3.0. Source: Department for Transport, STATS19.

Exact download URLs, acquisition timestamp, byte sizes and SHA-256 are in [manifest.json](manifest.json). `scripts/acquire.py` retrieves these exact objects, checks sizes and recorded hashes, and refuses changed files. It leaves original files read-only. A publisher revision requires investigation; do not silently replace the manifest to make verification pass.

| File | Bytes | Eligible collisions |
|---|---:|---:|
| collisions_2024.csv | 19,208,488 | 100,927 |
| collisions_2025.csv | 19,349,153 | 101,525 |
| data_guide_2025.xlsx | 85,112 | not applicable |

The spreadsheet's `2024_code_list` defines the current common fields. The `2011_to_2024_conversion` sheet contains an apparent weather/road-surface labelling inconsistency; no legacy conversion is applied. Both published CSVs already provide current `junction_detail`; historic fields are excluded. Known current labels are retained, but reporting comparability is not assumed perfect.

Recorded severity 1=Fatal, 2=Serious, 3=Slight; target 1 for codes 1/2. No target exclusions, duplicate IDs or invalid selected codes were found. Missing and Unknown are separate and never inferred from raw null counts alone. The codebook extract, all 44 field names/dtypes, missingness and samples are in `results/data/`.

Local `raw/`, `processed/` and `interim/` stay out of Git. Processed files contain deterministic cleaned rows and split assignments. Published derived predictions contain binary labels and model scores, in sorted-ID order, without collision IDs, coordinates or other raw fields; retain the OGL attribution with them. They support rescoring without re-downloading data. No source files are bundled in the public package.

Run from the P4 root:
```sh
.venv/bin/python scripts/acquire.py
.venv/bin/python scripts/prepare.py
```
An internet connection and the unchanged official files are required for a fresh download. Current local checks reuse verified raw files. No claim is made that future publisher revisions can reproduce these exact results.
