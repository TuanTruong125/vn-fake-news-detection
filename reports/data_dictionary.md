# Data Dictionary

| Column | Type | Description | Example |
|---|---|---|---|
| sample_id | string | Deterministic sample id, equal to hash_text. | 61b5a9a9e6f1818fa40fcd96be80f9df |
| text_raw | string | Raw text kept for audit and trace. | Phát biểu tại Ủy Ban Thường Vụ Quốc Hội hôm ... |
| text_clean | string | Normalized text used for modeling. | Phát biểu tại Ủy Ban Thường Vụ Quốc Hội hôm ... |
| hash_text | string | MD5 hash of text_clean for dedup/leakage checks. | 61b5a9a9e6f1818fa40fcd96be80f9df |
| label_binary | int | Model label (0=real, 1=fake). | 1 |
| label_name | string | Human-readable label (real/fake). | fake |
| source_file | string | Original source file name. | fix_test_data.csv |
| source_domain | string | Content source domain/platform. | unknown |
| content_type | string | Content type (news/social). | news |
| published_at | string | Published timestamp if available. |  |
| label_confidence | float | Source-level label confidence. | 0.9 |
| text_length | int | Length of text_clean. | 4947 |
| split | string | Data split assignment (train/val/test). | train |
