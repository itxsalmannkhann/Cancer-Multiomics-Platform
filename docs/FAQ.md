# Frequently Asked Questions

### Why only 3 modalities instead of the 4 in the brief?
No `.svs` histopathology slides were included in the uploaded dataset — only
tabular clinical, genome, transcriptome, and epigenome files. The imaging
pipeline is documented and stubbed in `src/image_pipeline_stub/` so it can be
plugged in once slide data exists.

### Why is Task 2 (stage) so much weaker than the other tasks?
Only ~70 patients spread across two imbalanced classes, combined with very high
omics dimensionality. The modest score (κ ≈ 0.25) is reported honestly rather
than hidden.

### Why binarize stage and smoking instead of full multiclass?
Per-stage counts are too small for stable 4-way classification at this cohort
size. Binarizing (Early I/II vs Late III/IV) yields usable class sizes.

### What is the data-leakage guard about?
Predicting smoking status from other exposure fields (e.g. `pack_years_smoked`)
would trivially solve the task without biology. `main.py` excludes exposure
fields from Task 3 and AJCC T/N/M fields from Task 2.

### Does the app retrain on every load?
No. The dashboard reads pre-computed artifacts in `models/` and `reports/`.
Generate them once with `python src/build_dataset.py && python main.py`.

### Can I use this on a different cohort?
Yes — drop files matching the same schema into `data/raw/` and re-run the two
pipeline steps. Nothing is hardcoded to the current 82-patient cohort.

### Is this safe for clinical use?
No. This is a research/hackathon prototype, not a certified medical device.
