# Histopathology Image Pipeline (inactive — no `.svs` data was provided)

This NeoHacks brief specifies a 4th data modality: H&E-stained `.svs` whole-slide
images. **No `.svs` files were included** in the uploaded dataset (only clinical,
exposure, follow-up, pathology-detail, mutations, transcriptomics, and methylation
tables), so no image model was trained, and the Streamlit "Upload → Histopathology"
tab is disabled rather than faking a prediction.

## To activate this pipeline once slide data is available

1. Add `.svs` files to `data/raw/svs/`, named or mapped to `patient_id` via a
   manifest CSV (`slide_id,patient_id,label`).
2. Install the extra dependencies (already listed, commented out, in
   `requirements.txt`): `openslide-python`, `opencv-python-headless`, `pillow`,
   `torch`, `torchvision`.
3. Implement, in this folder:
   - `patch_extraction.py` — read each slide with OpenSlide, tile into
     256x256 patches at 20x, discard background/low-tissue patches (Otsu
     threshold on saturation), stain-normalize (Macenko or Reinhard), and
     augment (flips/rotations/color jitter) for training.
   - `train_image_model.py` — fine-tune EfficientNet-B0 / ConvNeXt-Tiny /
     ResNet50 (ImageNet-pretrained) as patch-level classifiers, aggregate
     patch predictions to slide-level via mean-pooling or attention-based
     MIL (multiple-instance learning — recommended given weak slide-level
     labels), and compare the three backbones the same way
     `src/training/classification.py` compares tabular models.
   - `gradcam.py` — Grad-CAM on the trained CNN backbone for slide-level
     explainability, saved next to the SHAP outputs in `reports/figures/`.
4. Wire the trained model into `main.py` as a 5th task, and flip
   `image_pipeline.enabled: true` in `config.yaml`.
5. Enable the "Histopathology" tab in `streamlit_app/pages/1_Upload_Data.py`
   (currently `disabled=True` on the file uploader).

This is intentionally left as a stub rather than executable-but-untested code,
since training an image model with zero image data would only produce
placeholder metrics that look real but mean nothing.
