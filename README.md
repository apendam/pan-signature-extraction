# PAN Signature Extraction

Locate and crop the handwritten signature from a PAN card image using
Google Cloud Document AI (for OCR / text anchors) and OpenCV (to refine
the crop down to the actual ink).

## Important: what Document AI's "signature" feature actually gives you

It's tempting to assume Document AI has a "detect signature" call that
returns a bounding box. It doesn't, currently. Document AI does have a
signature entity type (in Custom Document Extractor foundation models
`pretrained-foundation-model-v1.4-2025-02-05` / `v1.5-2025-05-05`, and as
a `visualElement` type in the general Document proto), but it's explicitly
a **derived, presence-only field**: the response is a `"Detected"` flag
with a confidence score, and Google's own docs note that `textAnchor` and
`pageAnchor` are *not* populated for it, precisely because no bounding
box is produced.

So this pipeline takes a different, well-supported route to get real
coordinates:

1. Run a plain **OCR / Form Parser** processor over the image. That
   returns ordinary text tokens, each with a real bounding box
   (`normalizedVertices`).
2. If the page has a caption like `"Signature"` printed on it, use that
   token's position as an anchor and build a region of interest (ROI)
   relative to it (`pan_signature/signature_locator.py`).
3. If no such caption is found (common — many PAN card fronts don't
   print the word "Signature" at all), fall back to a fixed ROI expressed
   as a fraction of the page, which you calibrate against your own
   samples.
4. Feed that coarse ROI into OpenCV (`pan_signature/signature_extractor.py`):
   threshold, find connected ink components, and tighten the crop to
   their bounding box so you get the signature itself rather than a
   loose rectangle.

If you outgrow the anchor/fixed-ROI heuristic (e.g. PAN layouts vary too
much across your image sources), the next step up is training a **Custom
Document Extractor** with a manually box-labeled `signature` field (drawn
in the labeling UI on ~20-50 samples) — unlike the derived feature, a
manually labeled visual field does get you a trained bounding-box
detector. That's a bigger lift (labeling + training + a paid custom
processor), so this repo starts with the heuristic approach and treats
that as a future upgrade, not a prerequisite.

## Architecture

```
image.jpg -> DocumentAIClient.process_image_bytes()  -> Document (OCR tokens + boxes)
          -> locate_signature_region()                -> coarse ROI (BBox, pixel coords)
          -> refine_and_crop()  [OpenCV]               -> tight signature crop + refined BBox
```

- `pan_signature/docai_client.py` - calls the Document AI `process_document` RPC.
- `pan_signature/signature_locator.py` - OCR tokens -> coarse ROI.
- `pan_signature/signature_extractor.py` - OpenCV thresholding/contours -> tight crop.
- `pan_signature/pipeline.py` - wires the three together; optional debug overlay.
- `scripts/extract_signature.py` - CLI entry point.

## Google Cloud setup

1. **Enable the API** in your GCP project:
   `gcloud services enable documentai.googleapis.com`
2. **Create a processor** (console: Document AI -> create processor).
   Use an **OCR / Enterprise Document OCR** or **Form Parser** processor
   type — you specifically do *not* need a Custom Extractor for this
   heuristic approach. Note the processor ID and its location (`us`/`eu`).
3. **Grant access**: your user or service account needs the
   `roles/documentai.apiUser` role on the project.
4. **Authenticate locally**:
   `gcloud auth application-default login`
   (or set `GOOGLE_APPLICATION_CREDENTIALS` to a service-account key —
   avoid committing that key anywhere near this repo).
5. Copy `.env.example` to `.env` (or export the same variables) and fill
   in `GOOGLE_CLOUD_PROJECT`, `DOCAI_LOCATION`, `DOCAI_PROCESSOR_ID`.

## Install and run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt   # or requirements.txt without pytest

export $(grep -v '^#' .env | xargs)   # or use direnv/python-dotenv

python scripts/extract_signature.py \
  --image path/to/pan_sample.jpg \
  --out signature.png \
  --debug debug_overlay.jpg
```

`debug_overlay.jpg` draws the coarse ROI in orange and the refined crop
in green on top of the original image — use it to calibrate
`LocatorConfig.default_roi_fraction` in `pan_signature/config.py` against
your own PAN samples before relying on the fallback path in production.
(The debug overlay always shows the *unmodified* source image; the
whitening/rotation below only apply to `--out`, the final saved crop.)

### Final output: white background, landscape orientation

`signature.png` isn't just the raw crop -- `pipeline.py` runs two more
steps on it by default (`ExtractorConfig` in `pan_signature/config.py`):

- **`whiten_background`**: replaces everything except the detected ink
  with solid white, so you get a clean signature image instead of the
  card's textured background.
- **`orient_horizontal`**: rotates a taller-than-wide crop 90 degrees so
  it comes out landscape (a signature reads naturally left-to-right,
  wider than tall). No-op if the crop is already landscape or square.

Both were verified against the same real PAN card: the whitened output
contains only the signature ink, and the rotation direction
(`ExtractorConfig.rotate_direction`, default `"counterclockwise"`) was
picked by actually checking the signature reads left-to-right afterward,
not backwards. If your source images are scanned rotated the other way,
the signature will come out reading backwards and you should set
`rotate_direction="clockwise"` instead. Set `whiten_background=False` /
`orient_horizontal=False` on `ExtractorConfig` to disable either step.

## Alternative provider: Mistral OCR

By default this pipeline uses Google Document AI as described above. As
an alternative, pass `--provider mistral` to use Mistral's OCR API
instead.

Both providers end up doing basically the same kind of anchor-text
heuristic, just against different response shapes:

- **Document AI** (default): a plain OCR/Form Parser processor returns
  text tokens with bounding boxes; `signature_locator.py` anchors the ROI
  on a `"Signature"` caption token (or falls back to a fixed page
  fraction), because Document AI's own signature field is presence-only
  (see above).
- **Mistral**: its OCR block-classification feature (`include_blocks`,
  on by default) is documented as being able to tag a content block's
  `type` as `"signature"` directly. Tested against a real Indian PAN
  card, that never actually happened -- with both `mistral-ocr-latest`
  and the explicit `mistral-ocr-4-1` model, the signature's ink and the
  printed `"हस्ताक्षर / Signature"` caption next to it came back merged
  into one ordinary `"text"` block. So `mistral_signature_locator.py`
  keeps the type-based check (in case a future model version or a
  different layout does emit it) but falls back to the same
  content-keyword anchor `signature_locator.py` uses, then to
  `LocatorConfig.default_roi_fraction` -- same tiered degrade-gracefully
  shape on both providers.

Since the caption ends up inside the same box as the signature on both
providers, `refine_and_crop()` also excludes blue ink before
thresholding: real PAN cards print field captions in blue and signed/
entered values in black, so filtering by color cleanly strips the
caption back out of the crop. This is on by default
(`ExtractorConfig.exclude_blue_ink`) and is a no-op on grayscale scans.

### Setup

1. Get an API key from [console.mistral.ai](https://console.mistral.ai).
2. Add `MISTRAL_API_KEY` (and optionally `MISTRAL_OCR_MODEL`, which
   defaults to `mistral-ocr-latest`) to your `.env` -- see `.env.example`.
3. Run with the Mistral backend:

   ```bash
   python scripts/extract_signature.py \
     --image path/to/pan_sample.jpg \
     --out signature.png \
     --provider mistral
   ```

### Caveats

This path (block-type check -> content-keyword anchor -> default ROI,
feeding into `refine_and_crop()`'s blue-ink exclusion) has been run
end-to-end against a real Indian PAN card and produced a clean signature
crop with the caption excluded. What's still just one data point: only
one card layout, one rotation, one ink-color convention have actually
been tested. The bbox field names (`top_left_x/y`, `bottom_right_x/y`)
and the pixel-vs-normalized coordinate handling in
`mistral_signature_locator.py` are confirmed against that real response;
if your cards use a different layout or color convention, recalibrate
`LocatorConfig`/`ExtractorConfig` (`--debug` overlay is there for this)
before trusting it on your full corpus.

It's also worth noting that Mistral's usage policy disclaims use of its
models for "financial decisions." A PAN card is a financial-identity
document, so that's relevant context here -- but this pipeline only
*locates and crops* a signature image (an extraction/automation step); it
does not make any eligibility, verification, or other financial
decision. If your use case feeds this into a decision downstream, review
Mistral's usage policy yourself before relying on this provider for it.

## Tests

The unit tests are fully offline (no GCP credentials needed) — they
exercise the locator against synthetic OCR tokens and the extractor
against synthetic ink strokes:

```bash
pip install -r requirements-dev.txt
pytest
```

## Handling PAN images and signatures responsibly

PAN numbers and signatures are sensitive personal/financial identifiers.
A few things worth doing before this goes anywhere near production:

- Don't commit real PAN images, extracted signatures, or debug overlays
  to this repo (`.gitignore` already excludes `sample_images/` and
  `output/` — keep using those paths, or add your own).
- Scope the service account narrowly (`documentai.apiUser` only) and
  restrict who can read the Cloud Storage/logging locations Document AI
  writes to, if any.
- Delete or encrypt intermediate images per your organization's data
  retention policy; don't log raw image bytes.
- If your GCP org requires VPC Service Controls / CMEK for PII
  processing, apply the same policy to the Document AI processor's
  project.
