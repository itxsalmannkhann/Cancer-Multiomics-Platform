# Security Policy

## Research preview — not for clinical use

This project is a research/hackathon prototype trained on de-identified TCGA
data. It is **not** a certified medical device and must not be used to inform
real clinical decisions.

## Supported versions

| Version | Supported |
|---|---|
| 1.1.x   | ✅ |
| 1.0.x   | ⚠️ security fixes only |
| < 1.0   | ❌ |

## Reporting a vulnerability

If you discover a security issue (for example, exposure of patient-identifiable
data, an unsafe deserialization path, or a dependency vulnerability):

1. **Do not** open a public issue.
2. Email the maintainers with a description, reproduction steps, and impact.
3. You can expect an acknowledgement within a few business days.

## Handling data

- Never commit real patient-identifiable information to this repository.
- Model artifacts (`*.joblib`) are loaded with `joblib`; only load model files
  from trusted sources, since pickle-based deserialization can execute code.
