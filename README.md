## Production Considerations
- False positives are costly in industrial environments; decision persistence is used
- Sensor drift can degrade model performance and must be monitored
- Models should be retrained periodically using validated data

## What Breaks in Production
- Sudden sensor recalibration
- Long-term distribution drift
- Hardware faults producing invalid readings

## Engineering Trade-offs
- Isolation Forest chosen for robustness and low supervision
- Deep learning avoided to preserve interpretability
- Baseline model retained for fallback and diagnostics
