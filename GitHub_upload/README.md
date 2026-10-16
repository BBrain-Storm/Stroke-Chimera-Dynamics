# Stroke Chimera Dynamics

Source code for the delayed Wilson-Cowan model, synchronization-state classification, and extraction of dominant propagation pathways used in the study *Recovery of stimulation-evoked synchronization in post-stroke brain networks*.

## Code map

| Component | Source |
| --- | --- |
| Stochastic delayed Wilson-Cowan simulation | `stimulation/model.py` |
| Global and functional-system synchronization | `stimulation/metrics.py` |
| Coherent, chimera-like, and incoherent classification | `stimulation/patton.py`, `stimulation/classify_patton.py` |
| Dominant propagation scores, driver selection, directed cycles, and reachability | `stimulation/propagation.py` |
| Repeated stimulation and saved outputs | `stimulation/run_dynamics.py` |
| Propagation summaries and validation | `stimulation/summarize_propagation.py`, `stimulation/validate_results.py`, `stimulation/validate_classification.py` |
| Union of propagation trees and edge recruitment counts | `analysis/union_propagation_network.py` |
| Configuration | `config/stimulation.json` |
| Data-free demonstration | `examples/synthetic_demo.py` |

The `stimulation/`, `analysis/`, configuration, and original propagation test files are copied from the authors' local V4 source without changing the algorithms. Documentation, the synthetic demonstration, and the synthetic integration test were added for this release. `SOURCE_MANIFEST.json` records file hashes for traceability.
