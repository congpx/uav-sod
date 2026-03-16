# uav-sod

# UAV-YOLO VisDrone Ubuntu pipeline
This bundle reproduces the Ubuntu workflow for:

- Phase 1: TAL ablation at 640

- Phase 2: P2-head ablation at 640/800

- Phase 3: final confirmation with 2 runs at 800

* Assumptions:

Ubuntu with NVIDIA driver already working (nvidia-smi OK)
Dataset already converted to YOLO format with a dataset YAML
You want to modify Ultralytics v8.4.11 in editable mode
Main entry points:

```
scripts/setup_project.sh
scripts/run_phase1_tal.sh
scripts/run_phase3_p2.sh
scripts/run_phase4_confirm.sh
scripts/run_all_from_phase1_to4.sh
```
