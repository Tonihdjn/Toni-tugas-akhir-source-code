# YOLO Model Optimization for Armed Person Detection

This repository contains the source code for optimizing YOLO (You Only Look Once) models for armed person detection. The project compares baseline and optimized versions of YOLOv8, YOLOv9, YOLOv10, and YOLOv11.

## Project Structure

```
├── EDA.ipynb                           # Exploratory Data Analysis
├── validate-yolo-version.ipynb         # Model validation across versions
├── yolov8s.ipynb                       # YOLOv8 baseline implementation
├── yolov8s-optimize.ipynb              # YOLOv8 optimized version
├── yolov9s.ipynb                       # YOLOv9 baseline implementation
├── yolov9s-optimize.ipynb              # YOLOv9 optimized version
├── yolov10s.ipynb                      # YOLOv10 baseline implementation
├── yolov10s-optimize.ipynb             # YOLOv10 optimized version
├── yolov11s.ipynb                      # YOLOv11 baseline implementation
├── yolov11s-optimize.ipynb             # YOLOv11 optimized version
├── the most optimize yolov8.ipynb      # Best YOLOv8 configuration
├── the most optimize yolov9.ipynb      # Best YOLOv9 configuration
├── the most optimize yolov10.ipynb     # Best YOLOv10 configuration
├── the most optimize yolov11.ipynb     # Best YOLOv11 configuration
└── hasil/                              # Results directory
    ├── Baseline/                       # Baseline model results
    └── Optimized/                      # Optimized model results
```

## Dataset

The project uses the "Armed Person Recognition" dataset from Roboflow:
- **Workspace**: s-gbust
- **Project**: armed-person-recognition-gohff
- **Version**: 5
- **Format**: YOLOv8

The dataset is automatically downloaded using the Roboflow API in the notebooks.

## Requirements

### Python Libraries
```
ultralytics
roboflow
opencv-python
matplotlib
numpy
torch
torchvision
```

### Installation
```bash
pip install ultralytics roboflow opencv-python matplotlib numpy torch torchvision
```

## Usage

### 1. Exploratory Data Analysis
Start with `EDA.ipynb` to understand the dataset:
- Dataset statistics
- Class distribution
- Sample visualizations
- Data quality checks

### 2. Baseline Models
Run the baseline notebooks to establish performance benchmarks:
- `yolov8s.ipynb` - YOLOv8 baseline
- `yolov9s.ipynb` - YOLOv9 baseline
- `yolov10s.ipynb` - YOLOv10 baseline
- `yolov11s.ipynb` - YOLOv11 baseline

### 3. Optimized Models
Run the optimized versions to see improvements:
- `yolov8s-optimize.ipynb` - YOLOv8 optimized
- `yolov9s-optimize.ipynb` - YOLOv9 optimized
- `yolov10s-optimize.ipynb` - YOLOv10 optimized
- `yolov11s-optimize.ipynb` - YOLOv11 optimized

### 4. Best Configurations
Check the "most optimize" notebooks for the best performing configurations:
- `the most optimize yolov8.ipynb`
- `the most optimize yolov9.ipynb`
- `the most optimize yolov10.ipynb`
- `the most optimize yolov11.ipynb`

### 5. Model Validation
Use `validate-yolo-version.ipynb` to compare all models across different versions.

## Key Features

### Optimization Techniques
- Hyperparameter tuning
- Data augmentation strategies
- Training configuration optimization
- Model architecture adjustments

### Evaluation Metrics
- Precision
- Recall
- mAP (mean Average Precision)
- F1-Score
- Inference time
- Model size

## Results

Results are stored in the `hasil/` directory:
- **Baseline/**: Contains results from baseline models
- **Optimized/**: Contains results from optimized models

Each result typically includes:
- Training metrics
- Validation metrics
- Confusion matrices
- Sample predictions
- Performance graphs

## Model Training

### Basic Training Command
```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8s.pt')

# Train
results = model.train(
    data='path/to/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

### Optimization Parameters
Common optimization parameters include:
- Learning rate scheduling
- Batch size adjustment
- Image size optimization
- Augmentation techniques
- Early stopping
- Model pruning

## Hardware Requirements

### Recommended
- GPU: NVIDIA GPU with CUDA support (T4 or better)
- RAM: 16GB or more
- Storage: 10GB free space

### Minimum
- GPU: Any CUDA-compatible GPU
- RAM: 8GB
- Storage: 5GB free space

## Google Colab

All notebooks are designed to run on Google Colab with GPU acceleration:
1. Upload notebooks to Google Colab
2. Enable GPU: Runtime → Change runtime type → GPU
3. Run cells sequentially

## API Keys

The project uses Roboflow API for dataset access. Replace the API key in the notebooks:
```python
rf = Roboflow(api_key="YOUR_API_KEY_HERE")
```

## Citation

If you use this code in your research, please cite:
```
@misc{armed-person-detection-yolo,
  author = {Toni},
  title = {YOLO Model Optimization for Armed Person Detection},
  year = {2024},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/yourusername/yourrepo}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Ultralytics for the YOLO implementation
- Roboflow for dataset hosting and management
- Google Colab for providing free GPU resources

## Contact

For questions or issues, please open an issue in the repository or contact the author.

## Future Work

- [ ] Implement real-time detection
- [ ] Deploy models to edge devices
- [ ] Add more optimization techniques
- [ ] Expand dataset with more samples
- [ ] Create web interface for inference
- [ ] Add model quantization
- [ ] Implement ensemble methods

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Reduce image size
   - Use gradient accumulation

2. **Dataset Download Fails**
   - Check API key
   - Verify internet connection
   - Check Roboflow workspace access

3. **Poor Model Performance**
   - Increase training epochs
   - Adjust learning rate
   - Check data quality
   - Verify augmentation settings

## Version History

- **v1.0** - Initial release with baseline models
- **v1.1** - Added optimization techniques
- **v1.2** - Improved documentation and results visualization
- **v2.0** - Added YOLOv11 support

---

**Note**: This is a research project for academic purposes. The models should be thoroughly tested before deployment in production environments.
