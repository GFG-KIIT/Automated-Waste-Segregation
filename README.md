# Automated-Waste-Segregation-through-Object-Recognition-for-Sustainable-Waste-Management
# Garbage Classification and Detection with YOLOv5

This project performs garbage classification using YOLOv5 for object detection. Each image is processed to detect objects (garbage) and save the results with bounding boxes and confidence scores.

## Libraries and Tools Used

1. **os**: A standard library for file and directory manipulation, used to navigate the dataset directory and manage output files.
  
2. **shutil**: Useful for managing and copying files and directories.

3. **numpy**: A widely-used library for numerical and array computations. Here, it is useful for data manipulation and image processing.
   
4. **time**: A standard library used for tracking the execution time of processes.

5. **matplotlib.pyplot**: Although imported, it may not be used directly in this code. It is often used for visualizations, such as plotting ROC curves, confusion matrices, or other analysis charts.

6. **scikit-learn**: Specifically, metrics like `confusion_matrix`, `classification_report`, and `roc_auc_score` can be used for model evaluation. However, these metrics are not used in the code snippet provided.

7. **json**: A standard library for parsing and storing data in JSON format. Not directly used here but can be helpful for configuration management.

8. **seaborn**: Another visualization library, often used alongside `matplotlib` to create appealing and informative charts. Although imported, it is not utilized directly in this code.

9. **cv2 (OpenCV)**: OpenCV is used for reading and manipulating images. Here, it loads images and performs operations such as drawing bounding boxes and labels for each detected object.

10. **torch**: The primary deep learning library used for loading and running the YOLOv5 model. Here, it loads the YOLOv5 model and applies it to perform object detection.

11. **torchvision**: A PyTorch library that provides various tools for computer vision, including models, datasets, and transformations. Although not explicitly called in this code, it is essential for certain computer vision tasks.

12. **pathlib.Path**: Provides an object-oriented approach to file and directory paths, allowing easy creation and management of the output directory.

## Setup Instructions

1. **Install the required libraries**:
   ```bash
   pip install opencv-python-headless numpy torchvision

