# Automated-Waste-Segregation-through-Object-Recognition-for-Sustainable-Waste-Management
This project performs garbage classification using YOLOv5 for object detection. Each image is processed to detect objects (garbage) and save the results with bounding boxes and confidence scores.

Libraries and Tools Used
os: A standard library for file and directory manipulation, used to navigate the dataset directory and manage output files.

shutil: Useful for managing and copying files and directories.

numpy: A widely-used library for numerical and array computations. Here, it is useful for data manipulation and image processing.

time: A standard library used for tracking the execution time of processes.

matplotlib.pyplot: Although imported, it may not be used directly in this code. It is often used for visualizations, such as plotting ROC curves, confusion matrices, or other analysis charts.

scikit-learn: Specifically, metrics like confusion_matrix, classification_report, and roc_auc_score can be used for model evaluation. However, these metrics are not used in the code snippet provided.

json: A standard library for parsing and storing data in JSON format. Not directly used here but can be helpful for configuration management.

seaborn: Another visualization library, often used alongside matplotlib to create appealing and informative charts. Although imported, it is not utilized directly in this code.

cv2 (OpenCV): OpenCV is used for reading and manipulating images. Here, it loads images and performs operations such as drawing bounding boxes and labels for each detected object.

torch: The primary deep learning library used for loading and running the YOLOv5 model. Here, it loads the YOLOv5 model and applies it to perform object detection.

torchvision: A PyTorch library that provides various tools for computer vision, including models, datasets, and transformations. Although not explicitly called in this code, it is essential for certain computer vision tasks.

pathlib.Path: Provides an object-oriented approach to file and directory paths, allowing easy creation and management of the output directory.

Setup Instructions
Install the required libraries:

bash
Copy code
pip install opencv-python-headless numpy torchvision
Clone the YOLOv5 repository:

bash
Copy code
git clone https://github.com/ultralytics/yolov5.git
Load the YOLOv5 model: This script uses the pretrained YOLOv5 model for object detection on images.

Code Process
Load YOLOv5 Model:

The script loads a pre-trained yolov5s model from the ultralytics/yolov5 repository using torch.hub.load().
Set Input and Output Paths:

It sets up the paths for the input images (where the dataset resides) and the output directory for storing processed images.
Iterate through Dataset and Perform Detection:

For each class folder and each image file in the input_dir, the code checks if the file is a valid image (.png, .jpg, .jpeg).
Images are loaded with OpenCV and fed into the YOLOv5 model for object detection.
Detection and Visualization:

The results from YOLOv5 are processed, extracting bounding box coordinates, confidence scores, and class labels.
Bounding boxes and labels are drawn onto each image to indicate detected objects and their confidence levels.
Save Results:

Each processed image is saved to output_dir, prefixed with detected_.
Completion Message:

After all images are processed, a message confirms the completion and provides the location of saved images.
Run the Code
Ensure Required Packages and Model: Follow the setup instructions to install dependencies and download the YOLOv5 repository.

Run the Script:

bash
Copy code
python your_script.py
Processed images will be saved in the output directory (/kaggle/working/detected_images in this case).
