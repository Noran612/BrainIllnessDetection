
### Project Title
**MRI-Based Brain Tumor Detection Using Grad-CAM Visualization**

### Project Description
This project leverages Convolutional Neural Networks (CNNs) to detect brain tumors from MRI scans and provides visual explanations of the model’s predictions using Grad-CAM (Class Activation Mapping). The goal is to identify critical regions within MRI images that the model focuses on for tumor detection, making the model's predictions more interpretable and reliable in a medical context.

As a future extension, I plan to develop this project into a user-friendly application, with expanded functionality and features to support broader use cases in medical imaging analysis.

### Key Features
- **Custom CNN Architecture**: A tailored CNN model designed to process MRI images and predict the presence of brain tumors.
- **Grad-CAM Visualization**: Highlights regions of the MRI scan that are most influential in the model’s decision, aiding interpretability.
- **Flexible Code Structure**: Allows for easy modification of layers and parameters, providing a foundation for future improvements.

### Folder Structure
- `data/`: Folder to store MRI images (not included here due to privacy and size constraints).
- `model/`: Model definition and helper functions for training and Grad-CAM visualization.
- `notebooks/`: Jupyter notebooks to test and experiment with different model architectures and Grad-CAM visualizations.
- `app/` (planned): Future folder to contain files for a web or mobile application interface.

---

### How to Run the Project

#### Prerequisites
1. **Python 3.7+**: Make sure Python is installed.
2. **Libraries**:
   - PyTorch
   - OpenCV
   - Matplotlib
   - Imutils (for image contour manipulation)
   
   You can install the dependencies with:
   ```bash
   pip install torch opencv-python matplotlib imutils
   ```

#### Dataset
Due to privacy concerns, the dataset (MRI images labeled for tumor presence) is not included in this repository. You may use any publicly available MRI dataset or your own dataset by placing it in the `data/` folder and updating the file paths in the code.

#### Running the Model and Grad-CAM
1. **Prepare the Data**: Load and preprocess the MRI scans, ensuring they are resized and normalized as required by the model.
2. **Train the Model** (optional): If you want to train the model, run the training script provided (or modify the notebook in `notebooks/`).
3. **Generate CAM Heatmaps**:
   - Use the `grad_cam` function to produce heatmaps for model predictions, highlighting areas of interest in the MRI images.
   - You can visualize the CAM heatmaps overlaid on the original MRI images by running the example notebook.

#### Example Usage
```python
# Import necessary libraries
import torch
from model import BrainDetectionModel, grad_cam
import cv2
import matplotlib.pyplot as plt

# Load and preprocess an MRI scan
image = cv2.imread("path/to/your/image.jpg")
preprocessed_image = preprocess(image)

# Initialize the model
model = BrainDetectionModel(input_shape=(3, 240, 240), num_cond=2)
model.load_state_dict(torch.load("path/to/your/model.pth"))  # Load pre-trained model

# Generate and visualize CAM
cam = grad_cam(model, preprocessed_image, target_class=1)
visualize_cam(cam, image)  # Display the heatmap overlayed on the original image
```

### Future Plans
I am planning to expand this project by:
- **Developing a Mobile/Web Application**: Create an intuitive interface where users can upload MRI images and get immediate tumor detection results, with visual explanations.
- **Enhanced Features**: Include additional functionality such as automated dataset processing, advanced visualization options, and support for multiple medical imaging modalities.

This project is a work in progress, and I am open to collaborations, feedback, and suggestions. Feel free to contribute or reach out if you have ideas for improvement!

