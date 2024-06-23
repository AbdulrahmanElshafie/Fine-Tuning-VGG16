# Fine-Tuning-VGG16 - Image Classification with Transfer Learning and Fine-Tuning
This repository demonstrates image classification using transfer learning and fine-tuning with TensorFlow and Keras. Here's how it works:

## Data
### Data Extraction (if using Google Colab):
   - The provided code assumes your compressed training, validation, and test data are in RAR archives named `train.rar`, `test.rar`, and `val.rar` located in your Colab workspace (`/content`).
   - The script `extract_rar_folder.py` extracts these archives into the corresponding directories (`/content/sample_data/train`, `/content/sample_data/test`, and `/content/sample_data/val`).

### Data Loading and Preprocessing

- **Data Paths:**
   - The script defines paths for the training, validation, and test directories after extraction.
- **ImageDataGenerator:**
   - Utilizes `ImageDataGenerator` from Keras for efficient image loading, resizing, and augmentation (optional in this example).
   - Sets `target_size` to (224, 224) to match the input size of the pre-trained MobileNet model.
   - Applies `mobilenet.preprocess_input` for pre-processing specific to the MobileNet model.
   - Creates data generators for training, validation, and testing.

## Transfer Learning

### Load Pre-trained Model:
   - Loads the pre-trained VGG16 model (you can experiment with other models like MobileNet as used in the data preprocessing).
   - Prints a summary of the model's architecture using `model.summary()`.
### Create Transfer Learning Model:
   - Initializes a new sequential model (`model`).
   - Iterates through all layers of the pre-trained model **except the last layer** and adds them to the new model.
   - Freezes the weights of these transferred layers using `layer.trainable = False` to prevent them from being re-trained during fine-tuning.
   - Adds a new dense layer with 2 units (assuming your classification task has 2 classes) and a softmax activation for multi-class classification.
   - Prints a summary of the modified model.

## Model Training

### Compilation:
   - Compiles the model using the Adam optimizer, categorical crossentropy loss function, and accuracy metric.
### Fine-Tuning:
   - Trains the model on the training data generator (`train_batches`) with validation data from the validation data generator (`valid_batches`).
   - Sets `steps_per_epoch` and `validation_steps` to control the number of batches per epoch for training and validation, respectively (adjust these values based on your dataset size).
   - Sets `epochs` to 5 (experiment with different epochs to find the optimal training duration).

## Evaluation (Optional)

- Uncomment the code for model evaluation:
   ```python
   # Evaluate the model on the test set
   test_loss, test_acc = model.evaluate(test_batches)
   print('Test accuracy:', test_acc)
   ```
