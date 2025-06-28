This code implements a machine learning pipeline for classifying data using a custom Random Forest model and compares it to a scikit-learn baseline. The main steps are:

1. **Data Loading and Preprocessing:**  
   - Loads brain, image, and text features from `.mat` files.
   - Extracts relevant time intervals and scales/reshapes the data.
   - Reduces dimensionality of image features.

2. **Data Partitioning:**  
   - The `partition_data` function splits the data into training and test sets, ensuring each class is equally represented and combines all modalities (brain, image, text) into a single feature array.

3. **Custom Model Implementation:**  
   - Implements a manual Random Forest classifier using custom Decision Tree logic.
   - Provides a function (`run_model`) to train and evaluate this model.

4. **Baseline and Improvements:**  
   - Compares the custom model to scikit-learn’s `RandomForestClassifier`.
   - Applies Linear Discriminant Analysis (LDA) for dimensionality reduction and scaling for improved performance.

5. **Hyperparameter Tuning:**  
   - Systematically tests different values for the number of trees, tree depth, and minimum samples split, saving results and plotting accuracy trends.

6. **Evaluation and Visualization:**  
   - Evaluates the best model using various metrics and visualizations.
   - Tests different train/test splits to find the optimal number of training samples.

Overall, the code is designed to experiment with and optimize a Random Forest classifier for multimodal data, providing both custom and standard implementations, and includes tools for thorough evaluation and hyperparameter tuning.
