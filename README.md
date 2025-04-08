**Grammar Scoring Engine**



**Audio-to-Text Conversion**  
The process starts by transcribing audio files into text using OpenAI's Whisper "base" model, leveraging GPU acceleration when available (via CUDA). Audio files are sourced from predefined training and testing directories (`audios_train` and `audios_test`) stored on Google Drive. Transcription uses FP16 precision if a GPU is detected, otherwise it defaults to FP32 for consistency. Each audio file’s transcription is printed immediately after processing, enabling real-time inspection of the text output for both training and test datasets.

**Feature Extraction**  
Two feature types are extracted from the transcribed text. First, LanguageTool (configured for American English, "en-US") analyzes the text and returns the count of grammatical errors as a feature. Second, a pre-trained BERT model (`bert-base-uncased`) generates embeddings on the GPU: the text is tokenized with a maximum length of 128 tokens, padded and truncated as needed, and the [CLS] token’s vector (768 dimensions) is extracted from the last hidden state. These features—grammar error count and BERT embeddings—are concatenated into a single feature vector per audio file using NumPy.

**Training Data Preparation**  
Training data is loaded from a CSV file (`train.csv`) containing filenames and corresponding labels. For each audio file, transcription is performed, followed by feature extraction (grammar errors and BERT embeddings). Only successfully transcribed files are processed; failed transcriptions are skipped, and valid indices are tracked. The resulting feature vectors form `X_train`, paired with labels in `y_train`. A `StandardScaler` normalizes these features to optimize performance for the downstream machine learning model.

**XGBoost Model for Scoring**  
An XGBoost regression model (`XGBRegressor`) is trained on the scaled training features to predict scores. It is configured with 100 estimators, a maximum depth of 5, a learning rate of 0.1, and uses `reg:squarederror` as the objective. GPU acceleration is enabled with the `tree_method="hist"` and `device="cuda"` settings when a GPU is available, otherwise it defaults to CPU. The model learns the mapping between the combined grammar-BERT features and the provided labels. Post-training, performance is evaluated on the training set: predictions are rounded to the nearest integer and clipped to a 1.0-to-5.0 range, with metrics including a classification report, confusion matrix, and mean squared error (MSE) printed for analysis.

**Test Data and Prediction**  
Test data, loaded from `test.csv`, follows a parallel pipeline: audio files are transcribed, features are extracted (grammar errors and BERT embeddings), and the resulting vectors are scaled using the same `StandardScaler` fitted on the training data. The trained XGBoost model predicts scores (`y_test_pred`) for the test set. Predictions are refined to a 1.0-to-5.0 scale in 0.5 increments using CuPy for GPU-accelerated rounding: raw scores are multiplied by 2, rounded, divided by 2, and clipped to the target range, then converted back to NumPy.

**Output and Evaluation**  
The final predictions, paired with valid test filenames, are saved to a `submission.csv` file, which is automatically downloaded via Google Colab’s `files.download()`. During training evaluation, the model’s performance is assessed with a classification report (treating scores as discrete classes from 1 to 5), confusion matrix, and MSE, though the description does not specify an exact accuracy (e.g., 85% was in the original but not verifiable here). This workflow suggests an audio assessment task, likely scoring speech quality or grammatical correctness.
