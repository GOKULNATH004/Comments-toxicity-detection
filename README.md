🚀 Comment Toxicity Detection

This project focuses on detecting toxicity in online comments using a deep learning approach. The model classifies comments into different toxicity categories, such as moderate, high, threatening, and racist. It employs Natural Language Processing (NLP) techniques and a hybrid CNN-LSTM architecture to manage word embeddings and text classification. The model is deployed using a Gradio web application, allowing users to input comments and receive real-time toxicity predictions.

🔥 Key Features

🏷️ Classification Categories

🟡 Moderately Toxic Comments

🔴 Highly Toxic Comments

⚠️ Threatening Comments

🏴 Racist Comments

🧠 Deep Learning Architecture

🖼️ CNN: Extracts essential features from comments.

📖 LSTM: Captures word dependencies and contextual meaning.

📊 Dense Layers: Performs final classification into toxicity categories.

⚡ Real-time Detection

🎛️ Gradio Interface: Users can input comments and instantly receive toxicity classification results.

🎯 Gen Z Term Adaptation

🆕 Incorporates Modern Internet Slang & Gen Z Language to ensure accurate detection of contemporary online speech patterns, improving model efficiency based on feedback.

📂 Project Structure

├── data/        
│   ├── train.csv       # Training dataset
│   ├── test.csv        # Testing dataset
│
├── models/       
│   ├── toxicity_model.h5 # Trained model in h5 format
│
├── notebooks/  
│   ├── data_preprocessing.ipynb # Data cleaning and preprocessing
│   ├── model_training.ipynb # Model architecture and training
│
├── app/         
    ├── gradio_app.py # Gradio web app for model deployment


🔧 Installation

📌 Prerequisites

✅ Python 3.7+

✅ TensorFlow 2.x

✅ Gradio 3.x


🚀 Usage

Once the Gradio web app is running, users can input any comment, and the model will classify it into one of the following categories:

🟡 Moderately Toxic

🔴 Highly Toxic

⚠️ Threatening

🏴 Racist

The output will display a binary label (0 for non-toxic, 1 for toxic) for each category, allowing users to quickly identify harmful content.

📊 Dataset

The dataset used in this project contains a collection of online comments, each labeled with one or more toxicity categories. The comments undergo preprocessing, including:

✂ Tokenization: Splitting text into individual words/tokens.

🔢 Embedding: Converting tokens into numerical vectors for model input.

The dataset is split into training and testing subsets to ensure the model generalizes well to unseen data.

🔮 Future Improvements

🌍 Multilingual Support: Extending the model to handle multiple languages for broader applicability.

😊 Sentiment Analysis: Incorporating sentiment analysis to understand the emotional tone of comments.

🏳️‍🌈 Additional Toxicity Categories: Adding detection for misogyny, homophobia, and other nuanced toxicity types.

👨‍💻 Developers

🎓 [Anushya Varshini K](https://github.com/anushya03)

🎓 [Rithick M K](https://github.com/rithick-06)

🎓 [Gokulnath G](https://github.com/GOKULNATH004)

💡 Contributions are welcome! Feel free to open an issue or submit a pull request.

📜 License

This project is open-source and available under the MIT License.

