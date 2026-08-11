import requests, zipfile, io, os, datetime, time, json, splitfolders, joblib
from urllib.parse import urlparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.base import is_classifier, is_regressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
from htb_ai_library import use_htb_style, HTB_GREEN, NODE_BLACK, HACKER_GREY
use_htb_style()
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random_state=1337
torch.manual_seed(random_state)


#--------------------------------------------------------------------------------
# Download Dataset
def fetch_dataset(endpoint, zipped=True):
    # Download Dataset
    if not isinstance(endpoint, str):
        raise TypeError("Endpoint must be a valid string")
    
    result = urlparse(endpoint)
    if result.scheme != "https" or not result.netloc:
        raise ValueError("URL must be a valid HTTPS URL")
    url = endpoint   

    filename = Path(result.path).name
    download_dir = Path("downloads")
    download_dir.mkdir(exist_ok=True)
    download_path = download_dir/filename
    
    response = requests.get(url)
    response.raise_for_status()
    print("Download successful")
    
    if zipped:
    # Extract Dataset
        extract_path = download_dir/Path(filename).stem
        extract_path.mkdir(exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(extract_path)
        print(f"\nDataset successfully extracted to:")
        return extract_path.resolve()
        
    else:
        download_path.write_bytes(response.content)
        print(f"Dataset successfully downloaded to:")
        return download_path.resolve()


#--------------------------------------------------------------------------------
# Load Dataset
def load_data(file_path, header_names=None, dataset_type="csv", mean=None, std=None, sep=","):
    dataset_type = dataset_type.lower()
    if dataset_type == "csv":
        if header_names:
            if header_names != "not_available":
                header = None 
                names = header_names
            else:
                header = None
                names = None
        else:
            header = 0 
            names = None
        df = pd.read_csv(file_path, sep=sep, header=header, names=names)
        df = df.drop_duplicates()
        print("Dataset Structure:")
        print(df.describe())
        return df
    
    elif dataset_type == "image":
        DATA_BASE_PATH = file_path
        TARGET_BASE_PATH = "./newdata/"
        
        TRAINING_RATIO = 0.8
        TEST_RATIO = 1 - TRAINING_RATIO
        
        splitfolders.ratio(input=DATA_BASE_PATH, output=TARGET_BASE_PATH, ratio=(TRAINING_RATIO, 0, TEST_RATIO))
        
        # Define preprocessing transforms
        transform = transforms.Compose([transforms.Resize((75, 75)), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

        # Load training and test datasets
        train_dataset = ImageFolder(root=os.path.join(TARGET_BASE_PATH, "train"), transform=transform)
        test_dataset = ImageFolder(root=os.path.join(TARGET_BASE_PATH, "test"), transform=transform)
        TRAIN_BATCH_SIZE, TEST_BATCH_SIZE = 1024, 1024

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset,batch_size=TEST_BATCH_SIZE,shuffle=False,num_workers=2)
        n_classes = len(train_dataset.classes)
        return train_loader, test_loader, n_classes


#--------------------------------------------------------------------------------
# Preprocess Dataset
import re
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split

def clean_text(X):
    return X.apply(lambda x: re.sub(r"[^a-z\s$!]", "", x.lower()))
    #return X.fillna("").apply(lambda x: re.sub(r"[^a-z\s$!]", "", x.lower()))

def preprocess_data(df, y_col_name="price", text_cols=None, use_stopwords=True, test_size=0.2, random_state=random_state, mean=None, std=None):
    
    #X = df.drop('price', axis=1)
    X = df.drop(columns=y_col_name)
    y = df[y_col_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)  

    # Identify text, categorical and numerical columns automatically
    text_cols = text_cols or []
    # Validate text columns
    missing_text_cols = [col for col in text_cols if col not in X.columns]
    if missing_text_cols:
        raise ValueError(f"Text columns not found: {missing_text_cols}")
        
    categorical_cols = [col for col in X.select_dtypes(include=['object', 'category']).columns if col not in text_cols]
    numerical_cols = X.select_dtypes(include=["number"]).columns.tolist() 

    # Create preprocessing transformers for both categorical, text and numerical data
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numerical_transformer = StandardScaler()

    # Combine preprocessing steps
    transformers= [
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
    ]

    for i, col in enumerate(text_cols):
        text_transformer = Pipeline([
            ("clean", FunctionTransformer(clean_text)),
            ("tfidf", TfidfVectorizer(
                stop_words="english" if use_stopwords else None,
                min_df=1,
                max_df=0.9,
                ngram_range=(1, 2)
            ))
        ])
        transformers.append((f"text_{i}", text_transformer, col))

    preprocessor = ColumnTransformer(transformers=transformers)
    # Further split the training set into separate training and validation sets
    X_train_set, X_val_set, y_train_set, y_val_set = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state)

    # Fit the preprocessor on training data only
    X_train_set = preprocessor.fit_transform(X_train_set)
    X_val_set = preprocessor.transform(X_val_set)
    X_test_set = preprocessor.transform(X_test)  # Apply learned parameters without fitting
    
    
    return X_train_set, X_val_set, X_test_set, y_train_set, y_val_set, y_test, preprocessor


#---------------------------------------------------------------------------------------
# Implement Model Selection Logic
def train_model(X_train=None, y_train=None, model_training_alg="random_forest", preprocessor=None, train_loader=None, n_epochs=None, n_classes=None, **model_params):
    print(f"Training a {model_training_alg} model...")

    if model_training_alg == "random_forest":
        model = RandomForestClassifier(n_jobs=-1, **model_params)
    
    elif model_training_alg == "random_forest_regressor":
        # Default parameters if not specified
        model_params.setdefault('n_estimators', 100)  # Using a sensible default
        model_params.setdefault('random_state', 1337)  # For reproducibility
        model_params.setdefault('max_depth', 10)     
        model_params.setdefault('n_jobs', -1) 
        model = RandomForestRegressor(**model_params)

    elif model_training_alg == "linear":
        model = LinearRegression(**model_params)
        model_params.setdefault('random_state', 1337)

    elif model_training_alg == "naive_bayes":
        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", MultinomialNB())])
        param_grid = {"classifier__alpha": [0.001, 0.01, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0]}
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1")
        grid_search.fit(X_train, y_train)
        # Extract the best model identified by the grid search
        best_model = grid_search.best_estimator_
        print("Best model parameters:", grid_search.best_params_)
        return best_model

    elif model_training_alg == "cnn":
        model = ResNetClassifier(n_classes=n_classes, n_channels=3, fine_tune=False).to(DEVICE)
        model.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        training_data = {"accuracy": [], "loss": []}
        for epoch in range(n_epochs):
            running_loss, n_total, n_correct = 0, 0, 0
            checkpoint = time.time() * 1000

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                _, predicted = outputs.max(1)
                n_total += labels.size(0)
                n_correct += predicted.eq(labels).sum().item()
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(train_loader)
            epoch_duration = int(time.time() * 1000 - checkpoint)
            epoch_accuracy = compute_accuracy(n_correct, n_total)
            
            training_data["accuracy"].append(epoch_accuracy)
            training_data["loss"].append(epoch_loss)

            intervals = n_epochs // 4
            if epoch % intervals == 0:
                print(f"[i] Epoch {epoch+1} of {n_epochs}: Acc: {epoch_accuracy:.2f}% Loss: {epoch_loss:.4f} (Took {epoch_duration} ms).")    
        
        return model, training_data
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    print("Model training completed!")
    return model


#----------------------------------------------------------------------------------
# Reusable Evaluation Function
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def evaluate_model(model, X_val, y_val):
    # Generate predictions
    y_pred, prediction_probabilities = predict_with_model(model, X_val)

    if is_regressor(model):
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
            'r2': r2_score(y_val, y_pred),
            'mae': mean_absolute_error(y_val, y_pred)
        }
        
    else:
        conf_matrix = confusion_matrix(y_val, y_pred)
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, average='weighted'),
            'recall': recall_score(y_val, y_pred, average='weighted'),
            'f1_score': f1_score(y_val, y_pred, average='weighted'),
            'conf_matrix': conf_matrix,
            'cls_report': classification_report(y_val, y_pred)
        }
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='rocket_r')
        plt.title('Confusion Matrix - Evaluation Set')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    return metrics, y_pred, prediction_probabilities

# --- Evaluation for Neural Network Models ---
def compute_accuracy(n_correct, n_total):
    return round(100 * n_correct / n_total, 2)

def evaluate_nn(model, test_loader):
    model.eval()
    n_correct, n_total = 0, 0   
    with torch.no_grad():
        for data, target in test_loader:
            predicted = predict_with_model(model, data, model_type="neuralnet")
            n_total += target.size(0)
            n_correct += (predicted == target).sum().item()
    accuracy = compute_accuracy(n_correct, n_total) 
    print(f"[i] Inference accuracy: {accuracy}%.")
    return accuracy


#-----------------------------------------------------------------------------------
# Creating a Prediction Function (to be enhanced later with input validation, error handling and logging)
def predict_with_model(model, X, model_type=None):
    if model_type == "neuralnet":
        model.eval()
        with torch.no_grad():
            output = model(X)
            _, predicted = torch.max(output.data, 1)
        return predicted

    else:
        predictions = model.predict(X)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)
        else:
            probabilities = None
        return predictions, probabilities


#-------------------------------------------------------------------------------------
# Model Persistence: Saving and Loading ML Models:
def save_model(model, preprocessor=None, metadata=None, model_dir="saved_models", model_type=None):
    
    # Create the directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    if model_type == "neuralnet":
        model_name = f"{model.__class__.__name__}.pt"
        model_scripted = torch.jit.script(model)
        model_path = os.path.join(model_dir, model_name)
        model_scripted.save(model_path)
        print(f"{model_name} Model saved to {model_dir}")
        return model_path, model_name 
    
    else:
        # Generate a timestamped name if none provided
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{model.__class__.__name__}_{timestamp}"

        # Create file paths for each component
        model_path = os.path.join(model_dir, f"{model_name}.joblib")
        if preprocessor:
            preprocessor_path = os.path.join(model_dir, f"{model_name}_preprocessor.joblib")
        metadata_path = os.path.join(model_dir, f"{model_name}_metadata.json")
        
        # Save model and preprocessor using joblib
        joblib.dump(model, model_path)
        if preprocessor:
            joblib.dump(preprocessor, preprocessor_path)
        
        # Prepare and save metadata
        if metadata is None:
            metadata = {}
        
        # Enhance metadata with additional information
        metadata["timestamp"] = datetime.datetime.now().isoformat()
        metadata["model_path"] = model_path
        if preprocessor and preprocessor_path:
            metadata["preprocessor_path"] = preprocessor_path
        metadata["model_type"] = model.__class__.__name__
        
        # Save metadata as JSON (human-readable format)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"{model_name} Model saved to {model_path}")
        print(f"Preprocessor saved to {preprocessor_path}")
        print(f"Metadata saved to {metadata_path}")
        
        return model_path, model_name
        

#--------------------------------------------------------------------------------
# Basic Model Loading Function
def load_model(model_path, preprocessor_path=None, model_type=None):
    print(f"Loading model from {model_path}")

    if model_type == "neuralnet":
        model = torch.jit.load(model_path, map_location=DEVICE)
    else:
        model = joblib.load(model_path)
    
    # Optionally load preprocessor
    preprocessor = None
    if preprocessor_path:
        print(f"Loading preprocessor from {preprocessor_path}")
        preprocessor = joblib.load(preprocessor_path)
    
    return model, preprocessor


#-------------------------------------------------------------------------------------
# Complete Model Loading Function
def load_model_with_metadata(model_dir, model_name, model_type=None):

    if model_type == "neuralnet":
        model_path = os.path.join(model_dir, f"{model_name}.pt")
        model = load_model(model_path, model_type=model_type)
        
    else:
        # Construct file paths based on naming convention
        model_path = os.path.join(model_dir, f"{model_name}.joblib")
        preprocessor_path = os.path.join(model_dir, f"{model_name}_preprocessor.joblib")
        metadata_path = os.path.join(model_dir, f"{model_name}_metadata.json")
        
        # Validate that critical files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Handle missing preprocessor gracefully
        if not os.path.exists(preprocessor_path):
            preprocessor_path = None
            print(f"Warning: Preprocessor file not found")
        
        # Load model and preprocessor using the basic function
        model, preprocessor = load_model(model_path, preprocessor_path)
        
        # Load metadata if available
        metadata = None
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return model, preprocessor, metadata


#--------------------------------------------------------------------------------------
# Custom functions are placed here:

# --- KDD Dataset (Network Anomaly Detection) ---
# Dataset-specific configuration

kdd_columns = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','attack','level']

attack_map = {1: ("dos_attacks", ["apache2","back","land","neptune","mailbomb","pod","processtable","smurf","teardrop","udpstorm","worm"]), 2: ("probe_attacks", ["ipsweep","mscan","nmap","portsweep","saint","satan"]), 3: ("privilege_attacks", ["buffer_overflow","loadmdoule","perl","ps","rootkit","sqlattack","xterm"]), 4: ("access_attacks", ["ftp_write","guess_passwd","http_tunnel","imap","multihop","named","phf","sendmail","snmpgetattack","snmpguess","spy","warezclient","warezmaster","xclock","xsnoop"])}

def prepare_kdd(df):
    df['attack_flag'] = df['attack'].apply(lambda a: 0 if a == 'normal' else 1)
    attack_lookup = {attack: category_id for category_id, (_, attacks) in attack_map.items() for attack in attacks}
    df["attack_map"] = df["attack"].map(attack_lookup).fillna(0).astype(int)
    return df.drop(columns=["attack","attack_flag"])


# --- ResNet model class ---
HIDDEN_LAYER_SIZE = 1000
class ResNetClassifier(nn.Module):
    def __init__(self, n_classes, n_channels=3, fine_tune=False):
        super(ResNetClassifier, self).__init__()
        # Load pretrained ResNet50
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        if n_channels == 1:
            old_conv = self.model.conv1
            self.model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
            self.model.conv1.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        # Freeze ResNet parameters
        for param in self.model.parameters():
            param.requires_grad = False

        if fine_tune:
            for param in self.model.layer4.parameters():
                param.requires_grad = True
        
        # Replace the last fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, HIDDEN_LAYER_SIZE),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(HIDDEN_LAYER_SIZE, n_classes)
        )
        
    def forward(self, x):
        return self.model(x)


#-----------------------------------------------------------------------------------
def load_saved_model(model_name):
    # Later: Load the model and use it
    print("\n--- Loading and Using Model ---")
    loaded_model, loaded_preprocessor, loaded_metadata = load_model_with_metadata(model_dir="saved_models", model_name=model_name)
    
    # Display key metadata to verify what we loaded
    print("\nModel Metadata:")
    for key, value in loaded_metadata.items():
        if key not in ["metrics", "parameters"]:
            print(f"  - {key}: {value}")
    
    # Use the loaded model to make predictions
    print("\nMaking predictions with loaded model...")
    loaded_preds, _ = predict_with_model(loaded_model, X_test)
    loaded_metrics, _, _ = evaluate_model(loaded_model, X_test, y_test)
    
    print("Loaded Model Metrics:")
    for metric_name, metric_value in loaded_metrics.items():
        print(f"  - {metric_name}: {metric_value:.4f}")

def load_saved_nn_model(model_name):
    print("\n--- Loading and Using Model ---")
    loaded_model = load_model_with_metadata(model_dir="saved_models", model_name=model_name, model_type="neuralnet")

    # Use the loaded model to make predictions
    print("\nMaking predictions with loaded model...")
    loaded_preds = predict_with_model(loaded_model, X, model_type="neuralnet")
    loaded_metrics = evaluate_nn(loaded_model, test_loader)


#---------------------------------------------------------------------------------------
def upload_model_result():
    url = "http://localhost:8000/api/upload"
    
    # Path to the model file you want to upload
    model_file_path = "spam_detection_model.joblib"
    
    # Open the file in binary mode and send the POST request
    with open(model_file_path, "rb") as model_file:
        files = {"model": model_file}
        response = requests.post(url, files=files)
    
    # Pretty print the response from the server
    print(json.dumps(response.json(), indent=4))



# -------------------------------------------------------------------------------
def plot(data, title, label, xlabel, ylabel):
    # HTB Color Palette
    htb_green = "#9FEF00"
    node_black = "#141D2B"
    hacker_grey = "#A4B1CD"

    # plot
    plt.figure(figsize=(10, 6), facecolor=node_black)
    plt.plot(range(1, len(data)+1), data, label=label, color=htb_green)
    plt.title(title, color=htb_green)
    plt.xlabel(xlabel, color=htb_green)
    plt.ylabel(ylabel, color=htb_green)
    plt.xticks(color=hacker_grey)
    plt.yticks(color=hacker_grey)
    ax = plt.gca()
    ax.set_facecolor(node_black)
    ax.spines['bottom'].set_color(hacker_grey)
    ax.spines['top'].set_color(node_black)
    ax.spines['right'].set_color(node_black)
    ax.spines['left'].set_color(hacker_grey)

    legend = plt.legend(facecolor=node_black, edgecolor=hacker_grey, fontsize=10)
    plt.setp(legend.get_texts(), color=htb_green)
    
    plt.show()

def plot_training_accuracy(training_data):
    plot(training_data['accuracy'], "Training Accuracy", "Accuracy", "Epoch", "Accuracy (%)")

def plot_training_loss(training_data):
    plot(training_data['loss'], "Training Loss", "Loss", "Epoch", "Loss")
