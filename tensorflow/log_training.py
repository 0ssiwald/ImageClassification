import json
import os
from datetime import datetime
import tensorflow
from contextlib import redirect_stdout
import io
from tensorflow.keras import layers, models, optimizers

class TrainingLogger:

    def __init__(self, epochs, learning_rate, batch_size, optimizer, model=None):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.model_summary = None
        self.train_accuracy = None
        self.train_val_accuracy = None
        self.train_loss = None
        self.train_val_loss = None
        self.test_accuracy = None
        self.test_loss = None 
        self.log_data = {}
        self.trials_data = []

    def update_train_metrics(self, accuracy, val_accuracy, loss, val_loss):
        self.train_accuracy = accuracy
        self.train_val_accuracy = val_accuracy
        self.train_loss = loss
        self.train_val_loss = val_loss
    
    def update_test_metrics(self, accuracy, loss):
        self.test_accuracy = accuracy
        self.test_loss = loss

    def capture_model_summary(self, model):
        stream = io.StringIO()
        with redirect_stdout(stream):
            model.summary()
        self.model_summary = stream.getvalue()   

    def log_trial(self, trial):
        # Logs a single trial's hyperparameters and evaluation metrics.
        trial_data = {
            "trial_id": trial.trial_id,
            "hyperparameters": trial.hyperparameters.values,
            "score": trial.score,
            "best_step": trial.best_step,
        }
        self.trials_data.append(trial_data) 
    
    def trial_data_as_string(self):
        trial_string = ""
        for trial in self.trials_data:
            trial_string += f'''{trial} \n'''
        return trial_string

    def create_string(self):
        return f'''
"epochs": {self.epochs},
"learning_rate": {self.learning_rate},
"batch_size": {self.batch_size},
"optimizer": {self.optimizer},
"train_accuracy": {self.train_accuracy},
"train_val_accuracy": {self.train_val_accuracy},
"train_loss": {self.train_loss},
"train_val_loss": {self.train_val_loss},
"test_accuracy": {self.test_accuracy},
"test_loss": {self.test_loss},
"timestamp": {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, 
"model_summary": {self.model_summary},
"trials_data": 
{self.trial_data_as_string()}
'''

    def save_log_to_json(self, directory='training_logs'):
        # Create the directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)

        # Generate a filename with the current date
        current_date = datetime.now().strftime("%Y_%m_%d")
        file_name = f"training_log_{current_date}.json"
        file_path = os.path.join(directory, file_name)

        # Prepare log data
        self.log_data = {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "optimizer": self.optimizer,
            "train_accuracy": self.train_accuracy,
            "train_val_accuracy": self.train_val_accuracy,
            "train_loss": self.train_loss,
            "train_val_loss": self.train_val_loss,
            "test_accuracy": self.test_accuracy,
            "test_loss": self.test_loss,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_summary": self.model_summary,
            "trials_data": self.trials_data,
        }

        # Read the existing file or create a new list if the file doesn't exist
        if os.path.exists(file_path):
            with open(file_path, 'r') as logfile:
                try:
                    logs = json.load(logfile)
                except json.JSONDecodeError:
                    logs = []
        else:
            logs = []

        # Append the new log data to the logs list
        logs.append(self.log_data)

        # Save the updated logs list back to the JSON file
        with open(file_path, 'w') as logfile:
            json.dump(logs, logfile, indent=4)

    def save_log_to_txt(self, directory='training_logs'):
         # Create the directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)

        # Generate a filename with the current datetime
        current_datetime = datetime.now().strftime("%Y_%m_%d")
        file_name = f"training_log_{current_datetime}.txt"
        file_path = os.path.join(directory, file_name)

        log_string = self.create_string()
        
        with open(file_path, 'a') as logfile:
            logfile.write(log_string)
            
    def print_log(self):
        print(self.create_string())

    def print_and_save_log(self):
        self.print_log()
        self.save_log_to_json()
        self.save_log_to_txt()

    
