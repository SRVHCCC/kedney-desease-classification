import tensorflow as tf
from pathlib import Path
import dagshub
import mlflow
import mlflow.keras
from urllib.parse import urlparse
import json





class Evaluation:
    def __init__(self, config):
        self.config = config

    def _valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        # Ensure the model and valid_generator are available
        if self.model and self.valid_generator:
            self.score = self.model.evaluate(self.valid_generator)
            self.save_score()
        else:
            raise ValueError("Model or validation generator could not be loaded properly.")

    def save_score(self):
        # Ensure score has valid values
        if self.score and len(self.score) >= 2:
            scores = {"loss": self.score[0], "accuracy": self.score[1]}
            self.save_json(path=Path("scores.json"), data=scores)
        else:
            raise ValueError("Score should have at least loss and accuracy.")

    @staticmethod
    def save_json(path: Path, data: dict):
        # Saves the data into a JSON file
        with open(path, "w") as json_file:
            json.dump(data, json_file)

    def log_into_mlflow(self):
        try:
            # Set the tracking URI and validate
            mlflow.set_registry_uri(self.config.mlflow_uri)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # Start an MLflow run
            with mlflow.start_run():
                # Log parameters
                if isinstance(self.config.all_params, dict):
                    mlflow.log_params(self.config.all_params)
                else:
                    raise ValueError("self.config.all_params must be a dictionary")

                # Log metrics (ensure score has loss and accuracy)
                if len(self.score) >= 2:
                    mlflow.log_metrics(
                        {"loss": self.score[0], "accuracy": self.score[1]}
                    )
                else:
                    raise ValueError("self.score should have at least 2 values: [loss, accuracy]")

                # Log and register the model
                if tracking_url_type_store != "file":
                    mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
                else:
                    mlflow.keras.log_model(self.model, "model")
        
        except Exception as e:
            print(f"Error logging into MLflow: {e}")
            raise