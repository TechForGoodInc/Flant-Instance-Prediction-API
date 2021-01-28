import logging
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.models import save_model, load_model
import pandas as pd
from datetime import datetime, timedelta

from modelmanager import util
from modelmanager.enums import MMState, MMStatus
from modelmanager.constants import Constants


class ModelManager:

    def __init__(self):
        self.open_models = dict()
        self.state = MMState.FREE

    # Train models with data from specific file
    def __process_file(self, file_name):
        logging.info('Processing file ' + file_name)

        processor = self.DataProcessor(file_name, Constants.DATA_PATH)

        test_counter = 5
        for instance_tag in processor.instance_tags.values:

            # TODO: Remove test limiter
            if test_counter == 0:
                break
            test_counter -= 1

            model = self.__open_model(instance_tag)

            logging.info(f'Training model {model.name}')
            self.state = MMState.TRAINING

            history = model.train(processor.get_training_data(instance_tag))

            self.__save_model_training_history(model, file_name, history)

            self.state = MMState.PREDICTING
            model.populate_predictions()

            self.__close_model(model)

            self.state = MMState.PROCESSING

        self.processed_data_files.insert(0, file_name)
        self.__save_processed_file()

        logging.info(f'Finished processing file {file_name}')

    def __save_model_training_history(self, model, file_name, history):
        util.save_df(file_name.split('.')[0] + '__' + model.name, Constants.HISTORY_PATH, pd.DataFrame(history.history))
        logging.debug(f'Saved training history for {file_name.split(".")[0]}__{model.name}')

    # Load model object
    def __open_model(self, instance_tag):
        model_name = util.encode_name(*instance_tag)
        model = self.Model(model_name, Constants.MODEL_PATH)

        self.open_models[model_name] = model

        return model

    # Release model object
    def __close_model(self, model):
        model.save()
        self.open_models.pop(model.name)

    # Write new `processed_data_files.pkl` contents
    def __save_processed_file(self):
        util.save_obj(self.processed_data_files, Constants.PROCESSED_DATA_FILE_NAME)

    def __load_processed_file(self):
        self.processed_data_files = util.load_obj(Constants.PROCESSED_DATA_FILE_NAME)

    def __load_predictions_file(self, instance):
        try:
            y_pred = pd.read_csv(Constants.PREDICTIONS_PATH + instance + '.csv', index_col=['Timestamp'])
            y_pred.index = pd.to_datetime(y_pred.index)
            return y_pred, MMStatus.OK
        except IOError:
            return None, MMStatus.NO_PREDICTIONS

    # Train models with data from all unprocessed files
    def process_unprocessed(self):
        self.state = MMState.PROCESSING
        self.__load_processed_file()

        for file_name in os.listdir(Constants.DATA_PATH):
            if file_name not in self.processed_data_files:
                self.__process_file(file_name)
            else:
                logging.debug(f'File {file_name} already processed, omitting')

        self.state = MMState.FREE

        return MMStatus.OK

    # Predict prices during specified time of a given instance
    def get_predictions(self, instance, start_date, end_date):
        y_pred, status = self.__load_predictions_file(instance)
        return \
            y_pred[(y_pred.index > start_date) & (y_pred.index <= end_date)] \
                if status is MMStatus.OK else pd.DataFrame(), \
            status

    class Model:

        def __init__(self, name, models_path):
            self.name = name
            self.tf_model = None
            # Predicted data with dates attached
            self.predicted_dates = None
            # Newest timestamp of data used to train model
            self.end_date = None
            self.end_data = None
            self.MODELS_PATH = models_path

            try:
                self.tf_model = load_model(self.MODELS_PATH + name)
                logging.debug(f'Loaded model {name} from file')
            except OSError:
                # Model hasn't yet been created, creating now
                self.tf_model = self.__generate_new_model()
                logging.debug(f'Created new model {name}')

        @staticmethod
        def __generate_new_model():
            model = keras.Sequential()
            model.add(keras.layers.LSTM(
                units=128,
                input_shape=(Constants.TRAIN_DATA_SHAPE[0], Constants.TRAIN_DATA_SHAPE[1])
            ))
            model.add(keras.layers.Dense(units=128, activation='relu',
                                         kernel_regularizer=regularizers.l1_l2(l1=1e-20, l2=1e-19),
                                         bias_regularizer=regularizers.l2(1e-19),
                                         activity_regularizer=regularizers.l2(1e-20)))
            model.add(keras.layers.Dense(units=Constants.TARGET_SIZE))
            model.compile(
                loss='mean_squared_error',
                optimizer=keras.optimizers.Adam(0.001)
            )

            return model

        # Reattach datetime timestamps to predicted data
        def __reattach_datetime(self, data):
            date = self.end_date + timedelta(days=1)
            self.predicted_dates = pd.DataFrame(dict(SpotPrice=data[0]),
                                                index=np.array(
                                                    [date + timedelta(days=i) for i in range(len(data[0]))]))

        def __save_predictions(self):
            util.save_df(
                self.name,
                'predictions/',
                self.predicted_dates,
                index_label='Timestamp'
            )
            logging.debug(f'Predictions written to file for {self.name}')

        # Save tensorflow model to file
        def save(self):
            save_model(self.tf_model, self.MODELS_PATH + self.name)
            logging.debug(f'Saved model {self.name} to file')

        def train(self, training_data):
            has_test_data = (training_data[4] is not None and training_data[5] is not None)
            self.end_date, self.end_data, X_train, y_train, X_test, y_test = training_data

            history = self.tf_model.fit(
                X_train,
                y_train,
                epochs=Constants.EPOCHS,
                batch_size=Constants.BATCH_SIZE,
                validation_split=0.1,
                verbose=1,
                shuffle=False
            )

            return history

        def populate_predictions(self):
            y_pred = self.tf_model.predict(np.array([self.end_data]))
            logging.debug(f'Obtained predictions for {self.name}')
            self.__reattach_datetime(y_pred)
            self.__save_predictions()

    class DataProcessor:

        def __init__(self, file_name, data_path, filter_constant_prices=True):
            self.df = pd.read_csv(data_path+file_name)
            self.instance_tags = None

            self.__convert_timestamps()
            if filter_constant_prices:
                self.__filter_constant_prices()
            self.__populate_instance_tags()

        # Cast 'Timestamp' column as DateTime type
        def __convert_timestamps(self):
            self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])

        # Remove data for instances with constant prices
        def __filter_constant_prices(self):
            group_cols = Constants.GROUP_COLS

            # Find standard deviation of price for each instance
            standard_dev = self.df.groupby(group_cols)['SpotPrice'].std().reset_index()

            # Isolate instances that have variable pricing (non 0 std)
            variable_pricing_names = standard_dev.query('SpotPrice > 0')[
                ['AvailabilityZone', 'InstanceType', 'ProductDescription']]

            # Filter instances with variable pricing
            self.df = pd.merge(self.df,
                               variable_pricing_names,
                               how='inner',
                               on=['AvailabilityZone', 'InstanceType', 'ProductDescription'])

        # Create dataset for training
        @staticmethod
        def __create_dataset(X, y):
            Xs, ys = [], []
            for i in range(len(X) - Constants.TIME_STEPS - Constants.TARGET_SIZE + 1):
                Xs.append(X.iloc[i:(i + Constants.TIME_STEPS)].values)
                ys.append(y.iloc[(i + Constants.TIME_STEPS):(i + Constants.TIME_STEPS + Constants.TARGET_SIZE)].values)
            return np.array(Xs), np.array(ys)

        # Extract all unique instances from data
        def __populate_instance_tags(self):
            # Get unique instance tags
            self.instance_tags = self.df[Constants.GROUP_COLS].drop_duplicates()

        @staticmethod
        def __get_end_data(data):
            return np.array(data.iloc[-Constants.TIME_STEPS:])

        # Return data ready for training
        def get_training_data(self, instance_tag, create_test_data=False):
            # Isolate target instance
            instance = self.df[
                (self.df.AvailabilityZone == instance_tag[0]) &
                (self.df.InstanceType == instance_tag[1]) &
                (self.df.ProductDescription == instance_tag[2])]

            # Prepare data frame
            instance = pd.DataFrame(
                dict(SpotPrice=instance['SpotPrice'].values),
                index=instance['Timestamp'].values,
                columns=['SpotPrice']).sort_index()

            end_date = instance.index[-1]

            end_data = self.__get_end_data(instance)

            # Split into train, test data
            train_size = int(len(instance) * Constants.TRAINING_SPLIT) if create_test_data else len(instance)
            train = instance.iloc[0:train_size]
            test = instance.iloc[train_size:len(instance)] if create_test_data else None

            # reshape to [samples, time_steps, n_features]
            X_train, y_train = self.__create_dataset(train, train.SpotPrice)
            X_test, y_test = self.__create_dataset(test, test.SpotPrice) if create_test_data else None, None

            return end_date, end_data, X_train, y_train, X_test, y_test