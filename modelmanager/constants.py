class Constants:
    # ModelManager
    DATA_PATH = 'data/'
    MODEL_PATH = 'model/'
    HISTORY_PATH = 'history/'
    PREDICTIONS_PATH = 'predictions/'

    PROCESSED_DATA_FILE_NAME = 'processed_data_files'

    # Model
    TRAIN_DATA_SHAPE = (10, 1)
    EPOCHS = 30
    TARGET_SIZE = 30
    BATCH_SIZE = 1

    # Data Processor
    TIME_STEPS = 10
    TRAINING_SPLIT = 0.8
    GROUP_COLS = ['AvailabilityZone', 'InstanceType', 'ProductDescription']

    # Util
    OBJ_PATH = 'objs/'
