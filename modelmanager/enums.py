from enum import Enum, auto


class MMState(Enum):
    FREE = auto()
    PROCESSING = auto()
    TRAINING = auto()
    PREDICTING = auto()


class MMStatus(Enum):

    # Return status of Model Manager methods
    OK = auto()
    NO_MODEL = auto()
    NO_PREDICTIONS = auto()

    # Map of status to http codes
    http_codes = {
        OK: 200,
        NO_MODEL: 204,
        NO_PREDICTIONS: 204
    }

    messages = {
        OK: 'OK',
        NO_MODEL: 'No model found for specified instance',
        NO_PREDICTIONS: 'No prediction file for model'
    }

    def get_http_code(self):
        return self.http_codes.value[self.value]

    def get_message(self):
        return self.messages.value[self.value]
