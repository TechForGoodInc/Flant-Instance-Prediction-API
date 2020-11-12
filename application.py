import logging
from flask import Flask, request
from datetime import datetime
from modelmanager.ModelManager import ModelManager
from modelmanager.enums import MMStatus

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
mm = ModelManager()


# Return state of model manager
@app.route('/getState')
def get_state():
    return {
        'name': mm.state.name,
        'value': mm.state.value
    }


# Train models with data from unprocessed data files
@app.route('/processUnprocessed')
def process_unprocessed():
    # TODO: Add async handling
    status = mm.process_unprocessed()
    return '',  status.get_http_code()


# Get predicted prices for a given instance during specified time period
@app.route('/predict/<instance>')
def predict(instance):
    start_date_str = request.args.get('start')
    end_date_str = request.args.get('end')

    try:
        start_date = datetime.strptime(start_date_str, "%y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%y-%m-%d")
    except (ValueError, TypeError):
        return 'Invalid Parameters', 400

    results, status = mm.get_predictions(instance, start_date, end_date)
    return results.to_json(), status.get_http_code()


if __name__ == '__main__':
    app.run()
