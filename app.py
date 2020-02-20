# local
import age_prediction
import s3_helper

from PIL import Image
import os
from json import dumps, loads
from kafka import KafkaProducer
from kafka import KafkaConsumer

# log
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Get Environment variables
KAFKA_HOST = os.environ['KAFKA_HOST']
KAFKA_PORT = os.environ['KAFKA_PORT']
KAFKA_TOPIC_FACE_IMAGE = os.environ['KAFKA_TOPIC_FACE_IMAGE']
KAFKA_TOPIC_AGE_RESULT = os.environ['KAFKA_TOPIC_AGE_RESULT']

# display environment variable
logger.info('KAFKA_HOST: {}'.format(KAFKA_HOST))
logger.info('KAFKA_PORT: {}'.format(KAFKA_PORT))
logger.info('KAFKA_TOPIC_FACE_IMAGE: {}'.format(KAFKA_TOPIC_FACE_IMAGE))
logger.info('KAFKA_TOPIC_AGE_RESULT: {}'.format(KAFKA_TOPIC_AGE_RESULT))


def main():
    # TODO: Kafka Config
    consumer = KafkaConsumer(KAFKA_TOPIC_FACE_IMAGE,
                             bootstrap_servers=[
                                 '{}:{}'.format(KAFKA_HOST, KAFKA_PORT)],
                             auto_offset_reset='earliest',
                             enable_auto_commit=True,
                             group_id='age-prediction-group')

    producer = KafkaProducer(
        bootstrap_servers=['{}:{}'.format(KAFKA_HOST, KAFKA_PORT)])

    logger.info('Ready for consuming messages')
    for message in consumer:
        # de-serialize
        input_json = loads(message.value.decode('utf-8'))
        logger.info('Input JSON: {}'.format(dumps(input_json, indent=2)))

        # Get image from S3
        img_stream = s3_helper.get_file_stream_s3(
            input_json['face_image_path'])

        # Reference position
        ref_position = None
        if input_json['position_top'] and input_json['position_right'] and input_json['position_bottom'] and input_json['position_left']:
            ref_position = {
                'position_top': input_json['position_top'],
                'position_right': input_json['position_right'],
                'position_bottom': input_json['position_bottom'],
                'position_left': input_json['position_left']}

        # inference
        prediction = age_prediction.predict(
            img_stream, ref_position=ref_position)

        # Response
        age_result = {'face_image_id': input_json['face_image_id'],
                      'min_age': prediction['min_age'],
                      'max_age': prediction['max_age'],
                      'confidence': prediction['confidence'],
                      'position_top': prediction['position_top'],
                      'position_right': prediction['position_right'],
                      'position_bottom': prediction['position_bottom'],
                      'position_left': prediction['position_left'],
                      'time': prediction['time']}
        logger.info('Age Result JSON: {}'.format(
            dumps(age_result, indent=2)))

        # Send to Kafka
        producer.send(KAFKA_TOPIC_AGE_RESULT,
                      value=dumps(age_result).encode('utf-8'))

if __name__ == '__main__':
    main()
