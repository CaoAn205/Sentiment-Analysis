import requests

from config import TELEGRAM_SEND_MESSAGE_URL
from data_sentiment import Sentiment_01, cv, clean_text_1, contraction_remover

class TelegramBot:

    def __init__(self):
        self.chat_id = None
        self.text = None
        
    def parse_webhook_data(self, data):
        message = data['message']

        self.chat_id = message['chat']['id']
        self.incoming_message_text = message['text'].lower()

    def action(self):
        success = None

        if self.incoming_message_text == "/start":
            self.outgoing_message_text = "Welcome! I'm a Sentiment Bot. Let me guess your emotion throughout a small text :) \nText me something! "
        else:
            self.outgoing_message_text = 'I think you are quite {}. Stay positive!'.format(Sentiment_01.predict(cv.transform([contraction_remover(clean_text_1(self.incoming_message_text))]))[0].lower())

        success = self.send_message()

        return success

    def send_message(self):
        res = requests.get(TELEGRAM_SEND_MESSAGE_URL.format(self.chat_id, self.outgoing_message_text))

        return True if res.status_code == 200 else False

    @staticmethod
    def init_webhook(url):
        requests.get(url)

