# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 17:48:47 2021

@author: LocalAdmin

---

Usage:
    from telegram_assistant import TelegramAssistant
    ta = TelegramAssistant(token='<bot token here>', chat_id=<your chat_id>)
    ta.send('check out this!')
"""

import telegram


class TelegramAssistant(object):
    token = ''
    chat_id = int()

    def __init__(self, token, chat_id):
        self.chat_id = chat_id
        self.token = token
        self.bot = telegram.Bot(token=token)

    def send(self, msg=str):
        """
        Sends a message to a string
        """
        self.bot.sendMessage(chat_id=self.chat_id, text=msg)

    def send_image(self, img_path=str):
        """
        Sends an image specifying the image's path
        """
        self.bot.sendPhoto(chat_id=self.chat_id, photo=open(img_path, 'rb'))

    def send_video(self, video_path=str):
        """
        Sends a video specifying the video's path
        """
        self.bot.sendVideo(chat_id=self.chat_id, video=open(video_path, 'rb'))

    def send_doc(self, doc_path=str):
        """
        Sends a document specifying the document's path
        """
        self.bot.sendDocument(chat_id=self.chat_id, doc=open(doc_path, 'rb'))
