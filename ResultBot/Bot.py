import telebot
import matplotlib.pyplot as plt

TOKEN = '1903174383:AAGCQ6-k8fF-IY7cBjALqjHFFnjSQ9f6Wxw'


class ResultBot:

    def __init__(self,
                 users=None,
                 project_name="",
                 type_=""):
        """

        :param users: Массив ids в телеграмме
        :param project_name: Название проекта
        :param type_: Тип исследование
        """
        if users is None:
            users = [844253826]
        self.bot = telebot.TeleBot(TOKEN)
        self.users = users
        self.project_name = project_name
        self.params = {}
        self.type_ = type_

    def clear_params(self):
        self.params = {}

    def del_params(self, key):
        if key in self.params:
            del self.params[key]

    def send_message(self, message=""):
        str_message = ""
        if self.project_name != "":
            str_message += "Проект: {0}\n".format(self.project_name)
        if self.type_ != "":
            str_message += "Тип исследование: {0}\n".format(self.type_)
        if len(self.params.values()):
            str_message += "Параметры:\n"
        for key, item in self.params.items():
            str_message += "{0}: {1}\n".format(key, item)
        str_message += message
        for user in self.users:
            self.bot.send_message(chat_id=user,
                                  text=str_message)

    def send_plot(self, plot):
        name = "plot.jpg"
        plot.savefig(name)
        for user in self.users:
            self.bot.send_photo(chat_id=user,
                                photo=open(name, 'rb'))
