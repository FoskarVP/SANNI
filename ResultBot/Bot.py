import telebot
from telebot import types
import ast
import matplotlib.pyplot as plt

TOKEN = '1903174383:AAGCQ6-k8fF-IY7cBjALqjHFFnjSQ9f6Wxw'


class ResultBot:

    def __init__(self,
                 users=None,
                 project_name="chtpz",
                 type_=""):
        """

        :param users: Массив ids в телеграмме
        :param project_name: Название проекта
        :param type_: Тип исследование
        """
        if users is None:
            users = [844253826, 1854811596]

        self.projects = {"chtpz": [], "sanni": []}
        self.func = {}
        self.bot = telebot.TeleBot(TOKEN)
        self.users = users
        self.project_name = project_name
        self.params = {}
        self.choice = ""
        self.type_ = type_
        self.insert_params = []

        @self.bot.message_handler(commands=['launch'])
        def start(message):
            mess = "Выбор проекта:"
            keyboard = types.InlineKeyboardMarkup()
            for i, item in enumerate(self.projects.keys()):
                key = types.InlineKeyboardButton(text=item,
                                                 callback_data=item)
                keyboard.add(key)
            self.bot.send_message(message.from_user.id, text=mess,
                                  reply_markup=keyboard)  # следующий шаг – функция get_name

        @self.bot.callback_query_handler(
            func=lambda call: call.data in self.projects and call.data == self.project_name)
        def callback_project(call):
            mess = "Выбор функции:"
            keyboard = types.InlineKeyboardMarkup()
            for i, item in enumerate(self.projects[call.data]):
                key = types.InlineKeyboardButton(text=item,
                                                 callback_data=item)
                keyboard.add(key)
            self.bot.send_message(call.message.chat.id, text=mess, reply_markup=keyboard)

        @self.bot.callback_query_handler(func=lambda call: call.data in self.func)
        def callback_project(call):
            self.choice = call.data
            keys = list(self.func[self.choice][1])
            send = self.bot.send_message(call.message.chat.id, keys[0])

            self.bot.register_next_step_handler(send, self.verify_page)

    def clear_params(self):
        self.params = {}

    def del_params(self, key):
        if key in self.params:
            del self.params[key]

    def send_message(self, message=""):
        str_message = ""
        if self.project_name != "":
            str_message += "Проект: #{0}\n".format(self.project_name)
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

    def verify_page(self, message):

        keys = list(self.func[self.choice][1])
        try:
            func = self.func[self.choice][1][keys[len(self.insert_params)]]
            self.insert_params.append(func(message.text))

        except ValueError as e:

            self.bot.send_message(message.chat.id, "Ошибка типа")

        """ if message.text.isalpha():
                self.insert_params.append(message.text)
                print(type(self.insert_params[len(self.insert_params) - 1]))
        else:
                self.insert_params.append(ast.literal_eval(message.text))
                print(type(self.insert_params[len(self.insert_params) - 1]))
        """

        if len(self.insert_params) < len(self.func[self.choice][1]):
            self.request_page(message)
        else:
            print("Вызов функции")
            func = self.func[self.choice][0]
            self.insert_params.append(message.chat.id)
            insert_params = self.insert_params
            self.insert_params = []
            func(insert_params)
            """try:
                func(insert_params)
            except ValueError as e:
                print(e)
                self.send_message(str(e))"""

    def request_page(self, message):
        keys = list(self.func[self.choice][1])
        print(keys)
        send = self.bot.send_message(message.chat.id, keys[len(self.insert_params)])
        self.bot.register_next_step_handler(send, self.verify_page)

    def get(self, data):
        window, count, chat_id = data
        self.bot.send_message(chat_id, str((window, count)))

    def add_func(self, name, project, func):
        if project not in self.projects:
            self.projects[project] = []
        if name not in self.projects[project]:
            self.projects[project].append(name)
        if name not in self.func:
            self.func[name] = func
