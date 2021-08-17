from ResultBot.Bot import ResultBot
import json
import os
from Head.Center import start

def start_train(data):
    windows, dir_, snippet_count, chat_id = data
    params = {
        "snippet_count": snippet_count,
        "size_subsequent": windows,
        "percent_test": 0.3,
        "dataset_name": dir_
    }
    dataset = "..\\Dataset\\" + dir_
    with open(dataset + '\\params.json', 'w') as outfile:
        json.dump(params, outfile)
    bot.bot.send_message(chat_id, "Запускаю обучение со следующими парамаетрами" +
                         str(params))
    start(dataset,bot)


bot = ResultBot(project_name="sanni")
bot.add_func(name="start_train",
             project="sanni",
             func=[start_train, {"windows": int,
                                 "dir": str,
                                 "snippet_count": int}])
bot.bot.polling(none_stop=True, interval=0)
