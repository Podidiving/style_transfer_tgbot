import torch
import copy
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import albumentations as A

import io
from PIL import Image
import numpy as np
import telebot

from constants import greeting
from style import process_style, get_model
from second_style import get_2_model, go_2_style

with open("token.txt") as f:
    token = f.readline().strip()

bot = telebot.TeleBot(token)
db = {}


def startup_detectron():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = device
    predictor = DefaultPredictor(cfg)
    return cfg, predictor


cfg, detectron = startup_detectron()
style_model = get_model()
style2 = get_2_model()


def handle_content(user_id):
    image = copy.deepcopy(db[user_id]["content"])
    outputs = detectron(image)
    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    print(outputs["instances"].pred_classes)
    where_car = np.where(outputs["instances"].pred_classes.cpu().numpy() == 2)[0]
    if len(where_car) > 1:
        bot.send_message(user_id, f"Найдено {len(where_car)} машин. Отправь другую")
        del db[user_id]
        return
    elif not len(where_car):
        bot.send_message(user_id, f"Не найдено машин. Отправь другую")
        del db[user_id]
        return

    db[user_id]["mask"] = outputs["instances"].to("cpu").pred_masks[where_car[0]].numpy()
    # bot.send_message(user_id, "отправляю только в test mode")
    # bot.send_photo(user_id, Image.fromarray(out.get_image()[:, :, ::-1]))


def handle_style(user_id, t):
    content = copy.deepcopy(db[user_id]["content"])
    if t == "bg":
        style = db[user_id]["style"]
        result = process_style(style_model, user_id, content, style)
        # bot.send_photo(user_id, result)
        content = db[user_id]["content"]
        resizer = A.Resize(*content.shape[:2])
        content_image = resizer(image=np.array(result))["image"]
        mask = db[user_id]["mask"]
        content_image[mask] = content[mask]
        db[user_id]["mask2"] = copy.deepcopy(content_image)
        # bot.send_message(user_id, "1й алгоритм")
        # bot.send_photo(user_id, Image.fromarray(content_image))

        # Image.fromarray(content_image).save(f"images/{np.random.randint(0, 100)}_{user_id}_image.jpg")
        # result = go_2_style(style2, copy.deepcopy(db[user_id]["content"]), copy.deepcopy(db[user_id]["style"]))
        # bot.send_message(user_id, "2й алгоритм")
        # bot.send_photo(user_id, result)
    else:
        style = db[user_id]["style2"]
        result = process_style(style_model, user_id, content, style)
        # bot.send_photo(user_id, result)
        content = db[user_id]["content"]
        resizer = A.Resize(*content.shape[:2])
        content_image = resizer(image=np.array(result))["image"]
        mask = db[user_id]["mask"]
        con = db[user_id]["mask2"]
        content_image[np.where(mask, False, True)] = con[np.where(mask, False, True)]
        bot.send_message(user_id, "1й алгоритм")
        bot.send_photo(user_id, Image.fromarray(content_image))
        result = go_2_style(
            style2,
            copy.deepcopy(db[user_id]["content"]),
            copy.deepcopy(db[user_id]["style"]),
            copy.deepcopy(db[user_id]["style2"]),
            mask=copy.deepcopy(db[user_id]["mask"]).astype(np.uint8)
        )
        bot.send_message(user_id, "2й алгоритм")
        bot.send_photo(user_id, result)
        # Image.fromarray(content_image).save(f"images/{np.random.randint(0, 100)}_{user_id}_image.jpg")


def handle(id_, type_):
    if id_ in db:
        flag1 = db[id_].get("content") is not None and isinstance(db[id_].get("content"), np.ndarray)
        flag2 = db[id_].get("style") is not None and isinstance(db[id_].get("style"), np.ndarray)
        if not flag1 or not flag2:
            bot.send_message(id_, "Не все условия соблюдены. Начинай заново")
            del db[id_]
            return
        bot.send_message(id_, f"Заглушка {type_}")
    else:
        bot.send_message(id_, greeting)


@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    if message.text.lower() == "/start":
        bot.send_message(message.from_user.id, greeting)
    elif message.from_user.id not in db:
        bot.send_message(message.from_user.id, "Не понимаю. Прочти приветственное письмо")
        bot.send_message(message.from_user.id, greeting)
    if message.text.lower() == "delete":
        try:
            del db[message.from_user.id]
        except KeyError:
            print(message.from_user.id)
            print("error (key)")
        print("deleted")
        print(db)


@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    try:
        if message.from_user.id not in db:
            bot.send_message(message.from_user.id, "Сохраняю картинку с машиной")
            file = bot.download_file(bot.get_file(message.photo[-1].file_id).file_path)
            image = Image.open(io.BytesIO(file))
            db[message.from_user.id] = {}
            db[message.from_user.id]["content"] = np.array(image)
            handle_content(message.from_user.id)
            bot.send_message(message.from_user.id, "Готово")
        else:
            if db[message.from_user.id].get("style") is not None:
                bot.send_message(message.from_user.id, "Сохраняю картинку со стилем для фона")
                file = bot.download_file(bot.get_file(message.photo[-1].file_id).file_path)
                image = Image.open(io.BytesIO(file))
                db[message.from_user.id]["style2"] = np.array(image)
                handle_style(message.from_user.id, "car")
                print(f"{message.from_user.id} done")
                bot.send_message(message.from_user.id, "Готово")
                del db[message.from_user.id]["style"]
                del db[message.from_user.id]["style2"]

            else:
                bot.send_message(message.from_user.id, "Сохраняю картинку со стилем для фона")
                file = bot.download_file(bot.get_file(message.photo[-1].file_id).file_path)
                image = Image.open(io.BytesIO(file))
                db[message.from_user.id]["style"] = np.array(image)
                handle_style(message.from_user.id, "bg")
                print(f"{message.from_user.id} done")
                bot.send_message(message.from_user.id, "Готово")
            # bot.send_message(message.from_user.id, "Осталось написать car или background")
    except Exception as e:
        bot.send_message(
            message.from_user.id,
            "Произошла ошибка. Не отправляйте новое фото, пока не увидите надпись 'готово'"
        )
        raise e


bot.polling(none_stop=True, interval=0)
