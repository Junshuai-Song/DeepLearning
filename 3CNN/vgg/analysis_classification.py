# Auther: Alan
from PIL import Image
import numpy as np


def test1():
    input_path = "./laska.png"
    maxsize = (32, 32)

    # box = (0, 0, 28, 28)

    lena = Image.open(input_path)
    # print(type(lena))
    lena.thumbnail(maxsize, Image.ANTIALIAS)
    lena.save("temp.jpg", "JPEG")
    # lena = Image.open("temp.jpg").crop(box)

    minsize = (224, 224)
    input_path = "./temp.jpg"
    lena = Image.open(input_path)
    # print(type(lena))
    # lena.thumbnail(minsize, Image.ANTIALIAS)
    lena.resize(minsize, Image.ANTIALIAS).save("temp.png")
    # lena.save("temp.png", "PNG")


def test2():
    # input_path = "./temp.jpg"
    # lena = Image.open(input_path)
    # print((np.array(lena)).shape)
    # print(np.array(lena))
    # batch = np.reshape(lena, [-1])
    # print(batch[0:10])
    #
    # ans = []
    # for j in range(3):
    #     for k in range(7):
    #         # print(j, k, len(ans))
    #         for i in range(32):
    #             t = batch[i * 32 + j * 1024:(i + 1) * 32 + j * 1024]
    #             t = list(t)
    #             ans.extend(t * 7)  # 扩充7倍刚好

    new_img = Image.new('RGB', (224, 224), 255)
    x = 0
    for i in range(7):
        y = 0
        for j in range(7):
            img = Image.open("./temp.jpg")
            width, height = img.size
            new_img.paste(img, (x, y))
            y += height
        x += 32

    # lena = Image.fromarray(np.reshape(ans, [224,224,3]))
    new_img.save("test2.png")


if __name__ == "__main__":
    test2()

