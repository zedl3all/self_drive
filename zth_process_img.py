# ประมวลผลรูปภาพเป็นรูปแบบ npz
# รูปแบบ self driving
import os
import numpy as np
import matplotlib.image as mpimg
from time import time
import math
from PIL import Image

CHUNK_SIZE = 128    #บีบอัดรูปภาพและประมวลผลทุกๆ 256



# ย่อหน้านี้แตกต่างกัน
def process_img(img_path, key):

    print(img_path, key)
    image = Image.open(img_path)
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)  # เพิ่มมิติ


    #image_array = mpimg.imread(img_path)
    #image_array = np.expand_dims(image_array, axis=0)

    print(image_array.shape)

    if key == 2:
        label_array = [0., 0., 1., 0., 0.]
    elif key == 3:
        label_array = [0., 0., 0., 1., 0.]
    elif key == 0:
        label_array = [1., 0., 0., 0., 0.]
    elif key == 1:
        label_array = [0., 1., 0., 0., 0.]
    elif key == 4:
        label_array = [0., 0., 0., 0., 1.]

    return (image_array, label_array)
    # ส่งคืนข้อมูลรูปภาพ (เมทริกซ์) และค่าป้ายกำกับที่เกี่ยวข้อง


if __name__ == '__main__':
    path = "training_data"
    files = os.listdir(path)                             # บันทึกชื่อไฟล์ภายใต้เส้นทางนี้ไปยังรายการ
    turns = int(math.ceil(len(files) / CHUNK_SIZE))      # ปัดเศษแบ่งรูปภาพทั้งหมดออกเป็นหลาย ๆ รอบหนึ่งรอบต่อ CHUNK_SIZE
    print("number of files: {}".format(len(files)))
    print("turns: {}".format(turns))

    for turn in range(0, turns):
        train_labels = np.zeros((1, 5), 'float')           # เริ่มต้นอาร์เรย์เลเบล
        train_imgs = np.zeros([1, 120, 160, 3])            # เริ่มต้นอาร์เรย์รูปภาพ
        CHUNK_files = files[turn * CHUNK_SIZE: (turn + 1) * CHUNK_SIZE] # ถ่ายภาพรอบปัจจุบัน
        print("number of CHUNK files: {}".format(len(CHUNK_files)))
        for file in CHUNK_files:
            # ไม่ใช่โฟลเดอร์และไฟล์ jpg
            if not os.path.isdir(file) and file[len(file) - 3:len(file)] == 'jpg':
                try:
                    key = int(file[0])                     # ใช้อักขระตัวแรกเป็นคีย์
                    image_array, label_array = process_img(path + "/" + file, key)
                    train_imgs = np.vstack((train_imgs, image_array))
                    train_labels = np.vstack((train_labels, label_array))
                except:
                    print('prcess error')

        # ลบอาร์เรย์รูปภาพศูนย์ทั้งหมดที่บิต 0 อาร์เรย์รูปภาพที่เป็นศูนย์ทั้งหมดคือ train_imgs = np.zeros([1,120,160,3]) สร้างขึ้นในตอนแรก
        train_imgs = train_imgs[1:, :]
        train_labels = train_labels[1:, :]
        file_name = str(int(time()))
        directory = "training_data_npz"

        if not os.path.exists(directory):
            os.makedirs(directory)
        try:
            np.savez(directory + '/' + file_name + '.npz', train_imgs=train_imgs, train_labels=train_labels)
        except IOError as e:
            print(e)


