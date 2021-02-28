# สร้างโมเดลการเรียนรู้เชิงลึก
# นำเข้าไลบรารี
# รูปแบบการขับขี่อัตโนมัติแบบจำลองการขับขี่บนถนนจริง
import keras
import tensorflow
import sys
import os
import h5py
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.models import load_model, Model, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import Adam, SGD

np.random.seed(0)

# ตัวแปรส่วนกลาง
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 120, 160, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)



# step1,โหลดข้อมูลและแยกเป็นชุดการTrainingและการตรวจสอบความถูกต้อง
# ปัญหาชุดข้อมูลมีขนาดใหญ่เกินไปและเกินหน่วยความจำคอมพิวเตอร์
def load_data():
    # load
    image_array = np.zeros((1, 120, 160, 3))               # 初始化
    label_array = np.zeros((1, 5), 'float')
    training_data = glob.glob('training_data_npz/*.npz')
    # จับคู่ไฟล์ที่มีสิทธิ์ทั้งหมดและส่งคืนในรูปแบบรายการ
    print("จับคู่เสร็จสมบูรณ์ เริ่มRun")
    print("รวม% d รอบ", len(training_data))

    # if no data, exit，
    if not training_data:
        print("No training data in directory, exit")
        sys.exit()
    i = 0
    for single_npz in training_data:
        with np.load(single_npz) as data:
            print(data.keys())
            i = i + 1
            print("พิมพ์ค่าคีย์", i)
            train_temp = data['train_imgs']
            train_labels_temp = data['train_labels']
        image_array = np.vstack((image_array, train_temp)) # ใส่ไฟล์ทั้งหมดที่อ่านลงในหน่วยความจำ
        label_array = np.vstack((label_array, train_labels_temp))
        print("รอบเสร็จสิ้น% d", i)
    print("การวนซ้ำสิ้นสุดลงแล้ว")
    X = image_array[1:, :]
    y = label_array[1:, :]
    print('Image array shape: ' + str(X.shape))
    print('Label array shape: ' + str(y.shape))
    print(np.mean(X))
    print(np.var(X))

    # now we can split the data into a training (80), testing(20), and validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

    return X_train, X_valid, y_train, y_valid


# step2 การสร้างแบบจำลอง
def build_model(keep_prob):
    print("เริ่มรวบรวมโมเดล")
    model = Sequential()
    model.add(Lambda(lambda x: (x/102.83 - 1), input_shape = INPUT_SHAPE))
    model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(64, (3, 3),activation='elu'))
    model.add(Conv2D(64, (3, 3),activation='elu'))
    model.add(Dropout(keep_prob))  # Dropout เปอร์เซ็นต์การเชื่อมต่อของเซลล์ประสาทอินพุตจะถูกตัดการเชื่อมต่อแบบสุ่มทุกครั้งที่มีการอัปเดตพารามิเตอร์ในระหว่างกระบวนการฝึกอบรม
    model.add(Flatten())
    #model.add(Dense(500, activation='elu'))
    model.add(Dense(250, activation='elu'))
    #model.add(Dense(50, activation='elu'))
    model.add(Dense(5, activation='softmax'))
    model.summary()

    return model

# step3 รูปแบบการtraining
def train_model(model, learning_rate, nb_epoch, samples_per_epoch,
                batch_size, X_train, X_valid, y_train, y_valid):
    # โมเดวที่ดีที่สุดจะถูกsave
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='min')
    # EarlyStopping อดทน: เมื่อ earlystop ถูกเปิดใช้งาน（หากพบว่าการสูญเสียไม่ได้ลดลงเมื่อเทียบกับการฝึกอบรมยุคก่อนหน้านี้），
    # จากนั้นหยุดtrainingหลังจาก patience หนึ่ง epoch
    # mode：‘auto’，‘min’，‘max’หนึ่ง，ในโหมด min หากค่าการตรวจจับหยุดลดลงการฝึกจะถูกยกเลิก。ในโหมด max การฝึกจะหยุดลงเมื่อค่าการตรวจจับไม่เพิ่มขึ้นอีกต่อไป
    early_stop = EarlyStopping(monitor='loss', min_delta=.0005, patience=10,
                               verbose=1, mode='min')
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=20, write_graph=True,write_grads=True,
                              write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                              embeddings_metadata=None)
    # รวบรวมแบบจำลองเครือข่ายประสาทเทียมฟังก์ชันการสูญเสีย loss เครื่องมือเพิ่มประสิทธิภาพ optimizer รายการ metrics ซึ่งมีเมตริกสำหรับการประเมินประสิทธิภาพเครือข่ายของโมเดลระหว่างการฝึกอบรมและการทดสอบ
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    # การฝึกอบรมแบบจำลองเครือข่ายประสาทเทียม，batch_size จำนวนตัวอย่างที่มีอยู่ในแต่ละชุดระหว่างการไล่ระดับสี，epochs การฝึกอบรมสิ้นสุดลงกี่รอบ
    # verbose แสดงข้อมูลบันทึกหรือไม่，validation_data ชุดข้อมูลที่ใช้ในการตรวจสอบ
    model.fit_generator(batch_generator(X_train, y_train, batch_size),
                        steps_per_epoch=samples_per_epoch/batch_size,
                        epochs = nb_epoch,
                        max_queue_size=1,
                        validation_data=batch_generator(X_valid, y_valid, batch_size),
                        validation_steps=len(X_valid)/batch_size,
                        callbacks=[tensorboard, checkpoint, early_stop],
                        verbose=2)

# step4
# สามารถbatchกับbatchสำหรับการtraining CPU และ GPU จะเริ่มทำงานพร้อมกัน
def batch_generator(X, y, batch_size):
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty([batch_size, 5])
    while True:
        i = 0
        for index in np.random.permutation(X.shape[0]):
            images[i] = X[index]
            steers[i] = y[index]
            i += 1
            if i == batch_size:
                break
        yield (images, steers)


# step5 แบบประเมิน
#def evaluate(x_test, y_test):
    #score = model.evaluate(x_test, y_test, verbose=0)
    #print('Test loss:', score[0])
    #print('Test accuracy:', score[1])


def main():
    # พิมพ์ไฮเปอร์พารามิเตอร์

    print('-'*30)
    print('parameters')
    print('-'*30)


    keep_prob = 0.5
    learning_rate = 0.0001
    nb_epoch = 100
    samples_per_epoch = 3000
    batch_size = 30

    print('keep_prob = ', keep_prob)
    print('learning_rate = ', learning_rate)
    print('nb_epoch = ', nb_epoch)
    print('samples_per_epoch = ', samples_per_epoch)
    print('batch_size = ', batch_size)
    print('-' * 30)

    # เริ่มโหลดข้อมูล
    data = load_data()
    print("โหลดข้อมูลแล้ว")
    # รวบรวมโมเดล
    model = build_model(keep_prob)
    # ฝึกโมเดลบนชุดข้อมูลและบันทึกเป็น model.h5
    train_model(model, learning_rate, nb_epoch, samples_per_epoch, batch_size, *data)
    print("แบบได้รับการฝึกฝน")


if __name__ == '__main__':
    main()






