import tensorflow as tf
import numpy as np
import PIL.Image as Image
import cv2
from tensorflow.keras import Input,Model
from tensorflow.keras.layers import Conv2D,Conv2DTranspose,MaxPooling2D,Concatenate
import tensorboard
import os
import random
import rawpy
import pandas as pd

train_dir = 'dataset/Sony/sony/long/'
test_dir = 'dataset/Sony/sony/short/'
BATCH_SIZE=1
EPOCH=100

class Network:

    def __init__(self):
        self.SIZE = (512, 512)
        self.CHANNELS = 4
        self.crops_per_image = 8
        self.k1=1
        self.k2=1
        self.model=None
        self.lr=1e-3
        self.batch_size=1
        self.losses=[]
    def my_model(self):
        inp = Input(shape=[self.SIZE[0] / 2, self.SIZE[1] / 2, 4])
        out1 = Conv2D(32, [3, 3], activation='relu', padding='same')(inp)
        out2 = Conv2D(32, [3, 3], activation='relu', padding='same')(out1)
        out3 = MaxPooling2D([2, 2], [2, 2], padding='same')(out2)

        out4 = Conv2D(64, [3, 3], activation='relu', padding='same')(out3)
        out5 = Conv2D(64, [3, 3], activation='relu', padding='same')(out4)
        out6 = MaxPooling2D([2, 2], [2, 2], padding='same')(out5)

        out7 = Conv2D(128, [3, 3], activation='relu', padding='same')(out6)
        out8 = Conv2D(128, [3, 3], activation='relu', padding='same')(out7)
        out9 = MaxPooling2D([2, 2], [2, 2], padding='same')(out8)

        out10 = Conv2D(256, [3, 3], activation='relu', padding='same')(out9)
        out11 = Conv2D(256, [3, 3], activation='relu', padding='same')(out10)
        out12 = MaxPooling2D([2, 2], [2, 2], padding='same')(out11)

        out13 = Conv2D(512, [3, 3], activation='relu', padding='same')(out12)
        out14 = Conv2D(512, [3, 3], activation='relu', padding='same')(out13)

        out15 = Conv2DTranspose(256, [2, 2], [2, 2], padding='same')(out14)
        out16 = Concatenate()([out11, out15])

        out17 = Conv2D(256, [3, 3], activation='relu', padding='same')(out16)
        out18 = Conv2D(256, [3, 3], activation='relu', padding='same')(out17)

        out19 = Conv2DTranspose(128, [2, 2], [2, 2], padding='same')(out18)
        out20 = Concatenate()([out8, out19])

        out21 = Conv2D(128, [3, 3], activation='relu', padding='same')(out20)
        out22 = Conv2D(128, [3, 3], activation='relu', padding='same')(out21)

        out23 = Conv2DTranspose(64, [2, 2], [2, 2], padding='same')(out22)
        out24 = Concatenate()([out5, out23])

        out25 = Conv2D(64, [3, 3], activation='relu', padding='same')(out24)
        out26 = Conv2D(64, [3, 3], activation='relu', padding='same')(out25)

        out27 = Conv2DTranspose(32, [2, 2], [2, 2], padding='same')(out26)
        out28 = Concatenate()([out2, out27])

        out29 = Conv2D(32, [3, 3], activation='relu', padding='same')(out28)
        #out30 = Conv2D(4, [3, 3], activation='relu', padding='same')(out29)

        #out31 = Conv2DTranspose(4, [2, 2], [2, 2], padding='same')(out30)

        out30 = Conv2D(16, [3, 3], activation='relu', padding='same')(out29)
        out31 = Conv2D(16, [3, 3], activation='relu', padding='same')(out30)
        out32 = Conv2D(3, [1, 1], activation='sigmoid', padding='same')(out31)
        #ans = tf.depth_to_space(out31, 2)
        self.model = Model(inputs=inp, outputs=out32)
        return self.model

    def loss(self,y_true,y_pred):

        im1 = tf.image.convert_image_dtype(y_true, tf.float32)
        im2 = tf.image.convert_image_dtype(y_pred, tf.float32)

        # psnr loss
        loss1 = tf.image.psnr(im1, im2, max_val=1.0)

        # ssim loss
        loss2 = tf.image.ssim(im1, im2, max_val=1.0)

        # canny edge detection
        
        edges1=tf.image.sobel_edges(y_true)
        edges2=tf.image.sobel_edges(y_pred)
        loss3 = tf.reduce_mean(tf.keras.losses.mean_squared_error(edges1, edges2))

        loss = tf.math.add(tf.math.subtract(tf.math.scalar_mul(self.k1, loss1),
                                            tf.math.scalar_mul(self.k2, loss2)),loss3)
        self.losses.append(loss)
        return loss


    #def image_normalization(self,image):
     #   return image / 255

    def randomcrop(self,img1,img2,size=[512,512]):
        assert img1.shape[0] >= size[0]
        assert img1.shape[1] >= size[1]
        assert img2.shape[0] >= size[0]
        assert img2.shape[1] >= size[1]
        x = random.randint(0, img1.shape[1] - size[0])
        y = random.randint(0, img1.shape[0] - size[1])

      
   img1 = img1[y:y + size[0], x:x + size[1]]
        img2 = img2[y:y + size[0], x:x + size[1]]
        return img1,img2

    def load_train_data(self):
        train = []
        for path in os.listdir(train_dir):
            image = cv2.imread(train_dir + path)
            for _ in range(self.crops_per_image):
                img = self.randomcrop(image, self.SIZE)
                #img = self.image_normalization(img)
                train.append(np.array(img))
                del img
            del image
        return np.array(train)

    def load_test_data(self):
        test = []
        for path in os.listdir(test_dir):
            image = cv2.imread(test_dir + path)
            for _ in range(self.crops_per_image):
                img = self.randomcrop(image, self.SIZE)
                #img = self.image_normalization(img)
                test.append(np.array(img))
                del img
            del image
        return np.array(test)


    def summary(self):
        print(self.model.summary())

    def convert(self,X):
        #i = raw.raw_image_visible.astype(np.float32)
        #i = np.maximum(im - 512, 0) / (16383 - 512)
        H = X.shape[0]
        W = X.shape[1]
        temp = np.empty(( H // 2, W // 2, 4))
        temp[:,:,0] = X[::2, ::2]
        temp[:,:,1] = X[::2, 1::2]
        temp[:,:,2] = X[1::2, 1::2]
        temp[:,:,3] = X[1::2, ::2]
        #X_train = X_train.reshape((H / 2, W / 2, 4))
        return temp

    def summary(self):
        print(self.model.summary())

    def model_compile(self):
        optimizer = tf.keras.optimizers.Adam(lr=self.lr, beta_1= 0.9,
                                             beta_2= 0.999, epsilon= None,
                                             decay= 0.0, amsgrad= False)

        self.model.compile(loss=self.loss,optimizer=optimizer,metrics=['accuracy'])

    def train(self,train_image,test_image):

        self.model.train_on_batch(train_image,test_image)

    def save_model(self):
        self.model.save('model.h5')

mod=Network()
mod.my_model()
mod.model_compile()
mod.summary()

train=pd.read_csv('train.csv')
z=0
for ep in range(EPOCH):
    print('Epoch {}/{}:'.format(ep+1,EPOCH))
    for row in range(train.shape[0]):
        print(row)
        inp=train.iat[row,0]
        out=train.iat[row,1]
        print(inp)
        if inp[-8:-5]=='0.04':
            amp_ratio=250
        else: amp_ratio=100
        try:
            in_img = rawpy.imread(inp).raw_image_visible.astype(np.float32)

            out_img = rawpy.imread(out).postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16).astype(np.float32)
        except:
            continue
        #in_img1 = np.maximum(in_img - 512, 0) / (16383 - 512)

        in_img1,out_img1=mod.randomcrop(in_img,out_img)
        #print(out_img1.shape)
        in_image=mod.convert(in_img1/65535)*amp_ratio
        out_image=out_img1/65535
        out_image=mod.convert(out_image)
        #print(out_img1.shape)
        #print(row)
        #print(mod.losses)
        shape=[1,256,256,4]
        a=np.reshape(in_image,shape)
        b=np.reshape(out_image,shape)
        mod.train(a,b)
        #print(z,'trained')
        z+=1
        del in_img,out_img,in_img1,out_img1,in_image,out_image,a,b
