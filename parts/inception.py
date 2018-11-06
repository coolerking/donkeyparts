# -*- coding: utf-8 -*-
"""
imagenetデータで学習済みのInceptionV3モデルを使ったオートパイロットクラスを提供します。
"""

# keras
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Flatten


from parts.sim import SimPilot
import donkeycar as dk
import numpy as np

class InceptionPilot(SimPilot):
    '''
    オートパイロットクラス。
    入力データ、出力データがKerasCategoricalと同じであるため
    継承して作成しています。
    '''

    def __init__(self, model=None, *args, **kwargs):
        '''
        入力データ型、モデルをインスタンス変数へ格納します。
        
        引数：
            *args      可変引数
            **kwargs    キーバリュー型可変引数
        '''

        #親クラス実装を実行
        super().__init__(*args, **kwargs)

        # 入力データの型
        self.image_shape = (120, 160, 3)

        # モデルオブジェクト
        if model:
            self.model = model
        else:
            self.model = self.create_model()

    def run(self, img_array):
        '''
        Vehicleへ追加可能なパーツとして必要なrun関数。
        メモリより'cam/image_array'を取得し、
        ステアリング値、スロットル値、最新の 'cam/prev_image_arrays' 値を返却します。

        引数：
            img_array   最新のイメージ配列
        戻り値
            steering    ステアリングPWM値
            throttle    スロットルPWM値
        '''
        img_arr = img_array.reshape((1,) + img_array.shape)
        angle_binned, throttle = self.model.predict(img_arr)
        angle_unbinned = dk.util.data.linear_unbin(angle_binned[0])
        print('angle_pwm:' + str(angle_unbinned) + ', throttle_pwm:' + str(throttle[0][0]))
        return angle_unbinned, throttle[0][0]

    def create_model(self):
        '''
        InceptionV3を用いた機械学習モデルを構築します。

        戻り値：
            compile済みモデルオブジェクト
        '''

        # inputの定義：cam/img_array 形式
        img_in = Input(shape=self.image_shape, name='img_in')

        # InceptionV3 canned model を使用
        x = InceptionV3(weights='imagenet', include_top=False)(img_in)
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='softmax')(x)
        x = Dropout(0.5)(x)
 
        # 従来モデルより流用
        x = Flatten(name='flattened')(x)
        x = Dense(100, activation='relu')(x)

        x = Dropout(.1)(x)
        x = Dense(50, activation='relu')(x)
        x = Dropout(.1)(x)
        # outputs[0] の定義：15分類確率ベクトル
        angle_out = Dense(15, activation='softmax', name='angle_out')(x)        

        # outputs[1] の定義：スロットルPWM値
        throttle_out = Dense(1, activation='relu', name='throttle_out')(x)
        print('model throttle_out:' + str(throttle_out))

        # モデルオブジェクトの定義
        model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])
        model.compile(optimizer='adam',
                      loss={'angle_out': 'categorical_crossentropy', 
                            'throttle_out': 'mean_absolute_error'},
                      loss_weights={'angle_out': 0.9, 'throttle_out': .001})
        return model