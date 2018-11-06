# -*- coding: utf-8 -*-
"""
ConvLSTM2D を使った機械学習モデルベースのオートパイロットクラス ConvLSTM2DPilot を提供します。
モデルのインプットとなるデータ構造に変換するユーティリティ関数 get_current_img_arrays を提供します。
"""
from tensorflow.python.keras.layers import Input, BatchNormalization
from tensorflow.python.keras.models import Model, load_model, Sequential
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D, LSTM, ConvLSTM2D
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Cropping2D, Lambda
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.python.keras.layers import TimeDistributed as TD

from parts.sim import SimPilot
import donkeycar as dk
import numpy as np

class ConvLSTM2DPilot(SimPilot):
    '''
    ConvLSTM2D を使ったNNモデルを使って推論を行うDonkey Car V2 パーツクラス。
    独自モデルを使用する場合、Donkey SimulaterへAPI提供する`donkey sim`コマンド
    が使用できないため、メソッドsim()を提供します。
    '''
    def __init__(self, model=None, 
        image_w=160, image_h=120, image_d=3, seq_length=3, num_outputs=2, 
        *args, **kwargs):
        '''
        コンストラクタ、引数値をインスタンス変数化し、モデルを生成します。

        引数：
            model       モデルオブジェクト、指定しない場合新規作成
            image_w     イメージデータの幅
            image_h     イメージデータの高さ
            image_d     イメージデータの色要素数
            seq_length  過去何件のデータをインプットとして使用するか
            num_outputs アウトプットデータ数
            args        可変引数受け入れ
            kwargs      キー・バリュー指定による可変引数要素受け入れ
        '''

        # 可変引数はすべて親クラスで対処
        super().__init__(*args, **kwargs)

        # インスタンス変数の初期化        
        self.seq_length = seq_length
        self.image_d = image_d
        self.image_w = image_w
        self.image_h = image_h
        # 前回使用したインプット
        self.img_seq = []
        # イメージデータの型
        self.image_shape = (image_h, image_w, image_d)

        # モデルオブジェクトの初期化
        if model:
            # 引数で与えられた機械学習モデルを使用
            self.model = model
        else:
            # ConvLSTM2Dを使用した機械学習モデルを使用
            self.model = self.create_model()
        # 標準出力にモデル構造を出力したい場合はコメントアウト
        #print(self.model.summary())

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
        
        # 引数img_array、prev_img_arrayから、予測メソッドpredictの引数
        self.img_seq = get_current_img_arrays(self.seq_length, img_array, self.img_seq)

        # 1件データ作成
        x = np.reshape(self.img_seq, (1, self.seq_length) + self.image_shape)
        # 予測実行
        outputs = self.model.predict(x)

        # 結果の返却
        print('run outputs:')
        print(outputs)
        steering = dk.util.data.linear_unbin(outputs[0][0]) #TODOエラー！
        throttle = outputs[1]
        print('run throttle_out:' + str(throttle))
        return steering, throttle

    def train(self, train_gen, val_gen,
              saved_model_path, epochs=100, steps=100, train_split=0.8,
              verbose=1, min_delta=.0005, patience=5, use_early_stop=True): #, log_dir=None):
        """
        トレーニング処理を行います。

        引数：
            train_gen       トレーニングデータgeneratorオブジェクト
            val_gen         評価データgeneratorオブジェクト
            train_split     バッチデータのうちトレーニングに使用する比率
            verbose         ModelCheckpointクラスへ渡される(verboseモード)
            min_delta       EarlyStoppingクラスへ渡される(最小差分)
            patience        EarlyStoppingクラスへ渡される(何回しきい値内を繰り返したらExitするか)
            use_early_stop  EarlyStoppingをしようするかどうかの真偽値
        戻り値：
            hist            fit_generator戻り値
        """

        # checkpoint to save model after each epoch
        save_best = ModelCheckpoint(saved_model_path,
                                    monitor='val_loss',
                                    verbose=verbose,
                                    save_best_only=True,
                                    mode='min')
        
        # TensorBoard
        #tensor_board = TensorBoard(log_dir, histogram_freq=1)

        # stop training if the validation error stops improving.
        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=min_delta,
                                   patience=patience,
                                   verbose=verbose,
                                   mode='auto')

        callbacks_list = [save_best] #, tensor_board]

        if use_early_stop:
            callbacks_list.append(early_stop)

        hist = self.model.fit_generator(
            train_gen,
            steps_per_epoch=steps,
            epochs=epochs,
            verbose=1,
            validation_data=val_gen,
            callbacks=callbacks_list,
            validation_steps= steps*(1.0 - train_split))#steps * (1.0 - train_split) / train_split)
        return hist

    def create_model(self):
        '''
        ConvLSTM2Dを用いた機械学習モデルを構築します。

        戻り値：
            compile済みモデルオブジェクト
        '''
        img_seq_shape = (self.seq_length,) + self.image_shape
        img_in = Input(shape=img_seq_shape, name='img_in')
        x = img_in
        x = ConvLSTM2D(filters=24, kernel_size=(5, 5), strides=(2, 2),
                           padding='same', return_sequences=True)(x)
        x = BatchNormalization()(x)

        x = ConvLSTM2D(filters=32, kernel_size=(5, 5), strides=(2, 2),
                           padding='same', return_sequences=True)(x)
        x = BatchNormalization()(x)

        x = ConvLSTM2D(filters=64, kernel_size=(5, 5), strides=(2, 2),
                           padding='same', return_sequences=True)(x)
        x = BatchNormalization()(x)

        x = ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=(2, 2),
                           padding='same', return_sequences=True)(x)
        x = BatchNormalization()(x)

        x = ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                           padding='same', return_sequences=True)(x)
        x = BatchNormalization()(x)

        x = Flatten(name='flattened')(x)
        x = Dense(100, activation='relu')(x)

        x = Dropout(.1)(x)
        x = Dense(50, activation='relu')(x)
        x = Dropout(.1)(x)
        angle_out = Dense(15, activation='softmax', name='angle_out')(x)        

        #continous output of throttle
        throttle_out = Dense(1, activation='relu', name='throttle_out')(x)
        print('model throttle_out:' + str(throttle_out))
        model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])
        model.compile(optimizer='adam',
                      loss={'angle_out': 'categorical_crossentropy', 
                            'throttle_out': 'mean_absolute_error'},
                      loss_weights={'angle_out': 0.9, 'throttle_out': .001})
        return model

    def rgb2gray(self, rgb):
        '''
        イメージ配列rgbをグレースケール化して返却します。

        引数：
            rgb     イメージ配列
        戻り値：
            グレースケール化されたイメージ配列
        '''
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def get_current_img_arrays(seq_length, img_array, prev_img_arrays):
    '''
    ConvLSTM2DPilotモデルの入力値に変換します。
    
    引数：
        seq_length: シーケンス数
        img_array:  最新のイメージ配列
        prev_img_arrays:    過去シーケンス数分のイメージ配列リスト
    戻り値：
        最新のイメージ配列リスト
    '''
    # prev_img_arrays のチェック及び初期化
    if prev_img_arrays is None or type(prev_img_arrays) is not list:
        # 存在しないorリストではない場合、0ベクトルで埋める
        prev_img_arrays = []
        for _ in range(seq_length):
            prev_img_arrays.append(np.zeros(img_array.shape, dtype=int))
    if len(prev_img_arrays) < seq_length:
        # シーケンス数より小さい場合、古いデータ側に0ベクトルを埋めて補填
        cnt = seq_length - len(prev_img_arrays)
        for _ in range(cnt):
            prev_img_arrays.insert(0, np.zeros(img_array.shape, dtype=int))
    if len(prev_img_arrays) > seq_length:
        # リストの要素数がシーケンス数より多い場合、後半要素側シーケンス数だけ要素を残す
        prev_img_arrays = prev_img_arrays[len(prev_img_arrays) - seq_length:]
        
    # 先頭の要素を削除し、最後尾に最新のイメージ配列を追加
    current_img_arrays = prev_img_arrays[1:]
    current_img_arrays.append(img_array)

    # 最新のイメージ配列リストを返却
    return current_img_arrays