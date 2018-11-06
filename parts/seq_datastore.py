# -*- coding: utf-8 -*-
"""
seq_length 件をまとめてインプットとするgeneratorに書き換えるため
TubGroupの必要最小限のメソッドをオーバライドしたクラスSeqTubGroupを
作成した。
ConvLSTM2Pilotを使用する場合はTubGroupのかわりにSeqTubGroupに
差し替えてトレーニングを行う。

seq_length: 時系列長
X: (seq_length, 120,160,3) 画像データ
"""

import numpy as np
import pandas as pd
import donkeycar as dk

#from pilot import get_current_img_arrays

class SeqTubGroup(dk.parts.datastore.TubGroup):
    '''
    TubGroupを継承して、必要最小限のメソッドをオーバライドすることで、
    レコード辞書の'cam/image_array'要素をシーケンス数分の過去データ
    を含む配列に差し替えている。

    引数：
        tub_paths_arg   TubGroupへ渡す
        seq_length      時系列帳、デフォルトは3
    '''

    def __init__(self, tub_paths_arg, seq_length=3):
        '''
        コンストラクタをオーバライドして、self.seq_lengthを初期化する。
        引数seq_lengthを指定しない場合は3となる。
        '''
        print(tub_paths_arg)
        super().__init__(tub_paths_arg)
        # シーケンス長
        self.seq_length = seq_length
        # 前回データとして使用した 'cam/image_array' 要素
        # 初期値として空のリストを格納
        self.seq_img_arrays = []
        print('seq_length = ' + str(self.seq_length))
    
    def read_record(self, record_dict):
        '''
        最新のレコード辞書を読み込む。その際に、self.seq_length 件のイメージ配列を
        入力値とするために、親クラス側の同名メソッドを実行し、
        最新レコード辞書のイメージ配列格納領域に過去 self.seq_length 分のイメージ配列の
        リストを作成・格納して返却する。

        引数：
            record_dict     レコード辞書
        
        戻り値：
            new_record_dict 最新のレコード辞書
        '''
        # 新規レコード辞書を読み込み
        new_record_dict = super().read_record(record_dict)

        # self.seq_length が1未満場合、シーケンス処理ではない→親クラス実装結果を
        # そのまま返却
        if self.seq_length < 1:
            return new_record_dict
        
        # 最新レコード辞書からイメージ配列を取得
        new_img_array = new_record_dict['cam/image_array']

        # 最新レコード辞書のイメージ配列を格納していた領域に、作成したイメージ配列リストを格納
        self.seq_img_arrays = pilot.get_current_img_arrays(
            self.seq_length, 
            new_img_array, 
            self.seq_img_arrays)
        new_record_dict['cam/image_array'] = self.seq_img_arrays

        # 最新のレコード辞書を返却
        return new_record_dict



