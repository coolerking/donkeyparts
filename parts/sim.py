# -*- coding: utf-8 -*-
"""
独自モデルを作成する場合は、本クラスを継承することでDonkey Simulatorを使用することができる。
"""


#import donkeycar as dk
import socketio
from donkeycar.parts.simulation import SteeringServer
from donkeycar.parts.keras import KerasPilot

class SimPilot(KerasPilot):
    '''
    シミュレーション実行可能なオートパイロット抽象クラス。
    独自のオートパイロットを作成する場合は、このクラスを継承して、モデルを実装する。
    '''

    def sim(self, model_path=None, top_speed=0.0):
        '''
        Donkey Simulatorへ独自モデルの結果を提供するために推論APIを提供します。
        donkey simコマンドを使用する場合、KerasCategoricalもしくはKerasLinearクラス
        の実装モデルにしか対応していないため、manage.py上に機能を追加しました。
        ポートは9090を使用します(変更不可)。
    
        引数：
            model_path      モデルファイルへのパス、指定無しの場合はロードしない
            top_speed       トップスピード値（シミュレータではスロットル推論値は使用されない）、指定しない場合は速度は0.0となる
        '''

        # このメソッドのみで使用するパッケージのインポート


        # モデルのロード
        if model_path:
            print("start loading trained model file")
            super().load(model_path)
            print("finish loading trained model file")
    
        # ソケットサーバフレームワークの開始
        sio = socketio.Server()

        # Sim サーバハンドラの開始
        ss = SteeringServer(sio, kpart=self, top_speed=top_speed, image_part=None)

        # イベントおよびハンドラ関数を登録

        @sio.on('telemetry')
        def telemetry(sid, data):
            ss.telemetry(sid, data)

        @sio.on('connect')
        def connect(sid, environ):
            ss.connect(sid, environ)

        # リッスン開始
        ss.go(('0.0.0.0', 9090))