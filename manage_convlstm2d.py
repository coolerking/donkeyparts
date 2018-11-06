#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Donkey Car V2 を独自作成した ConvLSTM2Dモデルで自動運転させるためのスクリプト。donkey2.pyをベースに修正しています。

Usage:
    manage_convlstm2d.py (drive) [--model=<model>] [--js] [--chaos]
    manage_convlstm2d.py (train) [--tub=<tub1,tub2,..tubn>]  (--model=<model>) [--base_model=<base_model>] [--no_cache]
    manage_convlstm2d.py (sim) (--model=<model>) (--top_speed=<top_speed>)

Options:
    -h --help               このコードを表示する
    --tub TUBPATHS          tubデータが格納されているディレクトリへのパス、カンマをつけて複数指定可能、ワイルドカード指定可能（例 "~/tubs/*"）
    --js                    ジョイスティックを使用する
    --chaos                 マニュアル運転時に定期的にランダムにステアリングを切る
    --model MODELPATH       モデルファイルへのパス、１ファイルのみ指定可能
    --base_model MODELPATH  学習済みモデルの途中からトレーニングさせる際に最初にロードするモデルファイルへのパス、ファインチューニングなどで使用
    --no_cache              キャッシュしない
    --top_speed TOPSPEED    最大スロットル値を指定する、Simulator側ではスロットルを推論データを使用できないため
"""
import os
from docopt import docopt

import donkeycar as dk

#import parts
from donkeycar.parts.camera import PiCamera
from donkeycar.parts.transform import Lambda
from donkeycar.parts.keras import KerasCategorical
from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle
from donkeycar.parts.datastore import TubGroup, TubWriter
# PS3コントローラ使用時コメントアウト
from donkeycar.parts.controller import LocalWebController, JoystickController
from donkeycar.parts.clock import Timestamp

# 独自モデルをインポート
from parts.pilot import ConvLSTM2DPilot

# 独自モデル学習データ用generatorを使用
from parts.seq_datastore import SeqTubGroup

# Logcool社製ジョイパッド設定を使用時コメントアウト
#from parts.controller_logicool import JoystickController

def drive(cfg, model_path=None, use_joystick=False, use_chaos=False):
    """
    たくさんのパーツから作業用のロボットVehicleを構築します。
    各パーツはVehicleループ内のジョブとして実行され、コンストラクタフラグ `threaded`
    に応じて `run` メソッドまたは `run_threaded` メソッドを呼び出します。
    すべてのパーツは、 `cfg.DRIVE_LOOP_HZ` で指定されたフレームレート
    (デフォルト：20MHz)で順次更新され、各パーツが適時に処理を終了すると仮定して
    ループを無限に繰り返します。
    パーツにはラベル付きの `inputs` と `outputs` があります。
    フレームワークは、ラベル付き `outputs` の値を、
    同じ名前の `inputs` を要求する別のパーツに渡します。

    引数：
        cfg             config.py を読み込んだオブジェクト
        model_path      学習済みモデルファイルのパス
        use_joystick    ジョイスティックを使用するかどうかの真偽値
        use_chaos       操舵に一定のランダム操作を加えるかどうかの真偽値
    """

    V = dk.vehicle.Vehicle()

    clock = Timestamp()
    V.add(clock, outputs='timestamp')

    cam = PiCamera(resolution=cfg.CAMERA_RESOLUTION)
    V.add(cam, outputs=['cam/image_array'], threaded=True)

    if use_joystick or cfg.USE_JOYSTICK_AS_DEFAULT:
        ctr = JoystickController(max_throttle=cfg.JOYSTICK_MAX_THROTTLE,
                                 steering_scale=cfg.JOYSTICK_STEERING_SCALE,
                                 auto_record_on_throttle=cfg.AUTO_RECORD_ON_THROTTLE)
    else:
        # このWebコントローラでは、ステアリング、スロットル、モードなどを管理する
        # Webサーバを作成
        ctr = LocalWebController(use_chaos=use_chaos)

    V.add(ctr,
          inputs=['cam/image_array'],
          outputs=['user/angle', 'user/throttle', 'user/mode', 'recording'],
          threaded=True)

    # パイロットモジュールを走らせるべきかどうかを毎回判別させるためのパーツ
    # この関数の結果の真偽値で自動運転パーツ(ConvLSTM2DPilot)を実行するかを決定させる
    def pilot_condition(mode):
        if mode == 'user':
            return False
        else:
            return True
    pilot_condition_part = Lambda(pilot_condition)
    V.add(pilot_condition_part, inputs=['user/mode'],
                                outputs=['run_pilot'])

    # Run the pilot if the mode is not user.
    # 独自のパイロットに置き換え
    #kl = KerasCategorical()
    kl = ConvLSTM2DPilot()

    if model_path:
        print("start loading trained model file")
        kl.load(model_path)
        print("finish loading trained model file")

    V.add(kl, inputs=['cam/image_array'],
              outputs=['pilot/angle', 'pilot/throttle'],
              run_condition='run_pilot')

    # 実車のインプットとしてどの値を使うかのモード選択
    def drive_mode(mode,
                   user_angle, user_throttle,
                   pilot_angle, pilot_throttle):
        if mode == 'user':
            return user_angle, user_throttle

        elif mode == 'local_angle':
            return pilot_angle, user_throttle

        else:
            return pilot_angle, pilot_throttle

    drive_mode_part = Lambda(drive_mode)
    V.add(drive_mode_part,
          inputs=['user/mode', 'user/angle', 'user/throttle',
                  'pilot/angle', 'pilot/throttle'],
          outputs=['angle', 'throttle'])

    steering_controller = PCA9685(cfg.STEERING_CHANNEL)
    steering = PWMSteering(controller=steering_controller,
                           left_pulse=cfg.STEERING_LEFT_PWM,
                           right_pulse=cfg.STEERING_RIGHT_PWM)

    throttle_controller = PCA9685(cfg.THROTTLE_CHANNEL)
    throttle = PWMThrottle(controller=throttle_controller,
                           max_pulse=cfg.THROTTLE_FORWARD_PWM,
                           zero_pulse=cfg.THROTTLE_STOPPED_PWM,
                           min_pulse=cfg.THROTTLE_REVERSE_PWM)

    V.add(steering, inputs=['angle'])
    V.add(throttle, inputs=['throttle'])

    # add tub to save data
    inputs = ['cam/image_array', 'user/angle', 'user/throttle', 'user/mode', 'timestamp']
    types = ['image_array', 'float', 'float',  'str', 'str']

    #multiple tubs
    #th = TubHandler(path=cfg.DATA_PATH)
    #tub = th.new_tub_writer(inputs=inputs, types=types)

    # single tub
    tub = TubWriter(path=cfg.TUB_PATH, inputs=inputs, types=types)
    V.add(tub, inputs=inputs, run_condition='recording')

    # run the vehicle
    V.start(rate_hz=cfg.DRIVE_LOOP_HZ,
            max_loop_count=cfg.MAX_LOOPS)




def train(cfg, tub_names, new_model_path, base_model_path=None): #, logdir=None ):
    """
    tub_names で指定されたデータを学習コーパスとしてNNをトレーニングする
    ファイル名 base_model_path を指定した場合、トレーニング前にロードする
    学習済みトレーニングモデルをファイル名 model_name で保存する

    引数：
        cfg             config.py を読み込んだオブジェクト
        tub_names       タブディレクトリ名群
        new_model_path  作成するモデルファイルのパス
        base_model_path トレーニング前に読み込むモデルファイルのパス
    """
    X_keys = ['cam/image_array']
    y_keys = ['user/angle', 'user/throttle']
    def train_record_transform(record):
        """ convert categorical steering to linear and apply image augmentations """
        record['user/angle'] = dk.util.data.linear_bin(record['user/angle'])
        # TODO add augmentation that doesn't use opencv
        return record

    def val_record_transform(record):
        """ convert categorical steering to linear """
        record['user/angle'] = dk.util.data.linear_bin(record['user/angle'])
        return record

    new_model_path = os.path.expanduser(new_model_path)

    # モデルをConvRNNモデルベースに変更
    #kl = KerasCategorical()
    kl = ConvLSTM2DPilot()

    if base_model_path is not None:
        base_model_path = os.path.expanduser(base_model_path)
        kl.load(base_model_path)

    print('tub_names', tub_names)
    if not tub_names:
        tub_names = os.path.join(cfg.DATA_PATH, '*')
    
    # ConvLSTM2Pilot用学習データを作成するために
    # SeqTubGroup で代用
    # tubgroup = TubGroup(tub_names)
    tubgroup = SeqTubGroup(tub_names, seq_length=kl.seq_length)
    
    train_gen, val_gen = tubgroup.get_train_val_gen(X_keys, y_keys,
                                                    train_record_transform=train_record_transform,
                                                    val_record_transform=val_record_transform,
                                                    batch_size=cfg.BATCH_SIZE,
                                                    train_frac=cfg.TRAIN_TEST_SPLIT)

    total_records = len(tubgroup.df)
    total_train = int(total_records * cfg.TRAIN_TEST_SPLIT)
    total_val = total_records - total_train
    print('train: %d, validation: %d' % (total_train, total_val))
    steps_per_epoch = total_train // cfg.BATCH_SIZE
    print('steps_per_epoch', steps_per_epoch)

    #if logdir is None:
    #    logdir = cfg.LOG_DIR

    kl.train(train_gen,
             val_gen,
             saved_model_path=new_model_path,
             steps=steps_per_epoch,
             train_split=cfg.TRAIN_TEST_SPLIT) # ,
             #logdir=logdir)


if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config()

    # 自動・手動運転
    if args['drive']:
        drive(cfg, model_path = args['--model'], use_joystick=args['--js'], use_chaos=args['--chaos'])

    # トレーニング
    elif args['train']:
        tub = args['--tub']
        new_model_path = args['--model']
        base_model_path = args['--base_model']
        cache = not args['--no_cache']
        #logdir = args['--logdir']
        train(cfg, tub, new_model_path, base_model_path) #, logdir=logdir)

    # Donkey Simulatorに推論APIを提供
    elif args['sim']:

        # シミュレータへ推論APIを提供
        ConvLSTM2DPilot().sim(args['--model'], float(args['--top_speed']))
