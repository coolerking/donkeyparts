# donkeyparts

個人的に作成した、独自Donkey Car partsクラス。

## シミュレータ対応

独自モデルを作成した場合、Simulatorで実行できなくなるため、SimPilotクラスを作成。

## 機械学習モデル

* ConvLSTM2D

指定フレーム数分の過去イメージを入力とする。

* InceptionV3 ファインチューニング

InceptionV3自体が重かったため、作ったがPC上で動作させて終わり。

### Floyd Hub

機械学習のトレーニングをFloyd Hubでも実行できるようにYMLファイルを作成。
