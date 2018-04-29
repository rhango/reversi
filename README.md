# Reversi

## 概要
リバーシ（オセロ）のゲーム  
Deep Q-Network（DQN）を用いたAIを実装

## 依存パッケージ
* chainer
* chainerrl
* numpy
* cupy
* gym
* PyQt5
* matplotlib

## ファイル構成
| File | 概要 |
|:--- |:--- |
| main.py | ゲーム実行用 |
| gui.py | GUI |
| engine.py | ターン処理 |
| players.py | プレイヤーの実装 |
| dqn.py | DQNを用いたAI |
| train.py | DQNの学習用のスクリプト |
| tail_recursive.py | 再帰関数用のデコレータ |
| config.json | AIの設定 |
| DQN/ | AIの学習結果 |
| data/ | ログ |
