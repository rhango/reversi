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

## ファイル構成
| File | 概要 |
|:--- |:--- |
| reversi_main.py | ゲーム実行用 |
| reversi_gui.py | GUI |
| reversi_engine.py | ターン処理 |
| reversi_players.py | プレイヤーの実装 |
| reversi_dqn.py | DQNを用いたAI |
| reversi_train.py | DQNの学習用のスクリプト |
| tail_recursive.py | 再帰関数用のデコレータ |
| reversi_config.json | AIの設定 |
| DQN/ | AIの学習結果 |
| data/ | ログ |
