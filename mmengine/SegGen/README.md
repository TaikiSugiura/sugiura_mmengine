# 使い方

注意：このリポジトリをcloneしたら，以下のコマンドでsubmoduleを更新すること．

```bash
git submodule update --init --recursive
```

## 準備

必要ならDetectron2のインストール

```bash
sudo pip uninstall detectron2
sudo pip install git+https://github.com/facebookresearch/detectron2
```

その他準備

```bash
sudo sh prepare.sh
```

## 実行

- inference.py
- inference.ipynb