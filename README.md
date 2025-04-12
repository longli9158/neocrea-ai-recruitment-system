# AI採用システム - AIモデル

## 概要

AI採用システムで使用されるAIモデルコンポーネントです。このコンポーネントは、採用プロセスのさまざまな側面を最適化するための機械学習モデルを提供します。

## 主要モデル

- **企業文化マッチングモデル**: 候補者と企業の文化的適合性を評価
- **面接フィードバックモデル**: 面接のパフォーマンスを分析し、建設的なフィードバックを生成
- **採用予測モデル**: 候補者の履歴書と面接データから採用可能性を予測

## 開発環境のセットアップ

### 前提条件

- Python 3.8以上
- pip
- AWS CLI（SageMakerデプロイ用）

### インストールと実行

```bash
# 依存関係のインストール
pip install -r requirements.txt

# モデルのトレーニング
python scripts/train_culture_matching_model.py
python scripts/train_interview_feedback_model.py
python scripts/train_recruitment_prediction_model.py

# モデルのテスト
python -m pytest tests/
```

## フォルダ構造

```
ai_model/
├── src/
│   ├── culture_matching_model.py    # 文化マッチングモデル
│   ├── interview_feedback_model.py  # 面接フィードバックモデル
│   └── recruitment_prediction_model.py  # 採用予測モデル
├── scripts/
│   ├── train_culture_matching_model.py  # トレーニングスクリプト
│   ├── train_interview_feedback_model.py
│   └── train_recruitment_prediction_model.py
├── notebooks/                       # Jupyter Notebooks
├── data/                           # サンプルデータ
├── tests/                          # テストコード
├── requirements.txt
├── buildspec.yml                   # AWS CodeBuildの設定
└── README.md
```

## 主要技術

- TensorFlow
- PyTorch
- scikit-learn
- Hugging Face Transformers
- pandas
- NumPy

## モデルのデプロイ

モデルはAWS SageMakerを使用してデプロイされます。

```bash
# SageMakerにモデルをパッケージ化してアップロード
python scripts/package_model_for_sagemaker.py --model-name culture-matching

# SageMakerでモデルをデプロイ
aws sagemaker create-model --model-name culture-matching-model --primary-container image-uri=$(aws ecr describe-repositories --repository-name culture-matching --query 'repositories[0].repositoryUri' --output text)
aws sagemaker create-endpoint-config --endpoint-config-name culture-matching-endpoint-config --production-variants varinatName=variant-1,modelName=culture-matching-model,instanceType=ml.m5.large,initialInstanceCount=1
aws sagemaker create-endpoint --endpoint-name culture-matching-endpoint --endpoint-config-name culture-matching-endpoint-config
```

## データ要件

各モデルは以下のデータ形式を期待します：

### 企業文化マッチングモデル

入力：
- 候補者プロファイル（スキル、経験、価値観など）
- 企業プロファイル（企業文化、価値観、チーム構成など）

出力：
- マッチングスコア（0-100）
- マッチング分析（強み、弱み、改善点など）

### 面接フィードバックモデル

入力：
- 面接トランスクリプト
- 質問リスト
- 回答の音声特性データ（オプション）

出力：
- 面接パフォーマンス評価
- 改善点と具体的なフィードバック

### 採用予測モデル

入力：
- 候補者の履歴書データ
- 面接評価データ
- 文化マッチングスコア

出力：
- 採用確率
- 採用/不採用の予測とその理由

## モデルの再トレーニング

モデルは定期的に再トレーニングされ、最新のデータに適応する必要があります。再トレーニングプロセスは自動化され、AWS StepFunctionsによって管理されています。

```bash
# モデルの再トレーニングをトリガー
aws stepfunctions start-execution --state-machine-arn arn:aws:states:region:account-id:stateMachine:model-retraining-workflow
``` 