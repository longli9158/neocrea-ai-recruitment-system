"""
採用予測モデル

このモデルは、候補者のプロファイル、面接評価、文化マッチングスコアなどを
分析し、採用可能性を予測します。ランダムフォレスト、XGBoost、ニューラルネットワーク
などの機械学習モデルを組み合わせたアンサンブルアプローチを使用しています。

作者: AI採用システム開発チーム
バージョン: 1.0.0
最終更新日: 2023-04-01
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
import joblib
import json
import os
import logging

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RecruitmentPredictionModel:
    """
    採用予測モデルクラス
    候補者データに基づいて採用可能性を予測します。
    """
    
    def __init__(self, model_path=None):
        """
        モデルの初期化

        Args:
            model_path (str, optional): 保存済みモデルのパス
        """
        # 特徴量定義
        self.numeric_features = [
            'experience_years', 'education_level', 'interview_score_1',
            'interview_score_2', 'interview_score_3', 'culture_fit_score',
            'technical_score', 'leadership_score', 'teamwork_score'
        ]
        
        self.categorical_features = [
            'role_category', 'department', 'education_field', 
            'previous_industry', 'source'
        ]
        
        self.text_features = [
            'skills', 'certifications'
        ]
        
        # モデル定義
        self.models = {
            'random_forest': None,
            'xgboost': None,
            'neural_network': None
        }
        
        self.preprocessor = None
        self.is_fitted = False
        
        # 保存済みモデルがあれば読み込む
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            self.load(model_path)
            self.is_fitted = True
    
    def _build_preprocessor(self):
        """
        特徴量前処理パイプラインを構築する

        Returns:
            ColumnTransformer: 特徴量変換用のプリプロセッサ
        """
        # 数値特徴量の前処理
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        # カテゴリ特徴量の前処理
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # 特徴量変換器
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='drop'  # テキスト特徴量は別途処理
        )
        
        return preprocessor
    
    def _preprocess_data(self, X, fit=False):
        """
        データを前処理する

        Args:
            X (pd.DataFrame): 入力データ
            fit (bool, optional): プリプロセッサを新たに適合させるかどうか

        Returns:
            np.ndarray: 前処理済みの特徴量
        """
        # プリプロセッサがない場合は作成
        if self.preprocessor is None or fit:
            self.preprocessor = self._build_preprocessor()
            
        # 欠損値の処理
        X_filled = X.copy()
        for feature in self.numeric_features:
            if feature in X.columns:
                X_filled[feature] = X_filled[feature].fillna(X_filled[feature].median())
        
        for feature in self.categorical_features:
            if feature in X.columns:
                X_filled[feature] = X_filled[feature].fillna('unknown')
        
        # スキルとその他のテキスト特徴量の処理
        # 各スキルをバイナリ特徴量に変換
        skill_features = []
        common_skills = [
            'python', 'java', 'javascript', 'c++', 'sql', 'data_analysis',
            'machine_learning', 'project_management', 'communication',
            'leadership', 'teamwork', 'problem_solving'
        ]
        
        for skill in common_skills:
            if 'skills' in X.columns:
                X_filled[f'has_skill_{skill}'] = X_filled['skills'].apply(
                    lambda x: 1 if isinstance(x, list) and skill in [s.lower() for s in x] else 0
                )
                skill_features.append(f'has_skill_{skill}')
        
        # その他の数値特徴量の作成
        if 'skills' in X.columns:
            X_filled['skill_count'] = X_filled['skills'].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
            skill_features.append('skill_count')
        
        if 'certifications' in X.columns:
            X_filled['certification_count'] = X_filled['certifications'].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
            skill_features.append('certification_count')
        
        # プリプロセッサを適合または変換
        if fit:
            X_processed = self.preprocessor.fit_transform(X_filled)
        else:
            X_processed = self.preprocessor.transform(X_filled)
        
        # スキル特徴量を結合
        skill_features_array = X_filled[skill_features].values
        
        # processed_featuresが疎行列の場合は密行列に変換
        if hasattr(X_processed, 'toarray'):
            X_processed = X_processed.toarray()
        
        # スキル特徴量と前処理済み特徴量を結合
        X_final = np.hstack([X_processed, skill_features_array])
        
        return X_final
    
    def _build_neural_network(self, input_dim):
        """
        ニューラルネットワークモデルを構築する

        Args:
            input_dim (int): 入力特徴量の次元

        Returns:
            Model: Kerasモデル
        """
        inputs = Input(shape=(input_dim,))
        
        x = Dense(128, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        # 採用確率の出力
        hire_prob = Dense(1, activation='sigmoid', name='hire_prob')(x)
        
        # 主要因子の出力（どの特徴が予測に最も寄与しているか）
        key_factors = Dense(5, activation='softmax', name='key_factors')(x)
        
        model = Model(inputs=inputs, outputs=[hire_prob, key_factors])
        
        model.compile(
            optimizer='adam',
            loss={
                'hire_prob': 'binary_crossentropy',
                'key_factors': 'categorical_crossentropy'
            },
            metrics={
                'hire_prob': ['accuracy'],
                'key_factors': ['accuracy']
            }
        )
        
        return model
    
    def train(self, X, y, validation_split=0.2, epochs=50, batch_size=32):
        """
        モデルをトレーニングする

        Args:
            X (pd.DataFrame): 入力特徴量
            y (pd.Series or dict): ターゲット変数（採用/不採用）とオプションの追加ラベル
            validation_split (float, optional): 検証データの割合
            epochs (int, optional): エポック数（ニューラルネットワーク用）
            batch_size (int, optional): バッチサイズ（ニューラルネットワーク用）

        Returns:
            self: トレーニング済みモデル
        """
        # ターゲット変数の解析
        if isinstance(y, dict):
            y_hire = y['hire']
            y_factors = y.get('key_factors', None)
        else:
            y_hire = y
            y_factors = None
        
        # データの前処理
        X_processed = self._preprocess_data(X, fit=True)
        
        # トレーニング/検証データの分割
        X_train, X_val, y_train, y_val = train_test_split(
            X_processed, y_hire, test_size=validation_split, random_state=42
        )
        
        if y_factors is not None:
            _, _, y_factors_train, y_factors_val = train_test_split(
                X_processed, y_factors, test_size=validation_split, random_state=42
            )
        
        # ランダムフォレストモデルのトレーニング
        logger.info("Training Random Forest model...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model
        
        rf_val_accuracy = accuracy_score(y_val, rf_model.predict(X_val))
        logger.info(f"Random Forest validation accuracy: {rf_val_accuracy:.4f}")
        
        # XGBoostモデルのトレーニング
        logger.info("Training XGBoost model...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        self.models['xgboost'] = xgb_model
        
        xgb_val_accuracy = accuracy_score(y_val, xgb_model.predict(X_val))
        logger.info(f"XGBoost validation accuracy: {xgb_val_accuracy:.4f}")
        
        # ニューラルネットワークモデルのトレーニング
        logger.info("Training Neural Network model...")
        nn_model = self._build_neural_network(X_processed.shape[1])
        
        # 主要因子ラベルがある場合はそれも使用
        if y_factors is not None:
            nn_history = nn_model.fit(
                X_train,
                {'hire_prob': y_train, 'key_factors': y_factors_train},
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(
                    X_val,
                    {'hire_prob': y_val, 'key_factors': y_factors_val}
                ),
                verbose=2
            )
        else:
            # 主要因子ラベルがない場合はダミーデータを使用
            dummy_factors = np.zeros((len(y_train), 5))
            dummy_factors_val = np.zeros((len(y_val), 5))
            
            nn_history = nn_model.fit(
                X_train,
                {'hire_prob': y_train, 'key_factors': dummy_factors},
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(
                    X_val,
                    {'hire_prob': y_val, 'key_factors': dummy_factors_val}
                ),
                verbose=2
            )
        
        self.models['neural_network'] = nn_model
        
        # バリデーション精度を表示
        val_loss, val_hire_loss, val_factors_loss, val_hire_acc, val_factors_acc = nn_model.evaluate(
            X_val,
            {'hire_prob': y_val, 'key_factors': y_factors_val if y_factors is not None else dummy_factors_val},
            verbose=0
        )
        logger.info(f"Neural Network validation accuracy: {val_hire_acc:.4f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, candidate_data):
        """
        候補者の採用可能性を予測する

        Args:
            candidate_data (dict or pd.DataFrame): 候補者データ

        Returns:
            dict: 予測結果
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'train' first.")
        
        # 単一の候補者データをDataFrameに変換
        if isinstance(candidate_data, dict):
            X = pd.DataFrame([candidate_data])
        else:
            X = candidate_data
        
        # データの前処理
        X_processed = self._preprocess_data(X)
        
        # 各モデルでの予測
        rf_pred_proba = self.models['random_forest'].predict_proba(X_processed)[:, 1]
        xgb_pred_proba = self.models['xgboost'].predict_proba(X_processed)[:, 1]
        
        nn_pred = self.models['neural_network'].predict(X_processed)
        nn_pred_proba = nn_pred[0].flatten()
        key_factors_proba = nn_pred[1][0]
        
        # アンサンブル予測（重み付き平均）
        ensemble_proba = (0.3 * rf_pred_proba + 0.3 * xgb_pred_proba + 0.4 * nn_pred_proba)[0]
        
        # 主要因子の解析
        factor_categories = [
            'technical_skills', 'experience', 'cultural_fit',
            'interview_performance', 'education'
        ]
        
        key_factors = [
            {
                'factor': factor_categories[i],
                'impact': float(key_factors_proba[i])
            }
            for i in range(len(factor_categories))
        ]
        
        # 重要度の高い順にソート
        key_factors = sorted(key_factors, key=lambda x: x['impact'], reverse=True)
        
        # 予測とその説明
        hire_prediction = "採用" if ensemble_proba >= 0.5 else "不採用"
        confidence_score = max(ensemble_proba, 1 - ensemble_proba)
        
        return {
            'hireChance': float(ensemble_proba),
            'prediction': hire_prediction,
            'confidenceScore': float(confidence_score),
            'keyFactors': key_factors[:3]  # 上位3つの要因のみ返す
        }
    
    def evaluate(self, X_test, y_test):
        """
        モデルのパフォーマンスを評価する

        Args:
            X_test (pd.DataFrame): テストデータ
            y_test (pd.Series): テストラベル

        Returns:
            dict: 評価メトリクス
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'train' first.")
        
        # データの前処理
        X_processed = self._preprocess_data(X_test)
        
        # 各モデルでの予測
        rf_pred = self.models['random_forest'].predict(X_processed)
        xgb_pred = self.models['xgboost'].predict(X_processed)
        
        nn_pred = self.models['neural_network'].predict(X_processed)
        nn_pred_binary = (nn_pred[0].flatten() >= 0.5).astype(int)
        
        # アンサンブル予測
        ensemble_pred_proba = (
            0.3 * self.models['random_forest'].predict_proba(X_processed)[:, 1] +
            0.3 * self.models['xgboost'].predict_proba(X_processed)[:, 1] +
            0.4 * nn_pred[0].flatten()
        )
        ensemble_pred = (ensemble_pred_proba >= 0.5).astype(int)
        
        # 評価メトリクスの計算
        metrics = {
            'random_forest': {
                'accuracy': accuracy_score(y_test, rf_pred),
                'precision': precision_score(y_test, rf_pred),
                'recall': recall_score(y_test, rf_pred),
                'f1': f1_score(y_test, rf_pred)
            },
            'xgboost': {
                'accuracy': accuracy_score(y_test, xgb_pred),
                'precision': precision_score(y_test, xgb_pred),
                'recall': recall_score(y_test, xgb_pred),
                'f1': f1_score(y_test, xgb_pred)
            },
            'neural_network': {
                'accuracy': accuracy_score(y_test, nn_pred_binary),
                'precision': precision_score(y_test, nn_pred_binary),
                'recall': recall_score(y_test, nn_pred_binary),
                'f1': f1_score(y_test, nn_pred_binary)
            },
            'ensemble': {
                'accuracy': accuracy_score(y_test, ensemble_pred),
                'precision': precision_score(y_test, ensemble_pred),
                'recall': recall_score(y_test, ensemble_pred),
                'f1': f1_score(y_test, ensemble_pred)
            }
        }
        
        return metrics
    
    def save(self, model_dir):
        """
        モデルを保存する

        Args:
            model_dir (str): モデル保存ディレクトリ
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'train' first.")
        
        # ディレクトリが存在しなければ作成
        os.makedirs(model_dir, exist_ok=True)
        
        # 各モデルを保存
        joblib.dump(self.models['random_forest'], os.path.join(model_dir, 'random_forest.pkl'))
        joblib.dump(self.models['xgboost'], os.path.join(model_dir, 'xgboost.pkl'))
        self.models['neural_network'].save_weights(os.path.join(model_dir, 'neural_network.h5'))
        
        # プリプロセッサを保存
        joblib.dump(self.preprocessor, os.path.join(model_dir, 'preprocessor.pkl'))
        
        # 設定を保存
        config = {
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'text_features': self.text_features,
            'input_dim': self._preprocess_data(pd.DataFrame({feature: [0] for feature in self.numeric_features + self.categorical_features})).shape[1]
        }
        
        with open(os.path.join(model_dir, 'config.json'), 'w') as f:
            json.dump(config, f)
        
        logger.info(f"Model saved to {model_dir}")
    
    def load(self, model_dir):
        """
        保存されたモデルを読み込む

        Args:
            model_dir (str): モデル保存ディレクトリ

        Returns:
            self: 読み込まれたモデル
        """
        # 設定を読み込む
        with open(os.path.join(model_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        
        self.numeric_features = config['numeric_features']
        self.categorical_features = config['categorical_features']
        self.text_features = config['text_features']
        
        # プリプロセッサを読み込む
        self.preprocessor = joblib.load(os.path.join(model_dir, 'preprocessor.pkl'))
        
        # 各モデルを読み込む
        self.models['random_forest'] = joblib.load(os.path.join(model_dir, 'random_forest.pkl'))
        self.models['xgboost'] = joblib.load(os.path.join(model_dir, 'xgboost.pkl'))
        
        # ニューラルネットワークを再構築して重みを読み込む
        nn_model = self._build_neural_network(config['input_dim'])
        nn_model.load_weights(os.path.join(model_dir, 'neural_network.h5'))
        self.models['neural_network'] = nn_model
        
        self.is_fitted = True
        logger.info(f"Model loaded from {model_dir}")
        
        return self

# モジュールとして実行された場合の動作
if __name__ == "__main__":
    # サンプルデータ
    candidate_data = {
        'candidateId': 'c123456',
        'positionId': 'p456789',
        'experience_years': 5,
        'education_level': 4,  # 4=修士
        'education_field': '情報科学',
        'role_category': 'データサイエンティスト',
        'department': '製品開発',
        'previous_industry': 'テクノロジー',
        'source': '社員紹介',
        'skills': ['Python', '機械学習', 'データ分析', 'SQL', 'TensorFlow'],
        'certifications': ['AWS認定ソリューションアーキテクト', 'TensorFlow開発者認定'],
        'interview_score_1': 85,
        'interview_score_2': 78,
        'interview_score_3': 92,
        'culture_fit_score': 85,
        'technical_score': 88,
        'leadership_score': 75,
        'teamwork_score': 82
    }
    
    # モデルのインスタンス化とサンプル予測
    model = RecruitmentPredictionModel()
    
    # モデルがまだトレーニングされていないので、サンプルデータで簡易トレーニング
    # 実際のアプリケーションでは、事前にトレーニングされたモデルをロードするべき
    train_size = 100
    np.random.seed(42)
    
    # 簡易的なトレーニングデータを生成
    X_train = pd.DataFrame([{
        'experience_years': np.random.randint(1, 15),
        'education_level': np.random.randint(2, 6),
        'education_field': np.random.choice(['情報科学', '経営学', '工学', '数学', '自然科学']),
        'role_category': np.random.choice(['データサイエンティスト', 'ソフトウェアエンジニア', 'プロジェクトマネージャー']),
        'department': np.random.choice(['製品開発', '研究開発', 'マーケティング', '営業']),
        'previous_industry': np.random.choice(['テクノロジー', '金融', '小売', '製造', 'コンサルティング']),
        'source': np.random.choice(['社員紹介', '転職サイト', 'ヘッドハンター', '大学リクルート']),
        'skills': np.random.choice([
            ['Python', '機械学習', 'データ分析'],
            ['Java', 'C++', 'SQL'],
            ['JavaScript', 'React', 'Node.js'],
            ['Python', 'TensorFlow', 'SQL']
        ]),
        'certifications': np.random.choice([
            ['AWS認定ソリューションアーキテクト'],
            ['TensorFlow開発者認定'],
            [],
            ['PMP']
        ]),
        'interview_score_1': np.random.randint(50, 100),
        'interview_score_2': np.random.randint(50, 100),
        'interview_score_3': np.random.randint(50, 100),
        'culture_fit_score': np.random.randint(50, 100),
        'technical_score': np.random.randint(50, 100),
        'leadership_score': np.random.randint(50, 100),
        'teamwork_score': np.random.randint(50, 100)
    } for _ in range(train_size)])
    
    # シンプルな予測ルールでラベルを生成（実際にはもっと複雑）
    y_train = (
        (X_train['experience_years'] >= 3) &
        (X_train['education_level'] >= 3) &
        ((X_train['interview_score_1'] + X_train['interview_score_2'] + X_train['interview_score_3'])/3 >= 70) &
        (X_train['culture_fit_score'] >= 70) &
        (X_train['technical_score'] >= 75)
    ).astype(int)
    
    # 簡易トレーニング（実際のアプリケーションではもっと詳細に調整）
    model.train(X_train, y_train, epochs=5, batch_size=16)
    
    # サンプル予測
    result = model.predict(candidate_data)
    
    print("採用予測結果:")
    print(f"採用確率: {result['hireChance']:.2f}")
    print(f"予測: {result['prediction']}")
    print(f"信頼度: {result['confidenceScore']:.2f}")
    print("主要因子:")
    for factor in result['keyFactors']:
        print(f"  - {factor['factor']}: {factor['impact']:.2f}") 