"""
文化マッチングモデル

このモデルは、候補者と企業の文化的適合性を評価するためのモデルです。
自然言語処理と機械学習の手法を使用して、候補者の経歴、価値観と
企業の文化、価値観のマッチング度を分析します。

作者: AI採用システム開発チーム
バージョン: 1.0.0
最終更新日: 2023-04-01
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate
from tensorflow.keras.models import Model
from transformers import BertTokenizer, TFBertModel
import logging
import os
import json

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CultureMatchingModel:
    """
    企業文化マッチングモデルクラス
    候補者と企業の文化適合性を評価します。
    """
    
    def __init__(self, model_path=None):
        """
        モデルの初期化

        Args:
            model_path (str, optional): 保存済みモデルのパス
        """
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.bert_model = TFBertModel.from_pretrained('bert-base-multilingual-cased')
        
        # デフォルトの属性リスト
        self.candidate_attributes = [
            'skills', 'experience', 'education', 'values', 
            'work_style', 'interests', 'personality'
        ]
        
        self.company_attributes = [
            'culture', 'values', 'mission', 'team_structure',
            'work_environment', 'management_style'
        ]
        
        # モデル定義
        self.model = self._build_model()
        
        # 保存済みモデルがあれば読み込む
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            self.model.load_weights(model_path)
    
    def _build_model(self):
        """
        モデルアーキテクチャを構築する

        Returns:
            Model: コンパイル済みのKeras Model
        """
        # 候補者データの入力
        candidate_input = Input(shape=(768,), name='candidate_embedding')
        
        # 企業データの入力
        company_input = Input(shape=(768,), name='company_embedding')
        
        # 入力の結合
        merged = Concatenate()([candidate_input, company_input])
        
        # 隠れ層
        x = Dense(512, activation='relu')(merged)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        
        # 出力層（マッチングスコアと分析カテゴリ）
        match_score = Dense(1, activation='sigmoid', name='match_score')(x)
        category_scores = Dense(5, activation='softmax', name='category_scores')(x)
        
        # モデル定義
        model = Model(
            inputs=[candidate_input, company_input],
            outputs=[match_score, category_scores]
        )
        
        # コンパイル
        model.compile(
            optimizer='adam',
            loss={
                'match_score': 'mse',
                'category_scores': 'categorical_crossentropy'
            },
            metrics={
                'match_score': ['mae'],
                'category_scores': ['accuracy']
            }
        )
        
        return model
    
    def _prepare_text_embedding(self, text):
        """
        テキストデータをBERTエンベディングに変換する

        Args:
            text (str): 入力テキスト

        Returns:
            numpy.ndarray: テキストの768次元エンベディング
        """
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='tf'
        )
        
        outputs = self.bert_model(encoded)
        # CLSトークンのエンベディングを使用（文章全体の表現）
        embedding = outputs.last_hidden_state[:, 0, :].numpy()
        return embedding[0]
    
    def _prepare_candidate_data(self, candidate_data):
        """
        候補者データを前処理してエンベディングに変換する

        Args:
            candidate_data (dict): 候補者の属性データ

        Returns:
            numpy.ndarray: 候補者データの768次元エンベディング
        """
        # 候補者の全属性を1つのテキストに結合
        text = ""
        for attr in self.candidate_attributes:
            if attr in candidate_data:
                if isinstance(candidate_data[attr], list):
                    text += f"{attr}: {', '.join(candidate_data[attr])}. "
                else:
                    text += f"{attr}: {candidate_data[attr]}. "
        
        return self._prepare_text_embedding(text)
    
    def _prepare_company_data(self, company_data):
        """
        企業データを前処理してエンベディングに変換する

        Args:
            company_data (dict): 企業の属性データ

        Returns:
            numpy.ndarray: 企業データの768次元エンベディング
        """
        # 企業の全属性を1つのテキストに結合
        text = ""
        for attr in self.company_attributes:
            if attr in company_data:
                if isinstance(company_data[attr], list):
                    text += f"{attr}: {', '.join(company_data[attr])}. "
                else:
                    text += f"{attr}: {company_data[attr]}. "
        
        return self._prepare_text_embedding(text)
    
    def predict(self, candidate_data, company_data):
        """
        候補者と企業の文化マッチングを予測する

        Args:
            candidate_data (dict): 候補者の属性データ
            company_data (dict): 企業の属性データ

        Returns:
            dict: マッチングスコアと分析結果
        """
        # データ前処理
        candidate_embedding = self._prepare_candidate_data(candidate_data)
        company_embedding = self._prepare_company_data(company_data)
        
        # 予測
        candidate_embedding = np.expand_dims(candidate_embedding, axis=0)
        company_embedding = np.expand_dims(company_embedding, axis=0)
        
        match_score, category_scores = self.model.predict(
            [candidate_embedding, company_embedding]
        )
        
        # スコアを0-100のスケールに変換
        match_score_percent = float(match_score[0][0] * 100)
        
        # カテゴリスコアの分析
        categories = ['values_alignment', 'skill_relevance', 'team_fit', 
                     'growth_potential', 'cultural_contribution']
        
        category_results = {
            categories[i]: float(category_scores[0][i] * 100) 
            for i in range(len(categories))
        }
        
        # 強みと改善点の分析
        strengths = []
        improvements = []
        
        for category, score in category_results.items():
            if score >= 70:
                strengths.append(category)
            elif score <= 40:
                improvements.append(category)
        
        # レコメンデーションの生成
        recommendations = self._generate_recommendations(
            category_results, candidate_data, company_data
        )
        
        return {
            'match_score': round(match_score_percent, 1),
            'analysis_details': {
                'category_scores': category_results,
                'strengths': strengths,
                'improvements': improvements,
                'recommendations': recommendations
            }
        }
    
    def _generate_recommendations(self, category_scores, candidate_data, company_data):
        """
        マッチング結果に基づいて推奨事項を生成する

        Args:
            category_scores (dict): カテゴリごとのスコア
            candidate_data (dict): 候補者データ
            company_data (dict): 企業データ

        Returns:
            list: 推奨事項のリスト
        """
        recommendations = []
        
        # 値の整合性が低い場合
        if category_scores['values_alignment'] < 50:
            recommendations.append("企業の価値観と使命についてより詳しく調査してください")
        
        # スキル関連性が低い場合
        if category_scores['skill_relevance'] < 50:
            recommendations.append("このロールに関連する追加スキルの開発を検討してください")
        
        # チームフィットが低い場合
        if category_scores['team_fit'] < 50:
            recommendations.append("チーム構造とコミュニケーションスタイルについて詳細を確認してください")
        
        # 成長ポテンシャルが高い場合
        if category_scores['growth_potential'] > 70:
            recommendations.append("この企業でのキャリア開発パスについて詳しく質問してください")
        
        # デフォルトのレコメンデーション
        if not recommendations:
            recommendations.append("面接中に企業の日常業務について具体的な質問をしてください")
        
        return recommendations
    
    def train(self, train_data, validation_data=None, epochs=10, batch_size=32):
        """
        モデルをトレーニングする

        Args:
            train_data (list): トレーニングデータのリスト
            validation_data (list, optional): 検証データのリスト
            epochs (int, optional): エポック数
            batch_size (int, optional): バッチサイズ

        Returns:
            History: トレーニング履歴
        """
        # トレーニングデータの準備
        X_candidate = []
        X_company = []
        y_match = []
        y_category = []
        
        for item in train_data:
            candidate_embedding = self._prepare_candidate_data(item['candidate'])
            company_embedding = self._prepare_company_data(item['company'])
            
            X_candidate.append(candidate_embedding)
            X_company.append(company_embedding)
            y_match.append(item['match_score'] / 100.0)  # 0-1のスケールに正規化
            y_category.append(item['category_scores'])
        
        X_candidate = np.array(X_candidate)
        X_company = np.array(X_company)
        y_match = np.array(y_match)
        y_category = np.array(y_category)
        
        # 検証データの準備
        validation_data_processed = None
        if validation_data:
            X_val_candidate = []
            X_val_company = []
            y_val_match = []
            y_val_category = []
            
            for item in validation_data:
                candidate_embedding = self._prepare_candidate_data(item['candidate'])
                company_embedding = self._prepare_company_data(item['company'])
                
                X_val_candidate.append(candidate_embedding)
                X_val_company.append(company_embedding)
                y_val_match.append(item['match_score'] / 100.0)
                y_val_category.append(item['category_scores'])
            
            X_val_candidate = np.array(X_val_candidate)
            X_val_company = np.array(X_val_company)
            y_val_match = np.array(y_val_match)
            y_val_category = np.array(y_val_category)
            
            validation_data_processed = (
                [X_val_candidate, X_val_company],
                [y_val_match, y_val_category]
            )
        
        # モデルのトレーニング
        history = self.model.fit(
            [X_candidate, X_company],
            [y_match, y_category],
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data_processed,
            verbose=1
        )
        
        return history
    
    def save(self, model_path):
        """
        モデルを保存する

        Args:
            model_path (str): モデルの保存先パス
        """
        self.model.save_weights(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # モデル設定も保存
        config_path = os.path.join(os.path.dirname(model_path), 'model_config.json')
        config = {
            'candidate_attributes': self.candidate_attributes,
            'company_attributes': self.company_attributes
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        logger.info(f"Model configuration saved to {config_path}")

# モジュールとして実行された場合の動作
if __name__ == "__main__":
    # サンプルデータ
    candidate = {
        'skills': ['Python', 'データ分析', 'コミュニケーション'],
        'experience': '5年間のデータサイエンス経験',
        'education': '情報科学修士',
        'values': ['チームワーク', 'イノベーション', '誠実さ'],
        'work_style': '協力的で自己主導型',
        'interests': ['機械学習', 'データビジュアライゼーション'],
        'personality': '分析的で創造的'
    }
    
    company = {
        'culture': '革新的でチーム指向',
        'values': ['透明性', '顧客中心', '継続的改善'],
        'mission': 'データを活用して世界をより良くする',
        'team_structure': '小規模なクロスファンクショナルチーム',
        'work_environment': 'フレキシブルでリモートフレンドリー',
        'management_style': 'フラットな階層と自律性の重視'
    }
    
    # モデルのインスタンス化とサンプル予測
    model = CultureMatchingModel()
    result = model.predict(candidate, company)
    
    print("文化マッチングの結果:")
    print(f"マッチングスコア: {result['match_score']}")
    print(f"分析詳細: {json.dumps(result['analysis_details'], indent=2, ensure_ascii=False)}") 