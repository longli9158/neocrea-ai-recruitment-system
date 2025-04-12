"""
面接フィードバックモデル

このモデルは、面接トランスクリプトを分析し、候補者の面接パフォーマンスに関する
フィードバックを生成します。自然言語処理と感情分析を組み合わせて、
応答の質、コミュニケーションスキル、技術的知識などを評価します。

作者: AI採用システム開発チーム
バージョン: 1.0.0
最終更新日: 2023-04-01
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout
from tensorflow.keras.models import Model
from transformers import BertTokenizer, TFBertModel
import json
import os
import logging
import re

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InterviewFeedbackModel:
    """
    面接フィードバックモデルクラス
    面接トランスクリプトを分析し、詳細なフィードバックを生成します。
    """
    
    def __init__(self, model_path=None):
        """
        モデルの初期化

        Args:
            model_path (str, optional): 保存済みモデルのパス
        """
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.bert_model = TFBertModel.from_pretrained('bert-base-multilingual-cased')
        
        # 評価カテゴリ
        self.categories = [
            'communication', 'technical_knowledge', 'problem_solving',
            'cultural_fit', 'enthusiasm', 'experience_relevance'
        ]
        
        # フィードバックテンプレート
        self.feedback_templates = self._load_feedback_templates()
        
        # モデル定義
        self.model = self._build_model()
        
        # 保存済みモデルがあれば読み込む
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            self.model.load_weights(model_path)
    
    def _load_feedback_templates(self):
        """
        フィードバックテンプレートのロード
        
        Returns:
            dict: カテゴリごとのフィードバックテンプレート
        """
        # 実際のアプリケーションではJSONファイルから読み込む
        # ここではハードコードしておく
        return {
            'communication': {
                'high': [
                    "明確で簡潔なコミュニケーションスキルを示しています",
                    "質問に適切に回答し、思考プロセスを効果的に伝えています",
                    "専門用語と一般的な説明をバランス良く使用しています"
                ],
                'medium': [
                    "コミュニケーションは明確ですが、時々詳細が不足しています",
                    "回答は適切ですが、より構造化された説明が役立つでしょう",
                    "技術的な概念を説明する際に、より具体的な例を使用するとよいでしょう"
                ],
                'low': [
                    "回答がしばしば不明確または不完全です",
                    "技術的な概念を説明する際に困難が見られます",
                    "より簡潔で焦点を絞った回答を心がけるとよいでしょう"
                ]
            },
            'technical_knowledge': {
                'high': [
                    "該当分野における強固な技術的理解を示しています",
                    "複雑な技術的概念を正確に説明できています",
                    "実践的な経験に基づいた深い知識を持っています"
                ],
                'medium': [
                    "基本的な技術的理解を示していますが、一部の高度な概念でギャップがあります",
                    "主要な技術的概念を理解していますが、詳細が不足しています",
                    "より幅広い技術的知識の習得が役立つでしょう"
                ],
                'low': [
                    "基本的な技術的概念の理解に課題があります",
                    "質問に対する技術的回答が不正確または不完全です",
                    "この分野の基礎知識の強化が必要です"
                ]
            },
            'problem_solving': {
                'high': [
                    "問題を効果的に分析し、構造化されたアプローチで解決しています",
                    "複雑な問題を小さなステップに分解する能力を示しています",
                    "創造的で効率的な解決策を提案しています"
                ],
                'medium': [
                    "基本的な問題解決能力を示していますが、より複雑なケースでは詳細が不足しています",
                    "解決策は有効ですが、最適化の余地があります",
                    "問題解決のアプローチをより明確に表現するとよいでしょう"
                ],
                'low': [
                    "問題の分析と解決に体系的なアプローチが欠けています",
                    "提案された解決策は基本的または不完全です",
                    "問題解決の方法論を学ぶことが役立つでしょう"
                ]
            },
            'cultural_fit': {
                'high': [
                    "企業の価値観と使命に強い共感を示しています",
                    "チーム志向とコラボレーションの姿勢が明確です",
                    "企業文化に自然に適合すると思われます"
                ],
                'medium': [
                    "企業の価値観に対する基本的な理解を示しています",
                    "チームワークの重要性を認識していますが、具体例が限られています",
                    "会社の文化についてより深く理解することが有益でしょう"
                ],
                'low': [
                    "企業の価値観や文化への関心や理解が限られています",
                    "個人的な仕事の好みが、私たちのチーム環境と異なる可能性があります",
                    "チーム環境での成功体験についてより詳しく共有するとよいでしょう"
                ]
            },
            'enthusiasm': {
                'high': [
                    "役割と会社に対する強い情熱と熱意を示しています",
                    "業界の動向に関する優れた知識を持ち、将来のビジョンを示しています",
                    "自己啓発と継続的な学習への意欲が明確です"
                ],
                'medium': [
                    "役割に対する興味を示していますが、より深い熱意が見られるとよいでしょう",
                    "キャリア目標について基本的な理解を示していますが、詳細が欠けています",
                    "この分野でのより具体的な関心領域を特定するとよいでしょう"
                ],
                'low': [
                    "役割や会社に対する熱意が限定的です",
                    "キャリア目標が不明確または役割と一致していません",
                    "この分野でのあなたの関心を深く掘り下げることをお勧めします"
                ]
            },
            'experience_relevance': {
                'high': [
                    "この役割に直接関連する豊富な経験を持っています",
                    "過去のプロジェクトが現在の役割の要件と強く一致しています",
                    "すぐに貢献できる適切なスキルセットを持っています"
                ],
                'medium': [
                    "関連する経験を持っていますが、一部の主要分野では限られています",
                    "転用可能なスキルを持っていますが、特定の技術的要件において学習が必要です",
                    "経験をこの役割の要件により明確に関連付けるとよいでしょう"
                ],
                'low': [
                    "この役割に直接関連する経験が限られています",
                    "主要な必須スキルにおけるギャップが見られます",
                    "この分野での経験を増やすことをお勧めします"
                ]
            }
        }
    
    def _build_model(self):
        """
        モデルアーキテクチャを構築する

        Returns:
            Model: コンパイル済みのKerasモデル
        """
        # トランスクリプト入力（BERT埋め込み）
        transcript_input = Input(shape=(768,), name='transcript_embedding')
        
        # 質問入力（BERT埋め込み）
        questions_input = Input(shape=(768,), name='questions_embedding')
        
        # 音声特性入力（オプション）
        audio_features_input = Input(shape=(10,), name='audio_features')
        
        # 入力の結合
        combined = tf.keras.layers.Concatenate()([
            transcript_input, 
            questions_input, 
            audio_features_input
        ])
        
        # 隠れ層
        x = Dense(512, activation='relu')(combined)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        
        # 出力層 - 全体スコアと各カテゴリのスコア
        overall_score = Dense(1, activation='sigmoid', name='overall_score')(x)
        category_scores = Dense(len(self.categories), activation='sigmoid', name='category_scores')(x)
        
        # モデル定義
        model = Model(
            inputs=[transcript_input, questions_input, audio_features_input],
            outputs=[overall_score, category_scores]
        )
        
        # コンパイル
        model.compile(
            optimizer='adam',
            loss={
                'overall_score': 'mse',
                'category_scores': 'mse'
            },
            metrics={
                'overall_score': ['mae'],
                'category_scores': ['mae']
            }
        )
        
        return model
    
    def _preprocess_transcript(self, transcript, questions):
        """
        面接トランスクリプトを前処理する

        Args:
            transcript (str): 面接のトランスクリプト
            questions (list): 面接で尋ねられた質問のリスト

        Returns:
            tuple: 前処理されたトランスクリプトと質問の埋め込み
        """
        # 質問と回答のペアを抽出
        qa_pairs = []
        
        # トランスクリプトを文に分割
        sentences = re.split(r'[.!?]', transcript)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # すべての質問と回答のテキストを連結
        all_questions_text = " ".join(questions)
        all_answers_text = " ".join(sentences)
        
        # BERTエンベディングを取得
        transcript_embedding = self._get_bert_embedding(all_answers_text)
        questions_embedding = self._get_bert_embedding(all_questions_text)
        
        return transcript_embedding, questions_embedding
    
    def _get_bert_embedding(self, text):
        """
        テキストのBERT埋め込みを取得する

        Args:
            text (str): 入力テキスト

        Returns:
            numpy.ndarray: BERT埋め込み
        """
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='tf'
        )
        
        outputs = self.bert_model(encoded)
        # CLSトークンのエンベディングを使用
        embedding = outputs.last_hidden_state[:, 0, :].numpy()
        return embedding[0]
    
    def _preprocess_audio_features(self, audio_features=None):
        """
        音声特性を前処理する

        Args:
            audio_features (dict, optional): 音声特性の辞書

        Returns:
            numpy.ndarray: 前処理された音声特性
        """
        # 音声特性がない場合はゼロベクトルを返す
        if audio_features is None:
            return np.zeros(10)
        
        # 利用可能な音声特性を抽出
        features = []
        
        # トーナリティ（感情的なトーン）
        if 'tonality' in audio_features:
            features.extend(audio_features['tonality'][:3])  # 最大3つの値を使用
        else:
            features.extend([0, 0, 0])
        
        # 話速
        if 'speakingRate' in audio_features:
            # 話速を正規化（100-180 wpmを0-1にスケーリング）
            rate = (audio_features['speakingRate'] - 100) / 80
            rate = max(0, min(1, rate))  # 0-1の範囲に制限
            features.append(rate)
        else:
            features.append(0.5)  # デフォルト値
        
        # 一時停止の回数
        if 'pauses' in audio_features:
            # 一時停止の平均長さ
            avg_pause = sum(audio_features['pauses']) / len(audio_features['pauses'])
            # 一時停止の数を正規化（0-10の範囲に制限）
            pause_count = min(10, len(audio_features['pauses'])) / 10
            features.extend([avg_pause, pause_count])
        else:
            features.extend([0, 0])
        
        # 話し方の明瞭さ
        if 'clarity' in audio_features:
            features.append(audio_features['clarity'])
        else:
            features.append(0.5)  # デフォルト値
        
        # 音量変動
        if 'volumeVariation' in audio_features:
            features.append(audio_features['volumeVariation'])
        else:
            features.append(0.5)  # デフォルト値
        
        # 残りのスロットをゼロで埋める
        while len(features) < 10:
            features.append(0)
        
        return np.array(features[:10])  # 最大10次元に制限
    
    def predict(self, transcript, questions, audio_features=None):
        """
        面接のフィードバックを予測する

        Args:
            transcript (str): 面接のトランスクリプト
            questions (list): 面接で尋ねられた質問のリスト
            audio_features (dict, optional): 音声特性の辞書

        Returns:
            dict: 面接フィードバック結果
        """
        # データの前処理
        transcript_embedding, questions_embedding = self._preprocess_transcript(transcript, questions)
        audio_features_processed = self._preprocess_audio_features(audio_features)
        
        # 次元拡張
        transcript_embedding = np.expand_dims(transcript_embedding, axis=0)
        questions_embedding = np.expand_dims(questions_embedding, axis=0)
        audio_features_processed = np.expand_dims(audio_features_processed, axis=0)
        
        # 予測
        overall_score, category_scores = self.model.predict([
            transcript_embedding, 
            questions_embedding, 
            audio_features_processed
        ])
        
        # スコアを0-100のスケールに変換
        overall_score_percent = float(overall_score[0][0] * 100)
        category_scores_percent = {
            self.categories[i]: float(category_scores[0][i] * 100)
            for i in range(len(self.categories))
        }
        
        # フィードバックの生成
        feedback = self._generate_feedback(category_scores_percent)
        
        return {
            'overallScore': round(overall_score_percent, 1),
            'categoryScores': {k: round(v, 1) for k, v in category_scores_percent.items()},
            'feedback': feedback
        }
    
    def _generate_feedback(self, category_scores):
        """
        カテゴリスコアに基づいてフィードバックを生成する

        Args:
            category_scores (dict): カテゴリごとのスコア

        Returns:
            dict: フィードバック情報
        """
        strengths = []
        improvements = []
        suggestions = []
        
        for category, score in category_scores.items():
            # スコアに応じたフィードバックレベルを決定
            if score >= 80:
                level = 'high'
                strengths.append(self._get_random_feedback(category, level))
            elif score >= 60:
                level = 'medium'
                if len(strengths) < 3:  # 強みは最大3つまで
                    strengths.append(self._get_random_feedback(category, level))
                else:
                    suggestions.append(self._get_random_feedback(category, level))
            else:
                level = 'low'
                improvements.append(self._get_random_feedback(category, level))
        
        # フィードバックが少ない場合は追加
        if len(strengths) == 0:
            # 最高スコアのカテゴリを強みとして追加
            top_category = max(category_scores.items(), key=lambda x: x[1])[0]
            strengths.append(self._get_random_feedback(top_category, 'medium'))
        
        if len(improvements) == 0:
            # 最低スコアのカテゴリを改善点として追加
            bottom_category = min(category_scores.items(), key=lambda x: x[1])[0]
            improvements.append(self._get_random_feedback(bottom_category, 'medium'))
        
        # 提案が少ない場合は追加
        while len(suggestions) < 2:
            # 中程度のスコアを持つカテゴリから提案を追加
            mid_categories = sorted(category_scores.items(), key=lambda x: abs(x[1] - 70))
            for cat, _ in mid_categories:
                if len(suggestions) < 2:
                    suggestion = self._get_random_feedback(cat, 'medium')
                    if suggestion not in suggestions:
                        suggestions.append(suggestion)
        
        return {
            'strengths': strengths[:3],  # 強みは最大3つ
            'improvements': improvements[:3],  # 改善点は最大3つ
            'suggestions': suggestions[:3]  # 提案は最大3つ
        }
    
    def _get_random_feedback(self, category, level):
        """
        指定されたカテゴリとレベルからランダムなフィードバックを取得する

        Args:
            category (str): フィードバックカテゴリ
            level (str): フィードバックレベル（high/medium/low）

        Returns:
            str: フィードバックメッセージ
        """
        try:
            templates = self.feedback_templates[category][level]
            return np.random.choice(templates)
        except (KeyError, IndexError):
            # テンプレートが見つからない場合はデフォルトメッセージ
            return f"{category}は{level}レベルです。"
    
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
        X_transcript = []
        X_questions = []
        X_audio = []
        y_overall = []
        y_category = []
        
        for item in train_data:
            transcript_emb, questions_emb = self._preprocess_transcript(
                item['transcript'], item['questions']
            )
            audio_features = self._preprocess_audio_features(
                item.get('audioFeatures')
            )
            
            X_transcript.append(transcript_emb)
            X_questions.append(questions_emb)
            X_audio.append(audio_features)
            
            y_overall.append(item['overallScore'] / 100.0)  # 0-1のスケールに正規化
            
            # カテゴリスコアを正規化
            category_scores = [
                item['categoryScores'].get(cat, 50) / 100.0  # デフォルト値は0.5
                for cat in self.categories
            ]
            y_category.append(category_scores)
        
        X_transcript = np.array(X_transcript)
        X_questions = np.array(X_questions)
        X_audio = np.array(X_audio)
        y_overall = np.array(y_overall)
        y_category = np.array(y_category)
        
        # 検証データの準備
        validation_data_processed = None
        if validation_data:
            X_val_transcript = []
            X_val_questions = []
            X_val_audio = []
            y_val_overall = []
            y_val_category = []
            
            for item in validation_data:
                transcript_emb, questions_emb = self._preprocess_transcript(
                    item['transcript'], item['questions']
                )
                audio_features = self._preprocess_audio_features(
                    item.get('audioFeatures')
                )
                
                X_val_transcript.append(transcript_emb)
                X_val_questions.append(questions_emb)
                X_val_audio.append(audio_features)
                
                y_val_overall.append(item['overallScore'] / 100.0)
                
                category_scores = [
                    item['categoryScores'].get(cat, 50) / 100.0
                    for cat in self.categories
                ]
                y_val_category.append(category_scores)
            
            X_val_transcript = np.array(X_val_transcript)
            X_val_questions = np.array(X_val_questions)
            X_val_audio = np.array(X_val_audio)
            y_val_overall = np.array(y_val_overall)
            y_val_category = np.array(y_val_category)
            
            validation_data_processed = (
                [X_val_transcript, X_val_questions, X_val_audio],
                [y_val_overall, y_val_category]
            )
        
        # モデルのトレーニング
        history = self.model.fit(
            [X_transcript, X_questions, X_audio],
            [y_overall, y_category],
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
            'categories': self.categories
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        logger.info(f"Model configuration saved to {config_path}")

# モジュールとして実行された場合の動作
if __name__ == "__main__":
    # サンプルデータ
    transcript = """
    面接官: あなたのデータサイエンスの経験について教えてください。
    候補者: はい、私は5年間データサイエンティストとして働いており、主に顧客セグメンテーションと需要予測に焦点を当ててきました。
    私はPythonとSQLを使用してデータパイプラインを構築し、さまざまな統計モデルや機械学習アルゴリズムを実装してきました。
    最近のプロジェクトでは、ディープラーニングを使用して時系列データから予測モデルを開発しました。
    
    面接官: チームでの作業経験はありますか？
    候補者: はい、私は通常5-7人のチームで働いており、データエンジニア、プロダクトマネージャー、そしてビジネスステークホルダーと
    緊密に連携してきました。私は協力的な環境で働くことを楽しんでおり、複雑な問題を一緒に解決することを好みます。
    また、若手データサイエンティストのメンターとしても活動しており、彼らが新しいスキルを習得するのを支援しています。
    
    面接官: 難しい問題に直面したとき、どのようにアプローチしますか？
    候補者: まず問題を完全に理解するために時間をかけ、必要な情報をすべて収集します。次に、問題を小さなパーツに分解し、
    各部分に対して解決策を考えます。私はよく問題を視覚化し、チームメンバーと協力してブレインストーミングを行い、
    さまざまなアプローチを検討します。また、時間通りに解決策を提供するためにプロジェクト管理ツールを使用して
    進捗を追跡します。最終的には、結果を評価し、将来のプロジェクトのためにフィードバックを収集します。
    """
    
    questions = [
        "あなたのデータサイエンスの経験について教えてください。",
        "チームでの作業経験はありますか？",
        "難しい問題に直面したとき、どのようにアプローチしますか？"
    ]
    
    audio_features = {
        'tonality': [0.7, 0.2, 0.1],  # ポジティブ、ニュートラル、ネガティブな感情
        'speakingRate': 145,  # 1分あたりの単語数
        'pauses': [1.2, 0.8, 2.1, 0.5],  # 一時停止の長さ（秒）
        'clarity': 0.8,  # 明瞭度（0-1）
        'volumeVariation': 0.6  # 音量変動（0-1）
    }
    
    # モデルのインスタンス化とサンプル予測
    model = InterviewFeedbackModel()
    result = model.predict(transcript, questions, audio_features)
    
    print("面接フィードバック結果:")
    print(f"全体スコア: {result['overallScore']}")
    print(f"カテゴリスコア: {json.dumps(result['categoryScores'], indent=2, ensure_ascii=False)}")
    print(f"フィードバック: {json.dumps(result['feedback'], indent=2, ensure_ascii=False)}") 