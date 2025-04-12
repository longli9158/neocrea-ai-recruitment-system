"""
候補者-職務マッチングモデル

このモデルは、候補者のスキル、経験、価値観などのプロフィールと、
職務要件・企業文化をマッチングして、最適な候補者を推薦するアルゴリズムを提供します。
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import spacy
import joblib
import os
import json
from typing import Dict, List, Tuple, Any, Union


class CandidateMatchingModel:
    """
    求職者と職務要件のマッチングを行うAIモデル
    
    このクラスでは以下の主要な機能を提供します:
    1. スキル抽出: 履歴書や職務経歴書からスキル情報を抽出
    2. 経験分析: 過去の職歴から関連する経験を評価
    3. 文化適合性: 企業文化と候補者の価値観の適合度を分析
    4. マッチングスコア算出: 総合的なマッチングスコアを計算
    """
    
    def __init__(self, model_path: str = None):
        """
        モデルの初期化
        
        Args:
            model_path: 事前訓練されたモデルの保存パス
        """
        # 言語処理モデルの読み込み
        self.nlp = spacy.load("ja_core_news_md")
        
        # 文章エンベディングモデルの読み込み
        self.sentence_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
        
        # スキル辞書の読み込み
        self.skill_keywords = self._load_skill_keywords()
        
        # 業界辞書の読み込み
        self.industry_keywords = self._load_industry_keywords()
        
        # 事前訓練モデルの読み込み（存在する場合）
        self.trained_models = {}
        if model_path and os.path.exists(model_path):
            self.trained_models = joblib.load(model_path)
    
    def _load_skill_keywords(self) -> Dict[str, List[str]]:
        """
        スキルキーワード辞書を読み込む
        
        Returns:
            スキルカテゴリをキーとし、関連キーワードのリストを値とする辞書
        """
        # 実際の実装では外部ファイルからロードする
        return {
            "プログラミング言語": ["Python", "Java", "JavaScript", "C++", "Ruby", "PHP", "Go", "Swift", "C#"],
            "フレームワーク": ["React", "Angular", "Vue.js", "Django", "Flask", "Ruby on Rails", "Spring", "TensorFlow", "PyTorch"],
            "データベース": ["MySQL", "PostgreSQL", "MongoDB", "Oracle", "SQL Server", "SQLite", "Redis", "Cassandra"],
            "クラウド": ["AWS", "GCP", "Azure", "Heroku", "Docker", "Kubernetes"],
            "ソフトスキル": ["コミュニケーション", "リーダーシップ", "問題解決", "チームワーク", "時間管理", "創造性"],
            "ビジネス": ["マーケティング", "セールス", "交渉", "プレゼンテーション", "分析", "プロジェクト管理"]
        }
    
    def _load_industry_keywords(self) -> Dict[str, List[str]]:
        """
        業界キーワード辞書を読み込む
        
        Returns:
            業界カテゴリをキーとし、関連キーワードのリストを値とする辞書
        """
        # 実際の実装では外部ファイルからロードする
        return {
            "IT": ["ソフトウェア開発", "クラウドコンピューティング", "情報セキュリティ", "ネットワーク", "データセンター"],
            "金融": ["銀行", "保険", "投資", "資産管理", "フィンテック", "証券"],
            "製造": ["自動車", "電子機器", "機械", "化学", "食品加工"],
            "医療": ["病院", "製薬", "医療機器", "バイオテクノロジー", "ヘルスケアIT"],
            "小売": ["Eコマース", "実店舗", "卸売", "サプライチェーン", "消費財"],
            "メディア": ["デジタルメディア", "放送", "出版", "広告", "エンターテイメント"],
            "教育": ["学校", "大学", "オンライン学習", "教育技術", "研修"]
        }
    
    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """
        テキストからスキル情報を抽出する
        
        Args:
            text: 履歴書や職務経歴書のテキスト
            
        Returns:
            抽出されたスキルのカテゴリ別リスト
        """
        # テキストを解析
        doc = self.nlp(text)
        
        # 抽出されたスキル
        extracted_skills = {category: [] for category in self.skill_keywords}
        
        # スキルキーワードマッチング
        for category, keywords in self.skill_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    extracted_skills[category].append(keyword)
                
        # 特殊なスキル抽出ルール（例：「〜年の経験」等のパターン）
        experience_patterns = [
            r'\d+\s*年.*経験',
            r'経験.*\d+\s*年',
        ]
        
        return extracted_skills
    
    def analyze_experience(self, work_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        職歴情報を分析する
        
        Args:
            work_history: 職歴情報のリスト
            
        Returns:
            経験の分析結果
        """
        # 合計経験年数
        total_years = sum(item.get('duration_years', 0) for item in work_history)
        
        # 業界経験
        industries = {}
        for job in work_history:
            industry = job.get('industry', '')
            if industry:
                industries[industry] = industries.get(industry, 0) + job.get('duration_years', 0)
        
        # 役職レベル
        positions = [job.get('position', '') for job in work_history]
        has_management_exp = any('マネージャー' in pos or '管理' in pos or 'リーダー' in pos for pos in positions)
        
        # プロジェクト規模
        project_scales = [job.get('project_scale', 0) for job in work_history if 'project_scale' in job]
        avg_project_scale = sum(project_scales) / len(project_scales) if project_scales else 0
        
        return {
            'total_years': total_years,
            'industries': industries,
            'has_management_exp': has_management_exp,
            'avg_project_scale': avg_project_scale
        }
    
    def calculate_culture_fit(self, 
                              candidate_values: List[str], 
                              company_values: List[str]) -> float:
        """
        候補者の価値観と企業文化の適合度を計算する
        
        Args:
            candidate_values: 候補者の価値観や優先事項のリスト
            company_values: 企業の文化的価値観のリスト
            
        Returns:
            文化適合性スコア (0.0〜1.0)
        """
        if not candidate_values or not company_values:
            return 0.0
        
        # 文章エンベディングを使用して意味的類似性を計算
        candidate_embeddings = self.sentence_model.encode(candidate_values)
        company_embeddings = self.sentence_model.encode(company_values)
        
        # 各価値観の最高類似度を計算
        max_similarities = []
        for c_emb in candidate_embeddings:
            similarities = [np.dot(c_emb, co_emb) / (np.linalg.norm(c_emb) * np.linalg.norm(co_emb)) 
                           for co_emb in company_embeddings]
            max_similarities.append(max(similarities) if similarities else 0)
        
        # 平均類似度を適合度として返す
        return sum(max_similarities) / len(max_similarities) if max_similarities else 0.0
    
    def calculate_skill_match(self, 
                              candidate_skills: Dict[str, List[str]], 
                              job_required_skills: Dict[str, List[str]]) -> Tuple[float, Dict[str, float]]:
        """
        候補者のスキルと職務要件のマッチ度を計算する
        
        Args:
            candidate_skills: 候補者のスキル（カテゴリ別）
            job_required_skills: 職務で必要なスキル（カテゴリ別）
            
        Returns:
            (全体のマッチ度スコア, カテゴリ別マッチ度)
        """
        category_scores = {}
        
        # 各カテゴリごとのマッチングスコアを計算
        for category in self.skill_keywords:
            # 候補者のスキルと職務要件の両方が存在する場合のみ
            if category in candidate_skills and category in job_required_skills:
                cand_skills = set(candidate_skills[category])
                req_skills = set(job_required_skills[category])
                
                if req_skills:  # 要件が存在する場合
                    # マッチしたスキルの割合を計算
                    matched = cand_skills.intersection(req_skills)
                    score = len(matched) / len(req_skills)
                    category_scores[category] = score
                else:
                    category_scores[category] = 0.0
            else:
                category_scores[category] = 0.0
        
        # 全体のスコアを計算（各カテゴリの平均）
        overall_score = sum(category_scores.values()) / len(category_scores) if category_scores else 0.0
        
        return overall_score, category_scores
    
    def calculate_experience_match(self, 
                                  candidate_exp: Dict[str, Any], 
                                  job_requirements: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        候補者の経験と職務要件のマッチ度を計算する
        
        Args:
            candidate_exp: 候補者の経験分析結果
            job_requirements: 職務の経験要件
            
        Returns:
            (全体のマッチ度スコア, 項目別マッチ度)
        """
        detail_scores = {}
        
        # 経験年数のマッチング
        if 'min_years' in job_requirements:
            years_score = min(1.0, candidate_exp['total_years'] / job_requirements['min_years']) if job_requirements['min_years'] > 0 else 1.0
            detail_scores['experience_years'] = years_score
        else:
            detail_scores['experience_years'] = 1.0
            
        # 業界経験のマッチング
        if 'preferred_industries' in job_requirements:
            industry_matches = []
            for ind in job_requirements['preferred_industries']:
                if ind in candidate_exp['industries']:
                    industry_matches.append(candidate_exp['industries'][ind])
                    
            industry_score = 1.0 if industry_matches else 0.0
            detail_scores['industry_match'] = industry_score
        else:
            detail_scores['industry_match'] = 1.0
            
        # マネジメント経験のマッチング
        if 'requires_management' in job_requirements:
            management_score = 1.0 if candidate_exp['has_management_exp'] == job_requirements['requires_management'] else 0.0
            detail_scores['management_match'] = management_score
        else:
            detail_scores['management_match'] = 1.0
            
        # プロジェクト規模のマッチング
        if 'min_project_scale' in job_requirements:
            scale_score = min(1.0, candidate_exp['avg_project_scale'] / job_requirements['min_project_scale']) if job_requirements['min_project_scale'] > 0 else 1.0
            detail_scores['project_scale_match'] = scale_score
        else:
            detail_scores['project_scale_match'] = 1.0
        
        # 全体のスコアを計算（各項目の平均）
        overall_score = sum(detail_scores.values()) / len(detail_scores) if detail_scores else 0.0
        
        return overall_score, detail_scores
        
    def calculate_matching_score(self, 
                                candidate_profile: Dict[str, Any], 
                                job_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        総合的なマッチングスコアを計算する
        
        Args:
            candidate_profile: 候補者情報
            job_requirements: 職務要件情報
            
        Returns:
            マッチング分析結果
        """
        # 各要素のスコアを計算
        skill_score, skill_details = self.calculate_skill_match(
            candidate_profile.get('skills', {}), 
            job_requirements.get('required_skills', {})
        )
        
        exp_score, exp_details = self.calculate_experience_match(
            candidate_profile.get('experience_analysis', {}), 
            job_requirements.get('experience_requirements', {})
        )
        
        culture_score = self.calculate_culture_fit(
            candidate_profile.get('values', []), 
            job_requirements.get('company_values', [])
        )
        
        # 総合スコアの計算（重み付け）
        weights = {
            'skills': 0.4,
            'experience': 0.3,
            'culture_fit': 0.3
        }
        
        overall_score = (
            skill_score * weights['skills'] +
            exp_score * weights['experience'] +
            culture_score * weights['culture_fit']
        )
        
        return {
            'overall_score': overall_score,
            'skill_score': skill_score,
            'experience_score': exp_score,
            'culture_fit_score': culture_score,
            'skill_details': skill_details,
            'experience_details': exp_details
        }
    
    def get_matching_recommendations(self, 
                                    candidate_profile: Dict[str, Any],
                                    job_positions: List[Dict[str, Any]],
                                    top_n: int = 5) -> List[Dict[str, Any]]:
        """
        候補者と最もマッチする職務ポジションを推薦する
        
        Args:
            candidate_profile: 候補者情報
            job_positions: 職務情報のリスト
            top_n: 上位の推薦数
            
        Returns:
            マッチングスコア付きの推薦結果
        """
        recommendations = []
        
        for job in job_positions:
            # マッチングスコアを計算
            match_result = self.calculate_matching_score(candidate_profile, job)
            
            # 推薦リストに追加
            recommendations.append({
                'job_id': job.get('job_id'),
                'job_title': job.get('job_title'),
                'company': job.get('company'),
                'matching_score': match_result['overall_score'],
                'match_details': {
                    'skill_score': match_result['skill_score'],
                    'experience_score': match_result['experience_score'],
                    'culture_fit_score': match_result['culture_fit_score']
                }
            })
        
        # マッチングスコアでソート
        recommendations.sort(key=lambda x: x['matching_score'], reverse=True)
        
        # 上位N件を返す
        return recommendations[:top_n]
    
    def save_model(self, path: str) -> None:
        """
        モデルを保存する
        
        Args:
            path: 保存先のパス
        """
        model_data = {
            'skill_keywords': self.skill_keywords,
            'industry_keywords': self.industry_keywords,
            'trained_models': self.trained_models
        }
        joblib.dump(model_data, path)
        print(f"モデルを {path} に保存しました")
        
    def load_model(self, path: str) -> None:
        """
        保存されたモデルを読み込む
        
        Args:
            path: モデルが保存されたパスs
        """
        if os.path.exists(path):
            model_data = joblib.load(path)
            self.skill_keywords = model_data.get('skill_keywords', self.skill_keywords)
            self.industry_keywords = model_data.get('industry_keywords', self.industry_keywords)
            self.trained_models = model_data.get('trained_models', {})
            print(f"モデルを {path} から読み込みました")
        else:
            print(f"モデルファイル {path} が見つかりません")


# 使用例
if __name__ == "__main__":
    # モデルのインスタンス化
    matcher = CandidateMatchingModel()
    
    # サンプル候補者プロフィール
    candidate = {
        "name": "山田太郎",
        "skills": {
            "プログラミング言語": ["Python", "JavaScript", "Java"],
            "フレームワーク": ["React", "Django", "Spring"],
            "データベース": ["MySQL", "MongoDB"],
            "クラウド": ["AWS", "Docker"],
            "ソフトスキル": ["コミュニケーション", "チームワーク"],
            "ビジネス": ["プロジェクト管理"]
        },
        "experience_analysis": {
            "total_years": 5,
            "industries": {"IT": 3, "金融": 2},
            "has_management_exp": True,
            "avg_project_scale": 8
        },
        "values": [
            "イノベーション重視", 
            "働きやすい環境", 
            "継続的な学習", 
            "チームワーク"
        ]
    }
    
    # サンプル職務要件
    job = {
        "job_id": "job123",
        "job_title": "シニアバックエンドエンジニア",
        "company": "テック株式会社",
        "required_skills": {
            "プログラミング言語": ["Python", "Java"],
            "フレームワーク": ["Django", "Spring"],
            "データベース": ["MySQL", "PostgreSQL"],
            "クラウド": ["AWS", "Docker", "Kubernetes"],
            "ソフトスキル": ["コミュニケーション", "リーダーシップ"],
            "ビジネス": ["プロジェクト管理"]
        },
        "experience_requirements": {
            "min_years": 3,
            "preferred_industries": ["IT", "金融"],
            "requires_management": True,
            "min_project_scale": 5
        },
        "company_values": [
            "イノベーション", 
            "顧客中心主義", 
            "継続的な改善", 
            "チームワーク"
        ]
    }
    
    # マッチングスコア計算
    result = matcher.calculate_matching_score(candidate, job)
    
    # 結果表示
    print(f"総合マッチングスコア: {result['overall_score']:.2f}")
    print(f"スキルマッチ: {result['skill_score']:.2f}")
    print(f"経験マッチ: {result['experience_score']:.2f}")
    print(f"文化適合性: {result['culture_fit_score']:.2f}") 