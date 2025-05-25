# backend/core/factuality_detector.py
import asyncio
import logging
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    pipeline
)
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class FactualityDetector:
    """
    Comprehensive factuality detection system combining multiple approaches:
    1. Retrieval-based verification
    2. Natural Language Inference (NLI)
    3. Semantic consistency checking
    4. Knowledge graph verification
    """
    
    def __init__(self, model_name: str, retrieval_model: str, nli_model: str):
        self.model_name = model_name
        self.retrieval_model_name = retrieval_model
        self.nli_model_name = nli_model
        
        # Models
        self.retrieval_model = None
        self.nli_pipeline = None
        self.tokenizer = None
        
        # Knowledge base
        self.knowledge_embeddings = None
        self.knowledge_index = None
        self.knowledge_texts = []
        
        # Configuration
        self.similarity_threshold = 0.75
        self.nli_threshold = 0.8
        self.fact_verification_threshold = 0.7
        
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize all models and knowledge base"""
        try:
            logger.info("Initializing Factuality Detector...")
            
            # Load retrieval model
            self.retrieval_model = SentenceTransformer(self.retrieval_model_name)
            
            # Load NLI pipeline
            self.nli_pipeline = pipeline(
                "text-classification",
                model=self.nli_model_name,
                tokenizer=self.nli_model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Initialize knowledge base
            await self._initialize_knowledge_base()
            
            self.is_initialized = True
            logger.info("Factuality Detector initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Factuality Detector: {e}")
            raise
    
    async def _initialize_knowledge_base(self):
        """Initialize knowledge base with embeddings and FAISS index"""
        try:
            # Load or create knowledge base
            knowledge_path = "data/knowledge_base.json"
            
            # For demo, create a sample knowledge base
            sample_knowledge = [
                "The Earth orbits around the Sun.",
                "Water boils at 100 degrees Celsius at sea level.",
                "The capital of France is Paris.",
                "Shakespeare wrote Romeo and Juliet.",
                "The speed of light is approximately 299,792,458 meters per second.",
                "DNA stands for Deoxyribonucleic Acid.",
                "The Great Wall of China was built over many centuries.",
                "Python is a programming language created by Guido van Rossum.",
                "The human brain has approximately 86 billion neurons.",
                "Gold has the chemical symbol Au."
            ]
            
            self.knowledge_texts = sample_knowledge
            
            # Create embeddings
            logger.info("Creating knowledge base embeddings...")
            embeddings = self.retrieval_model.encode(
                self.knowledge_texts,
                show_progress_bar=True,
                convert_to_tensor=True
            )
            
            # Create FAISS index
            self.knowledge_embeddings = embeddings.cpu().numpy()
            dimension = self.knowledge_embeddings.shape[1]
            
            self.knowledge_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.knowledge_embeddings)
            self.knowledge_index.add(self.knowledge_embeddings)
            
            logger.info(f"Knowledge base initialized with {len(self.knowledge_texts)} entries")
            
        except Exception as e:
            logger.error(f"Failed to initialize knowledge base: {e}")
            raise
    
    async def detect_hallucinations(self, 
                                   input_text: str, 
                                   generated_text: str,
                                   retrieved_context: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Main hallucination detection pipeline
        """
        if not self.is_initialized:
            raise RuntimeError("Factuality Detector not initialized")
        
        results = {
            'hallucination_risk': 0.0,
            'factuality_score': 1.0,
            'verification_results': {},
            'detected_hallucinations': [],
            'confidence_score': 0.0,
            'analysis_details': {}
        }
        
        try:
            # Step 1: Retrieval-based verification
            retrieval_results = await self._retrieval_based_verification(
                input_text, generated_text
            )
            results['verification_results']['retrieval'] = retrieval_results
            
            # Step 2: NLI-based verification
            nli_results = await self._nli_based_verification(
                generated_text, retrieved_context or []
            )
            results['verification_results']['nli'] = nli_results
            
            # Step 3: Semantic consistency checking
            consistency_results = await self._semantic_consistency_check(
                input_text, generated_text
            )
            results['verification_results']['consistency'] = consistency_results
            
            # Step 4: Fact extraction and verification
            fact_results = await self._fact_extraction_verification(generated_text)
            results['verification_results']['facts'] = fact_results
            
            # Step 5: Aggregate results
            aggregated_results = self._aggregate_verification_results(
                retrieval_results, nli_results, consistency_results, fact_results
            )
            
            results.update(aggregated_results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in hallucination detection: {e}")
            results['error'] = str(e)
            return results
    
    async def _retrieval_based_verification(self, input_text: str, generated_text: str) -> Dict[str, Any]:
        """Verify generated text against retrieved knowledge"""
        
        # Encode generated text
        query_embedding = self.retrieval_model.encode([generated_text])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search knowledge base
        k = min(5, len(self.knowledge_texts))  # Top-k retrieval
        similarities, indices = self.knowledge_index.search(query_embedding, k)
        
        retrieved_facts = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if similarity > self.similarity_threshold:
                retrieved_facts.append({
                    'text': self.knowledge_texts[idx],
                    'similarity': float(similarity),
                    'rank': i + 1
                })
        
        # Calculate retrieval confidence
        max_similarity = float(similarities[0][0]) if len(similarities[0]) > 0 else 0.0
        retrieval_confidence = max_similarity
        
        return {
            'retrieved_facts': retrieved_facts,
            'max_similarity': max_similarity,
            'confidence': retrieval_confidence,
            'supports_generation': max_similarity > self.similarity_threshold
        }
    
    async def _nli_based_verification(self, generated_text: str, context: List[str]) -> Dict[str, Any]:
        """Use NLI to verify factual consistency"""
        
        if not context:
            return {
                'nli_scores': [],
                'average_entailment': 0.0,
                'confidence': 0.0,
                'consistent': False
            }
        
        nli_results = []
        
        for ctx in context:
            # Create premise-hypothesis pair
            premise = ctx
            hypothesis = generated_text
            
            # Run NLI
            result = self.nli_pipeline(f"{premise} [SEP] {hypothesis}")
            
            # Extract entailment score
            entailment_score = 0.0
            for item in result:
                if item['label'].lower() in ['entailment', 'entail']:
                    entailment_score = item['score']
                    break
            
            nli_results.append({
                'premise': premise,
                'hypothesis': hypothesis,
                'entailment_score': entailment_score,
                'prediction': result[0]['label'] if result else 'unknown'
            })
        
        # Calculate average entailment
        avg_entailment = np.mean([r['entailment_score'] for r in nli_results])
        
        return {
            'nli_scores': nli_results,
            'average_entailment': float(avg_entailment),
            'confidence': float(avg_entailment),
            'consistent': avg_entailment > self.nli_threshold
        }
    
    async def _semantic_consistency_check(self, input_text: str, generated_text: str) -> Dict[str, Any]:
        """Check semantic consistency between input and generated text"""
        
        # Encode both texts
        embeddings = self.retrieval_model.encode([input_text, generated_text])
        
        # Calculate cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        
        # Check for topic drift
        topic_drift = similarity < 0.5
        
        return {
            'semantic_similarity': float(similarity),
            'topic_drift': topic_drift,
            'consistency_score': float(similarity),
            'consistent': not topic_drift
        }
    
    async def _fact_extraction_verification(self, generated_text: str) -> Dict[str, Any]:
        """Extract and verify individual facts from generated text"""
        
        # Simple fact extraction (can be enhanced with more sophisticated NER/fact extraction)
        sentences = generated_text.split('.')
        
        fact_verifications = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
            
            # Encode sentence
            sentence_embedding = self.retrieval_model.encode([sentence])
            sentence_embedding = sentence_embedding / np.linalg.norm(sentence_embedding, axis=1, keepdims=True)
            
            # Search knowledge base
            similarities, indices = self.knowledge_index.search(sentence_embedding, 3)
            max_similarity = float(similarities[0][0]) if len(similarities[0]) > 0 else 0.0
            
            fact_verifications.append({
                'sentence': sentence,
                'verification_score': max_similarity,
                'verified': max_similarity > self.fact_verification_threshold,
                'supporting_facts': [
                    self.knowledge_texts[idx] for idx in indices[0][:2]
                ] if len(indices[0]) > 0 else []
            })
        
        # Calculate overall fact verification score
        verification_scores = [f['verification_score'] for f in fact_verifications]
        avg_verification = np.mean(verification_scores) if verification_scores else 0.0
        
        return {
            'fact_verifications': fact_verifications,
            'average_verification_score': float(avg_verification),
            'verified_facts_ratio': sum(1 for f in fact_verifications if f['verified']) / len(fact_verifications) if fact_verifications else 0.0
        }
    
    def _aggregate_verification_results(self, retrieval_results: Dict, nli_results: Dict, 
                                      consistency_results: Dict, fact_results: Dict) -> Dict[str, Any]:
        """Aggregate all verification results into final scores"""
        
        # Weight different verification methods
        weights = {
            'retrieval': 0.3,
            'nli': 0.25,
            'consistency': 0.2,
            'facts': 0.25
        }
        
        # Extract confidence scores
        retrieval_confidence = retrieval_results.get('confidence', 0.0)
        nli_confidence = nli_results.get('confidence', 0.0)
        consistency_confidence = consistency_results.get('consistency_score', 0.0)
        fact_confidence = fact_results.get('average_verification_score', 0.0)
        
        # Calculate weighted factuality score
        factuality_score = (
            weights['retrieval'] * retrieval_confidence +
            weights['nli'] * nli_confidence +
            weights['consistency'] * consistency_confidence +
            weights['facts'] * fact_confidence
        )
        
        # Calculate hallucination risk (inverse of factuality)
        hallucination_risk = 1.0 - factuality_score
        
        # Detect specific hallucinations
        detected_hallucinations = []
        
        # Flag low-confidence facts as potential hallucinations
        for fact_verification in fact_results.get('fact_verifications', []):
            if not fact_verification['verified']:
                detected_hallucinations.append({
                    'text': fact_verification['sentence'],
                    'type': 'unverified_fact',
                    'confidence': fact_verification['verification_score'],
                    'reason': 'No supporting evidence found in knowledge base'
                })
        
        # Flag inconsistent content
        if not consistency_results.get('consistent', True):
            detected_hallucinations.append({
                'text': 'Overall response consistency',
                'type': 'topic_drift',
                'confidence': consistency_results.get('semantic_similarity', 0.0),
                'reason': 'Generated text semantically inconsistent with input'
            })
        
        # Calculate overall confidence
        confidence_scores = [retrieval_confidence, nli_confidence, consistency_confidence, fact_confidence]
        overall_confidence = np.mean([score for score in confidence_scores if score > 0])
        
        return {
            'hallucination_risk': float(hallucination_risk),
            'factuality_score': float(factuality_score),
            'confidence_score': float(overall_confidence),
            'detected_hallucinations': detected_hallucinations,
            'analysis_details': {
                'retrieval_confidence': retrieval_confidence,
                'nli_confidence': nli_confidence,
                'consistency_confidence': consistency_confidence,
                'fact_confidence': fact_confidence,
                'weights_used': weights
            }
        }
    
    def is_ready(self) -> bool:
        """Check if detector is ready for use"""
        return (self.is_initialized and 
                self.retrieval_model is not None and
                self.nli_pipeline is not None and
                self.knowledge_index is not None)
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up Factuality Detector...")
        
        # Clear GPU memory if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear models
        self.retrieval_model = None
        self.nli_pipeline = None
        self.tokenizer = None
        
        logger.info("Factuality Detector cleanup complete")