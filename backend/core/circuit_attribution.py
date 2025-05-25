# backend/core/circuit_attribution.py
import asyncio
import logging
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import transformer_lens
from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.utils import get_act_name
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class CircuitComponent:
    """Represents a neural circuit component"""
    layer: int
    component_type: str  # 'attention', 'mlp', 'embed', 'unembed'
    head_idx: Optional[int] = None
    importance_score: float = 0.0
    attribution_method: str = ""
    
class CircuitAttributor:
    """
    Circuit attribution system using TransformerLens for mechanistic interpretability
    Identifies which neural circuits are responsible for hallucinations
    """
    
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # TransformerLens model
        self.model = None
        self.tokenizer = None
        
        # Circuit analysis results cache
        self.circuit_cache = {}
        
        # Configuration
        self.attribution_methods = ['attention_knockout', 'activation_patching', 'gradient_attribution']
        self.importance_threshold = 0.1
        
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize TransformerLens model"""
        try:
            logger.info(f"Initializing Circuit Attributor with model: {self.model_name}")
            
            # Load model with TransformerLens
            # Note: For demo purposes, using a smaller model that works with TransformerLens
            model_name_tl = "gpt2-small"  # TransformerLens compatible model
            
            self.model = HookedTransformer.from_pretrained(
                model_name_tl,
                device=self.device,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16
            )
            
            self.tokenizer = self.model.tokenizer
            
            # Set model to evaluation mode
            self.model.eval()
            
            self.is_initialized = True
            logger.info("Circuit Attributor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Circuit Attributor: {e}")
            raise
    
    async def attribute_hallucination_circuits(self, 
                                               input_text: str,
                                               hallucinated_text: str,
                                               factual_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Main circuit attribution pipeline to identify hallucination-causing circuits
        """
        if not self.is_initialized:
            raise RuntimeError("Circuit Attributor not initialized")
        
        try:
            # Tokenize inputs
            input_tokens = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            hallucinated_tokens = self.tokenizer.encode(hallucinated_text, return_tensors="pt").to(self.device)
            
            factual_tokens = None
            if factual_text:
                factual_tokens = self.tokenizer.encode(factual_text, return_tensors="pt").to(self.device)
            
            # Run multiple attribution methods
            attribution_results = {}
            
            # Method 1: Attention-based attribution
            attention_results = await self._attention_based_attribution(
                input_tokens, hallucinated_tokens
            )
            attribution_results['attention'] = attention_results
            
            # Method 2: Activation patching
            if factual_tokens is not None:
                patching_results = await self._activation_patching_attribution(
                    input_tokens, hallucinated_tokens, factual_tokens
                )
                attribution_results['patching'] = patching_results
            
            # Method 3: Gradient-based attribution
            gradient_results = await self._gradient_based_attribution(
                input_tokens, hallucinated_tokens
            )
            attribution_results['gradient'] = gradient_results
            
            # Method 4: Logit lens analysis
            logit_lens_results = await self._logit_lens_analysis(
                input_tokens, hallucinated_tokens
            )
            attribution_results['logit_lens'] = logit_lens_results
            
            # Aggregate results to identify key circuits
            key_circuits = self._aggregate_attribution_results(attribution_results)
            
            # Generate visualization data
            visualization_data = self._generate_visualization_data(attribution_results, key_circuits)
            
            return {
                'key_circuits': key_circuits,
                'attribution_results': attribution_results,
                'visualization_data': visualization_data,
                'summary': self._generate_attribution_summary(key_circuits),
                'confidence': self._calculate_attribution_confidence(attribution_results)
            }
            
        except Exception as e:
            logger.error(f"Error in circuit attribution: {e}")
            raise
    
    async def _attention_based_attribution(self, input_tokens: torch.Tensor, 
                                          hallucinated_tokens: torch.Tensor) -> Dict[str, Any]:
        """Analyze attention patterns to identify problematic circuits"""
        
        with torch.no_grad():
            # Run model with cache
            logits, cache = self.model.run_with_cache(hallucinated_tokens)
            
            # Analyze attention patterns
            attention_scores = {}
            
            for layer in range(self.model.cfg.n_layers):
                attention_pattern = cache[get_act_name("attn", layer)]  # [batch, head, query_pos, key_pos]
                
                # Calculate attention concentration (how focused the attention is)
                attention_entropy = self._calculate_attention_entropy(attention_pattern)
                
                # Calculate attention to input vs generated tokens
                input_length = input_tokens.shape[1]
                attention_to_input = attention_pattern[:, :, input_length:, :input_length].mean()
                attention_to_generated = attention_pattern[:, :, input_length:, input_length:].mean()
                
                for head in range(self.model.cfg.n_heads):
                    head_key = f"L{layer}H{head}"
                    attention_scores[head_key] = {
                        'layer': layer,
                        'head': head,
                        'entropy': float(attention_entropy[0, head].mean()),
                        'attention_to_input': float(attention_to_input),
                        'attention_to_generated': float(attention_to_generated),
                        'concentration_score': float(1.0 / (attention_entropy[0, head].mean() + 1e-8))
                    }
            
            # Identify anomalous attention patterns
            anomalous_heads = []
            entropy_scores = [scores['entropy'] for scores in attention_scores.values()]
            entropy_threshold = np.percentile(entropy_scores, 90)  # Top 10% most entropic
            
            for head_key, scores in attention_scores.items():
                if scores['entropy'] > entropy_threshold:
                    anomalous_heads.append({
                        'component': CircuitComponent(
                            layer=scores['layer'],
                            component_type='attention',
                            head_idx=scores['head'],
                            importance_score=scores['entropy'],
                            attribution_method='attention_entropy'
                        ),
                        'scores': scores
                    })
            
            return {
                'attention_scores': attention_scores,
                'anomalous_heads': anomalous_heads,
                'method': 'attention_based'
            }
    
    async def _activation_patching_attribution(self, input_tokens: torch.Tensor,
                                              hallucinated_tokens: torch.Tensor,
                                              factual_tokens: torch.Tensor) -> Dict[str, Any]:
        """Use activation patching to identify critical circuits"""
        
        # Get activations for both sequences
        with torch.no_grad():
            _, hallucinated_cache = self.model.run_with_cache(hallucinated_tokens)
            _, factual_cache = self.model.run_with_cache(factual_tokens)
        
        # Test patching different components
        patching_results = {}
        
        # Patch attention heads
        for layer in range(self.model.cfg.n_layers):
            for head in range(self.model.cfg.n_heads):
                # Patch this attention head
                def patch_hook(activation, hook):
                    # Replace hallucinated activations with factual ones
                    activation[:, head, :, :] = factual_cache[hook.name][:, head, :, :]
                    return activation
                
                # Run with patched activation
                with self.model.hooks([(get_act_name("attn", layer), patch_hook)]):
                    patched_logits = self.model(hallucinated_tokens)
                
                # Calculate impact of patching
                original_logits = self.model(hallucinated_tokens)
                impact_score = torch.norm(patched_logits - original_logits).item()
                
                patching_results[f"L{layer}H{head}_attn"] = {
                    'component': CircuitComponent(
                        layer=layer,
                        component_type='attention',
                        head_idx=head,
                        importance_score=impact_score,
                        attribution_method='activation_patching'
                    ),
                    'impact_score': impact_score
                }
        
        # Patch MLP layers
        for layer in range(self.model.cfg.n_layers):
            def patch_mlp_hook(activation, hook):
                return factual_cache[hook.name]
            
            with self.model.hooks([(get_act_name("mlp_out", layer), patch_mlp_hook)]):
                patched_logits = self.model(hallucinated_tokens)
            
            impact_score = torch.norm(patched_logits - original_logits).item()
            
            patching_results[f"L{layer}_mlp"] = {
                'component': CircuitComponent(
                    layer=layer,
                    component_type='mlp',
                    importance_score=impact_score,
                    attribution_method='activation_patching'
                ),
                'impact_score': impact_score
            }
        
        # Sort by impact score
        sorted_results = sorted(
            patching_results.items(),
            key=lambda x: x[1]['impact_score'],
            reverse=True
        )
        
        return {
            'patching_results': dict(sorted_results),
            'top_components': [item[1]['component'] for item in sorted_results[:10]],
            'method': 'activation_patching'
        }
    
    async def _gradient_based_attribution(self, input_tokens: torch.Tensor,
                                         hallucinated_tokens: torch.Tensor) -> Dict[str, Any]:
        """Use gradients to identify important circuits"""
        
        # Enable gradients
        hallucinated_tokens.requires_grad_(True)
        
        # Forward pass
        logits = self.model(hallucinated_tokens)
        
        # Define loss (focus on problematic tokens)
        # For demo, use the last token's logits
        target_logits = logits[0, -1, :]
        loss = -torch.log_softmax(target_logits, dim=-1).max()  # Negative log likelihood of top prediction
        
        # Backward pass
        loss.backward()
        
        # Collect gradients from different components
        gradient_attributions = {}
        
        # Get attention head gradients
        for layer in range(self.model.cfg.n_layers):
            attn_layer = self.model.blocks[layer].attn
            if hasattr(attn_layer, 'W_Q') and attn_layer.W_Q.grad is not None:
                for head in range(self.model.cfg.n_heads):
                    grad_norm = torch.norm(attn_layer.W_Q.grad[head]).item()
                    gradient_attributions[f"L{layer}H{head}_attn"] = {
                        'component': CircuitComponent(
                            layer=layer,
                            component_type='attention',
                            head_idx=head,
                            importance_score=grad_norm,
                            attribution_method='gradient_attribution'
                        ),
                        'gradient_norm': grad_norm
                    }
        
        # Get MLP gradients
        for layer in range(self.model.cfg.n_layers):
            mlp_layer = self.model.blocks[layer].mlp
            if hasattr(mlp_layer, 'W_in') and mlp_layer.W_in.grad is not None:
                grad_norm = torch.norm(mlp_layer.W_in.grad).item()
                gradient_attributions[f"L{layer}_mlp"] = {
                    'component': CircuitComponent(
                        layer=layer,
                        component_type='mlp',
                        importance_score=grad_norm,
                        attribution_method='gradient_attribution'
                    ),
                    'gradient_norm': grad_norm
                }
        
        return {
            'gradient_attributions': gradient_attributions,
            'method': 'gradient_based'
        }
    
    async def _logit_lens_analysis(self, input_tokens: torch.Tensor,
                                  hallucinated_tokens: torch.Tensor) -> Dict[str, Any]:
        """Analyze logit lens to understand prediction formation"""
        
        with torch.no_grad():
            logits, cache = self.model.run_with_cache(hallucinated_tokens)
            
            # Apply logit lens at different layers
            logit_lens_results = {}
            
            for layer in range(self.model.cfg.n_layers):
                # Get residual stream at this layer
                residual_activation = cache[get_act_name("resid_post", layer)]
                
                # Apply unembedding to get "predictions" at this layer
                layer_logits = self.model.unembed(residual_activation)
                
                # Analyze prediction changes
                final_predictions = torch.softmax(logits[0, -1, :], dim=-1)
                layer_predictions = torch.softmax(layer_logits[0, -1, :], dim=-1)
                
                # Calculate KL divergence between layer predictions and final predictions
                kl_div = torch.nn.functional.kl_div(
                    torch.log(layer_predictions + 1e-8),
                    final_predictions,
                    reduction='sum'
                ).item()
                
                logit_lens_results[f"layer_{layer}"] = {
                    'layer': layer,
                    'kl_divergence': kl_div,
                    'top_predictions': torch.topk(layer_predictions, 5).indices.tolist(),
                    'prediction_confidence': torch.max(layer_predictions).item()
                }
            
            return {
                'logit_lens_results': logit_lens_results,
                'prediction_evolution': [
                    logit_lens_results[f"layer_{layer}"] for layer in range(self.model.cfg.n_layers)
                ],
                'method': 'logit_lens'
            }
    
    def _calculate_attention_entropy(self, attention_pattern: torch.Tensor) -> torch.Tensor:
        """Calculate entropy of attention patterns"""
        # attention_pattern: [batch, head, query_pos, key_pos]
        
        # Add small epsilon to avoid log(0)
        attention_pattern = attention_pattern + 1e-8
        
        # Calculate entropy across key positions
        entropy = -torch.sum(attention_pattern * torch.log(attention_pattern), dim=-1)
        
        return entropy
    
    def _aggregate_attribution_results(self, attribution_results: Dict[str, Any]) -> List[CircuitComponent]:
        """Aggregate results from different attribution methods"""
        
        component_scores = {}
        
        # Collect scores from all methods
        for method, results in attribution_results.items():
            if method == 'attention':
                for head_data in results.get('anomalous_heads', []):
                    component = head_data['component']
                    key = f"L{component.layer}H{component.head_idx}_{component.component_type}"
                    if key not in component_scores:
                        component_scores[key] = {'component': component, 'scores': []}
                    component_scores[key]['scores'].append(component.importance_score)
            
            elif method == 'patching':
                for comp_data in results.get('top_components', []):
                    component = comp_data
                    if hasattr(component, 'head_idx') and component.head_idx is not None:
                        key = f"L{component.layer}H{component.head_idx}_{component.component_type}"
                    else:
                        key = f"L{component.layer}_{component.component_type}"
                    
                    if key not in component_scores:
                        component_scores[key] = {'component': component, 'scores': []}
                    component_scores[key]['scores'].append(component.importance_score)
            
            elif method == 'gradient':
                for comp_key, comp_data in results.get('gradient_attributions', {}).items():
                    component = comp_data['component']
                    if comp_key not in component_scores:
                        component_scores[comp_key] = {'component': component, 'scores': []}
                    component_scores[comp_key]['scores'].append(component.importance_score)
        
        # Calculate aggregated scores
        key_circuits = []
        for key, data in component_scores.items():
            # Use average score across methods
            avg_score = np.mean(data['scores'])
            component = data['component']
            component.importance_score = avg_score
            
            if avg_score > self.importance_threshold:
                key_circuits.append(component)
        
        # Sort by importance
        key_circuits.sort(key=lambda x: x.importance_score, reverse=True)
        
        return key_circuits[:20]  # Return top 20 circuits
    
    def _generate_visualization_data(self, attribution_results: Dict[str, Any], 
                                   key_circuits: List[CircuitComponent]) -> Dict[str, Any]:
        """Generate data for circuit visualization"""
        
        # Prepare data for attention pattern heatmap
        attention_data = []
        if 'attention' in attribution_results:
            for head_key, scores in attribution_results['attention']['attention_scores'].items():
                attention_data.append({
                    'layer': scores['layer'],
                    'head': scores['head'],
                    'entropy': scores['entropy'],
                    'concentration': scores['concentration_score']
                })
        
        # Prepare data for circuit importance plot
        circuit_data = []
        for circuit in key_circuits:
            circuit_data.append({
                'layer': circuit.layer,
                'component_type': circuit.component_type,
                'head_idx': circuit.head_idx,
                'importance_score': circuit.importance_score,
                'attribution_method': circuit.attribution_method
            })
        
        return {
            'attention_heatmap_data': attention_data,
            'circuit_importance_data': circuit_data,
            'layer_summary': self._generate_layer_summary(key_circuits)
        }
    
    def _generate_layer_summary(self, key_circuits: List[CircuitComponent]) -> Dict[str, Any]:
        """Generate summary by layer"""
        layer_summary = {}
        
        for circuit in key_circuits:
            layer = circuit.layer
            if layer not in layer_summary:
                layer_summary[layer] = {
                    'total_importance': 0.0,
                    'attention_heads': [],
                    'mlp_importance': 0.0,
                    'component_count': 0
                }
            
            layer_summary[layer]['total_importance'] += circuit.importance_score
            layer_summary[layer]['component_count'] += 1
            
            if circuit.component_type == 'attention':
                layer_summary[layer]['attention_heads'].append({
                    'head_idx': circuit.head_idx,
                    'importance': circuit.importance_score
                })
            elif circuit.component_type == 'mlp':
                layer_summary[layer]['mlp_importance'] = circuit.importance_score
        
        return layer_summary
    
    def _generate_attribution_summary(self, key_circuits: List[CircuitComponent]) -> Dict[str, Any]:
        """Generate human-readable summary of attribution results"""
        
        summary = {
            'total_circuits_identified': len(key_circuits),
            'most_important_circuit': None,
            'layer_distribution': {},
            'component_type_distribution': {},
            'recommendations': []
        }
        
        if key_circuits:
            # Most important circuit
            most_important = key_circuits[0]
            summary['most_important_circuit'] = {
                'layer': most_important.layer,
                'component_type': most_important.component_type,
                'head_idx': most_important.head_idx,
                'importance_score': most_important.importance_score,
                'attribution_method': most_important.attribution_method
            }
            
            # Layer distribution
            for circuit in key_circuits:
                layer = circuit.layer
                summary['layer_distribution'][layer] = summary['layer_distribution'].get(layer, 0) + 1
            
            # Component type distribution
            for circuit in key_circuits:
                comp_type = circuit.component_type
                summary['component_type_distribution'][comp_type] = \
                    summary['component_type_distribution'].get(comp_type, 0) + 1
            
            # Generate recommendations
            if summary['component_type_distribution'].get('attention', 0) > 0:
                summary['recommendations'].append(
                    "Consider applying attention head intervention on identified problematic heads"
                )
            
            if summary['component_type_distribution'].get('mlp', 0) > 0:
                summary['recommendations'].append(
                    "MLP layers show hallucination patterns - consider activation steering"
                )