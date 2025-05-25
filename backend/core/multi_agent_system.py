# backend/core/multi_agent_system.py
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
import torch
import numpy as np
from datetime import datetime
import json

from .factuality_detector import FactualityDetector
from .circuit_attribution import CircuitAttributor
from .activation_steering import ActivationSteering
from .rl_trainer import HAARLTrainer
from .deployment_framework import AdaptiveDeployment
from ..agents.generator_agent import GeneratorAgent
from ..agents.retrieval_agent import RetrievalAgent
from ..agents.critic_agent import CriticAgent
from ..agents.arbiter_agent import ArbiterAgent
from ..utils.evaluation_metrics import HAA_Metrics

logger = logging.getLogger(__name__)

class MultiAgentSystem:
    """
    Core HAA Multi-Agent System orchestrating all components
    """
    
    def __init__(self, config):
        self.config = config
        self.is_initialized = False
        self.system_metrics = HAA_Metrics()
        
        # Core components
        self.factuality_detector = None
        self.circuit_attributor = None
        self.activation_steering = None
        self.rl_trainer = None
        self.deployment_framework = None
        
        # Agents
        self.generator_agent = None
        self.retrieval_agent = None
        self.critic_agent = None
        self.arbiter_agent = None
        
        # System state
        self.active_sessions = {}
        self.intervention_history = []
        self.performance_metrics = {
            'hallucination_rate': 0.0,
            'factuality_score': 0.0,
            'fluency_score': 0.0,
            'intervention_accuracy': 0.0
        }
        
    async def initialize(self):
        """Initialize all system components"""
        try:
            logger.info("Initializing HAA Multi-Agent System...")
            
            # Initialize core components
            await self._init_core_components()
            
            # Initialize agents
            await self._init_agents()
            
            # Setup inter-agent communication
            await self._setup_communication()
            
            # Initialize deployment framework
            self.deployment_framework = AdaptiveDeployment(
                factuality_detector=self.factuality_detector,
                circuit_attributor=self.circuit_attributor,
                activation_steering=self.activation_steering
            )
            
            self.is_initialized = True
            logger.info("HAA System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize HAA System: {e}")
            raise
    
    async def _init_core_components(self):
        """Initialize core HAA components"""
        # Factuality Detection System
        self.factuality_detector = FactualityDetector(
            model_name=self.config.BASE_MODEL,
            retrieval_model=self.config.RETRIEVAL_MODEL,
            nli_model=self.config.NLI_MODEL
        )
        await self.factuality_detector.initialize()
        
        # Circuit Attribution Module
        self.circuit_attributor = CircuitAttributor(
            model_name=self.config.BASE_MODEL,
            device=self.config.DEVICE
        )
        await self.circuit_attributor.initialize()
        
        # Activation Steering
        self.activation_steering = ActivationSteering(
            model=self.circuit_attributor.model,
            attributor=self.circuit_attributor
        )
        
        # RL Trainer
        self.rl_trainer = HAARLTrainer(
            model=self.circuit_attributor.model,
            factuality_detector=self.factuality_detector,
            config=self.config
        )
        
    async def _init_agents(self):
        """Initialize all agents"""
        # Generator Agent
        self.generator_agent = GeneratorAgent(
            model=self.circuit_attributor.model,
            activation_steering=self.activation_steering
        )
        
        # Retrieval Agent
        self.retrieval_agent = RetrievalAgent(
            retrieval_model=self.config.RETRIEVAL_MODEL,
            knowledge_base_path=self.config.KNOWLEDGE_BASE_PATH
        )
        await self.retrieval_agent.initialize()
        
        # Critic Agent
        self.critic_agent = CriticAgent(
            factuality_detector=self.factuality_detector,
            circuit_attributor=self.circuit_attributor
        )
        
        # Arbiter Agent
        self.arbiter_agent = ArbiterAgent(
            deployment_framework=self.deployment_framework
        )
        
    async def _setup_communication(self):
        """Setup inter-agent communication channels"""
        # Create communication channels between agents
        self.communication_channels = {
            'generator_to_critic': asyncio.Queue(),
            'critic_to_arbiter': asyncio.Queue(),
            'arbiter_to_generator': asyncio.Queue(),
            'retrieval_to_critic': asyncio.Queue()
        }
        
    async def process_input(self, input_text: str, session_id: str = None) -> Dict[str, Any]:
        """
        Main processing pipeline for HAA system
        """
        if not self.is_initialized:
            raise RuntimeError("HAA System not initialized")
            
        session_id = session_id or f"session_{datetime.now().timestamp()}"
        
        try:
            # Step 1: Determine deployment strategy
            deployment_strategy = await self.deployment_framework.determine_strategy(input_text)
            
            # Step 2: Generate initial response
            initial_response = await self.generator_agent.generate(
                input_text, 
                intervention_level=deployment_strategy['intervention_level']
            )
            
            # Step 3: Retrieve relevant context
            retrieved_context = await self.retrieval_agent.retrieve(input_text)
            
            # Step 4: Critic evaluation
            critic_analysis = await self.critic_agent.evaluate(
                input_text=input_text,
                generated_text=initial_response['text'],
                retrieved_context=retrieved_context
            )
            
            # Step 5: Arbiter decision
            arbiter_decision = await self.arbiter_agent.decide(
                input_text=input_text,
                initial_response=initial_response,
                critic_analysis=critic_analysis,
                deployment_strategy=deployment_strategy
            )
            
            # Step 6: Apply interventions if needed
            final_response = await self._apply_interventions(
                initial_response,
                arbiter_decision,
                session_id
            )
            
            # Step 7: Update system metrics
            await self._update_metrics(
                input_text=input_text,
                initial_response=initial_response,
                final_response=final_response,
                critic_analysis=critic_analysis,
                arbiter_decision=arbiter_decision
            )
            
            return {
                'session_id': session_id,
                'input_text': input_text,
                'response': final_response['text'],
                'confidence_score': final_response['confidence'],
                'hallucination_risk': critic_analysis['hallucination_risk'],
                'interventions_applied': final_response['interventions'],
                'factuality_score': critic_analysis['factuality_score'],
                'retrieval_context': retrieved_context,
                'processing_time': final_response.get('processing_time', 0),
                'deployment_strategy': deployment_strategy
            }
            
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            raise
    
    async def _apply_interventions(self, initial_response: Dict, arbiter_decision: Dict, session_id: str) -> Dict:
        """Apply interventions based on arbiter decision"""
        
        if not arbiter_decision['requires_intervention']:
            return {
                **initial_response,
                'interventions': []
            }
        
        interventions_applied = []
        current_response = initial_response
        
        for intervention in arbiter_decision['recommended_interventions']:
            if intervention['type'] == 'activation_steering':
                # Apply activation steering
                steered_response = await self.activation_steering.apply_steering(
                    input_text=current_response['input_text'],
                    target_circuits=intervention['target_circuits'],
                    steering_strength=intervention['strength']
                )
                current_response = steered_response
                interventions_applied.append(intervention)
                
            elif intervention['type'] == 'regeneration':
                # Regenerate with different parameters
                regenerated_response = await self.generator_agent.generate(
                    current_response['input_text'],
                    intervention_level='high',
                    avoid_patterns=intervention.get('avoid_patterns', [])
                )
                current_response = regenerated_response
                interventions_applied.append(intervention)
                
            elif intervention['type'] == 'fact_augmentation':
                # Augment with retrieved facts
                augmented_response = await self._augment_with_facts(
                    current_response,
                    intervention['facts']
                )
                current_response = augmented_response
                interventions_applied.append(intervention)
        
        # Record intervention history
        self.intervention_history.append({
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'interventions': interventions_applied,
            'initial_risk': arbiter_decision['initial_risk'],
            'final_risk': current_response.get('hallucination_risk', 0)
        })
        
        return {
            **current_response,
            'interventions': interventions_applied
        }
    
    async def _augment_with_facts(self, response: Dict, facts: List[str]) -> Dict:
        """Augment response with retrieved facts"""
        augmented_text = response['text']
        
        # Simple fact augmentation - can be made more sophisticated
        if facts:
            fact_context = "\n\n[Relevant Facts: " + "; ".join(facts) + "]"
            augmented_text += fact_context
        
        return {
            **response,
            'text': augmented_text,
            'augmented_with_facts': True
        }
    
    async def _update_metrics(self, **kwargs):
        """Update system performance metrics"""
        # Calculate hallucination rate
        if 'critic_analysis' in kwargs:
            hallucination_risk = kwargs['critic_analysis']['hallucination_risk']
            self.system_metrics.update_hallucination_rate(hallucination_risk)
        
        # Update factuality score
        if 'critic_analysis' in kwargs:
            factuality_score = kwargs['critic_analysis']['factuality_score']
            self.system_metrics.update_factuality_score(factuality_score)
        
        # Update intervention accuracy
        if 'arbiter_decision' in kwargs and 'final_response' in kwargs:
            intervention_needed = kwargs['arbiter_decision']['requires_intervention']
            intervention_applied = len(kwargs['final_response'].get('interventions', [])) > 0
            self.system_metrics.update_intervention_accuracy(intervention_needed, intervention_applied)
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        return {
            'performance_metrics': self.performance_metrics,
            'system_health': {
                'is_initialized': self.is_initialized,
                'active_sessions': len(self.active_sessions),
                'total_interventions': len(self.intervention_history)
            },
            'component_status': {
                'factuality_detector': self.factuality_detector.is_ready() if self.factuality_detector else False,
                'circuit_attributor': self.circuit_attributor.is_ready() if self.circuit_attributor else False,
                'generator_agent': self.generator_agent.is_ready() if self.generator_agent else False,
                'retrieval_agent': self.retrieval_agent.is_ready() if self.retrieval_agent else False,
                'critic_agent': self.critic_agent.is_ready() if self.critic_agent else False,
                'arbiter_agent': self.arbiter_agent.is_ready() if self.arbiter_agent else False
            },
            'recent_interventions': self.intervention_history[-10:] if self.intervention_history else []
        }
    
    def is_ready(self) -> bool:
        """Check if system is ready for processing"""
        return self.is_initialized and all([
            self.factuality_detector and self.factuality_detector.is_ready(),
            self.circuit_attributor and self.circuit_attributor.is_ready(),
            self.generator_agent and self.generator_agent.is_ready(),
            self.retrieval_agent and self.retrieval_agent.is_ready(),
            self.critic_agent and self.critic_agent.is_ready(),
            self.arbiter_agent and self.arbiter_agent.is_ready()
        ])
    
    async def cleanup(self):
        """Cleanup system resources"""
        logger.info("Cleaning up HAA System...")
        
        # Cleanup agents
        for agent in [self.generator_agent, self.retrieval_agent, self.critic_agent, self.arbiter_agent]:
            if agent and hasattr(agent, 'cleanup'):
                await agent.cleanup()
        
        # Cleanup core components
        if self.factuality_detector:
            await self.factuality_detector.cleanup()
        
        if self.circuit_attributor:
            await self.circuit_attributor.cleanup()
        
        logger.info("HAA System cleanup complete")