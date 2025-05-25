Hallucination-Aware Alignment (HAA) for Generative Transformers
A research-grade project implementing Hallucination-Aware Reinforcement Learning for Alignment in LLMs.

🧠 Overview
Large Language Models (LLMs) like GPT-4, Claude, and LLaMA-2 demonstrate impressive capabilities — but often generate hallucinations: outputs that are syntactically correct but factually wrong.

This project implements the Hallucination-Aware Alignment (HAA) framework proposed in our research to:

Detect hallucinations during generation,
Attribute hallucinations to specific neural circuits,
Modify activations dynamically to suppress hallucination patterns,
Reinforce factually consistent behavior through custom PPO-based Reinforcement Learning,
Employ a multi-agent system for verification and decision-making.
Our system reduces hallucination rates while maintaining model fluency and efficiency.

⚙️ Features
Factuality Detection System (retrieval-based + NLI verification)
Circuit Attribution Module (using TransformerLens)
Precision Activation Steering (PAS) (targeted neuron intervention)
Multi-Agent Architecture (Generator, Retriever, Critic, Arbiter agents)
Factuality-Aware Reinforcement Learning (PPO with dynamic reward shaping)
Adaptive Deployment Framework (light vs full intervention depending on input)
Dockerized End-to-End Deployment
