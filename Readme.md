# Deprofile

This repository contains the reference implementation and data schema for **Deprofile**,  
a data-grounded framework for mental patient simulation with longitudinal evidence.

The project focuses on constructing unified patient profiles by aligning **assessment dialogues**,  
**counseling interactions**, and **longitudinal social media records**, and enabling temporally grounded simulation with explicit factual constraints.

---

## Overview

Existing patient simulators often rely on snapshot-style prompts with limited profile information,  
which leads to homogeneous behaviors and incoherent symptom progression.

Deprofile addresses this limitation by:
- Integrating multi-source real-world data into a unified patient profile
- Explicitly modeling symptom attributes and life-event timelines
- Enforcing temporal and factual constraints during generation to reduce hallucination

The framework is designed for **research and evaluation purposes only** and is **not intended for real clinical use**.

---

## Repository Structure

---


---

## Unified Patient Profile

Each patient profile is constructed by aligning information from three sources:

1. **Assessment Dialogue Data**  
   - Provides standardized clinical symptom attributes  
   - Used as the primary source of diagnostic grounding

2. **Counseling Dialogue Data**  
   - Captures conversational style and affective response patterns  
   - Used for personality and interaction modeling (style-only)

3. **Social Media Data**  
   - Provides longitudinal life-event and symptom expressions  
   - Used to construct temporally grounded experience timelines

A two-stage matching process ensures demographic consistency and symptom compatibility across sources.

---

## Symptom Attribute Design

Symptom attributes are represented using a unified tag space:

- **Assessment-derived attributes**  
  - English translations of clinically grounded symptom labels  
  - Non-italic formatting in documentation

- **Social-media-derived attributes**  
  - Original English labels  
  - Italic formatting used to distinguish data source

Explicit paired mappings are defined for overlapping attributes across modalities.

---

## Chain-of-Change (CoC) Agent

The Chain-of-Change (CoC) agent converts noisy longitudinal records into structured memory representations:

1. Extracts symptom and event nodes from raw timelines  
2. Builds a temporally ordered graph with persistence relations  
3. Aggregates nodes into episodic memory cards with relative timestamps  

During simulation, responses referencing past events are constrained to retrieved memory cards,  
preventing unsupported or fabricated details.

---

## Evaluation

The framework supports both automatic and LLM-based evaluation:

- **Embedding-based metrics**
  - Semantic realism
  - Inter-patient diversity

- **LLM-as-a-Judge (G-Eval)**
  - Persona faithfulness
  - Event richness
  - Symptom consistency

Ablation settings allow controlled analysis of individual profile components.

---

## Usage Notes

- This repository is intended for **research reproducibility and analysis**
- The data and simulators **must not** be used for real-world diagnosis or treatment
- All profiles are de-identified and processed for research purposes only

---

## License

This repository is released for **academic research use only**.  
See the license file (if provided) for details.

---

## Disclaimer

This project simulates patient behavior for research and evaluation.  
It does **not** provide medical advice, diagnosis, or treatment recommendations.
