# 🌌 Aurora V7 Esencial Optimizado – Sistema Neuroacústico Modular

**Aurora V7** es un sistema avanzado de generación de audio neuroacústico terapéutico, diseñado para inducir estados mentales, emocionales y espirituales específicos mediante la síntesis dinámica de frecuencias, estructuras y texturas emocionales.

Este README documenta la estructura, funcionamiento y componentes principales de la versión **Aurora V7 Esencial Optimizado**, orientada a producción profesional y modularidad.

---

## 🧠 ¿Qué es Aurora?

Aurora es un sistema orquestado que genera pistas de audio en base a:
- Un **objetivo funcional** (ej. claridad, gratitud, enfoque, integración).
- Una **estructura emocional y progresiva** dividida en fases.
- Motores que generan sonidos neurofuncionales (neuro wave), emocionales (pads) y estéticos (ruido, paneo, filtros).

Produce archivos `.wav` estéreos que pueden usarse con audífonos o altavoces, dependiendo del objetivo.

---

## 🧩 Arquitectura Modular del Sistema

```
🌐 AURORA MASTER (Orquestador principal)
├── Aurora_Master.py

⚙️ MOTORES PRINCIPALES
├── neuromix_engine_v26_ultimate.py      → Motor neuroacústico (neuro wave, AM/FM, neurotransmisores)
├── hypermod_engine_v31.py               → Motor estructural (fases, intensidad, duración)
├── harmonicEssence_v33.py               → Motor emocional-estético (pads, ruido, estilo, paneo)

🔍 ANÁLISIS Y CALIDAD
├── Carmine_Analyzer.py                  → Análisis emocional y estructural
├── aurora_quality_pipeline.py           → Normalización, compresión y mastering
├── verify_structure.py                  → Validación técnica (bloques, capas, transiciones)

📐 CONTROL DE CAPAS
├── layer_scheduler.py                   → Activación de capas por fase
├── sync_manager.py                      → Sincronización entre motores

🎨 EMOCIÓN Y ESTILO
├── presets_emocionales_optimized.py     → Emoción ↔ pads, curvas, energía
├── presets_estilos.py                   → Estilo auditivo: filtros, textura, paneo
├── style_profiles.py                    → Perfiles de estilo: tribal, etéreo, minimalista…

🎯 PROPÓSITO Y FASES
├── objective_templates.py               → Plantillas por objetivo (claridad, integración…)
├── presets_fases.py                     → Configuración por fase
├── harmony_generator.py                 → Generación de pads armónicos

🌈 EFECTOS PSICODÉLICOS
├── psychedelic_effects_tables.json      → Fractalidad, disolución, respiración auditiva

📚 DOCUMENTACIÓN
├── aurora_v7_documentacion_extensa.pdf  → Manual técnico completo
├── README.md                            → Este archivo
```

---

## 🔄 Lógica de Funcionamiento

1. **Input del usuario:** objetivo, emoción, duración, estilo.
2. `Aurora_Master.py` consulta `objective_templates` y activa presets correctos.
3. `HyperMod` estructura las fases (ej. preparación → intención → clímax → resolución).
4. `NeuroMix` genera la capa funcional (neurotransmisores, AM/FM, pulsos).
5. `HarmonicEssence` agrega emoción y estética (pads, ruido texturizado, paneo).
6. `CarmineAnalyzer` evalúa la calidad emocional y estructural.
7. `AuroraQualityPipeline` finaliza el audio con mastering y protección auditiva.
8. Resultado: `.wav` profesional y funcional.

---

## ✅ ¿Qué genera Aurora?

- Audio estereofónico terapéutico o estimulante
- Estructurado por fases (como una historia emocional)
- Con modulación cerebral (binaural, isocrónica, AM/FM)
- Alineado a objetivos como:
  - Claridad mental
  - Gratitud
  - Regulación emocional
  - Integración psicodélica
  - Sueño profundo
  - Estado de flujo

---

## 📌 Requisitos para ejecución local

- Python 3.10 o superior
- `numpy`, `wave`, `scipy` (si se usa en versiones extendidas)
- Ejecutar `Aurora_Master.py` o integrarlo con GUI

---

## 🧭 Siguientes pasos

- Verificar si deseas integrar objetivos personalizados (`objective_profiles.py`)
- Conectar `neurotransmitter_profiles.py` a `NeuroMix` si usas lógica avanzada
- Añadir afirmaciones subliminales o sonidos ambientales si lo deseas

---

**Desarrollado con propósito terapéutico y expansión consciente.**
