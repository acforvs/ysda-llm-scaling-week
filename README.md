# Speeding up training with Triton and FP8 / Ускоряем обучения за счёт Triton и FP8

This repository contains materials for the lecture on FP8 & Triton, part of the short [course](https://llmscaling.yandex.com/en) on Scaling the LLM training, organized in collaboration with Yandex and [Yandex School of Data Analysis](https://dataschool.yandex.com/).


## Local setup

For materials in Russian, use the `ru/` directory. For materials in English, use the `en/` directory.


To open the notebook locally, use the following command from the root of this repo:
```bash
cd trace-viewer
npm install
npm run dev
```

Then navigate to either
- `http://localhost:5173?trace=var/traces/ru.lecture_triton_fp8.json` 
- or `http://localhost:5173?trace=var/traces/en.lecture_triton_fp8.json`, 

depending on your preferred language.


## Re-running the code (>= H100 is required)

To re-generate the traces, run:
```
python execute.py -m ru.lecture_triton_fp8
python execute.py -m en.lecture_triton_fp8
```


## Citation

If you find this content useful, consider citing it as follows:

```bibtex
@misc{LLMScalingWeekFP8Triton,
  author = {Savinov, Vladislav},
  title = {Speeding up training with Triton and FP8},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/acforvs/ysda-llm-scaling-week}},
  year = {2025}
}
```


## References

1. DeepSeek-AI. (2024). [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
2. DeepSeek-AI. [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
3. Team Cohere. (2025). [Command A: An Enterprise-Ready Large Language Model](https://arxiv.org/abs/2504.00698)
4. Micikevicius, P. et al. (2022). [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
5. OpenAI. [Triton](https://github.com/triton-lang/triton)
6. NVIDIA. [TransformerEngine](https://github.com/NVIDIA/TransformerEngine)
7. Austin et al. (2025). [How to Scale Your Model](https://jax-ml.github.io/scaling-book/), Google DeepMind
8. Modal Labs. [GPU Glossary](https://modal.com/gpu-glossary/readme)
9. Meta AI. (2025). [The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)


## Acknowledgments

I'd like to thank the team [behind CS336](https://github.com/stanford-cs336/spring2025-lectures), which was a big inspiration for how the materials are structured and for some parts of the talk.
