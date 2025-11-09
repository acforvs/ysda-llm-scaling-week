# Speeding up training with Triton and FP8 / Ускоряем обучения за счёт Triton и FP8

This repository contains the materials for the lecture on FP8 & Triton, which was a part of the short [course](https://llmscaling.yandex.com/en) on Scaling the LLM training, ran in collaboration with Yandex and [Yandex School of Data Analysis](https://dataschool.yandex.com/).


## Local setup

For materials in Russian, use the `ru/` directory. For materials in English, use the `en/` directory.


To locally open the notebook, use the following command from the root of this repo:
```bash
cd trace-viewer
npm install
npm run dev
```

Then navigate to either
- `http://localhost:5173?trace=var/traces/ru.lecture_triton_fp8.json` 
- or `http://localhost:5173?trace=var/traces/en.lecture_triton_fp8.json`, 

depending on the language you prefer to use.


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
  howpublished = {\url{https://github.com/acforvs/ysda-llm-scaling}},
  year = {2025}
}
```
