---
license: other
license_name: yandexgpt-5-lite-8b
license_link: LICENSE
language:
- ru
- en
base_model:
- yandex/YandexGPT-5-Lite-8B-pretrain
---

# YandexGPT-5-Lite-Instruct

Instruct-версия большой языковой модели YandexGPT 5 Lite на 8B параметров с длиной контекста 32k токенов. Также в отдельном [репозитории](https://huggingface.co/yandex/YandexGPT-5-Lite-8B-instruct-GGUF) опубликована квантизованная версия модели в формате GGUF.

Обучена на базе [YandexGPT 5 Lite Pretrain](https://huggingface.co/yandex/YandexGPT-5-Lite-8B-pretrain), без использования весов каких-либо сторонних моделей. Алайнмент Lite-версии совпадает с алайнментом YandexGPT 5 Pro и состоит из этапов SFT и RLHF (более подробно о них — в [статье](https://habr.com/ru/companies/yandex/articles/885218/) на Хабре). 

Задавайте вопросы в discussions.

## Бенчмарки
По результатам международных бенчмарков и их адаптаций для русского языка, YandexGPT 5 Lite вплотную приблизилась к аналогам (Llama-3.1-8B-instruct и Qwen-2.5-7B-instruct) и превосходит их в ряде сценариев, в том числе — в знании русской культуры и фактов. 

<img src="https://habrastorage.org/r/w1560/getpro/habr/upload_files/6b5/eb4/9ea/6b5eb49ea757bc124c938717b21f1cf7.png" alt="Таблица бенчмарков" width="100%"/>

MMLU — 5-shot, все остальные бенчмарки — 0-shot.

## Как использовать

Модель можно запустить через HF Transformers:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "yandex/YandexGPT-5-Lite-8B-instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cuda",
    torch_dtype="auto",
)

messages = [{"role": "user", "content": "Для чего нужна токенизация?"}]
input_ids = tokenizer.apply_chat_template(
    messages, tokenize=True, return_tensors="pt"
).to("cuda")

outputs = model.generate(input_ids, max_new_tokens=1024)
print(tokenizer.decode(outputs[0][input_ids.size(1) :], skip_special_tokens=True))
```

Или через vLLM:
```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


MODEL_NAME = "yandex/YandexGPT-5-Lite-8B-instruct"

sampling_params = SamplingParams(
    temperature=0.3,
    top_p=0.9,
    max_tokens=1024,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
llm = LLM(
    MODEL_NAME,
    tensor_parallel_size=1,
)

messages = [{"role": "user", "content": "В чем смысл жизни?"}]
input_ids = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True
)[1:]  # remove bos
text = tokenizer.decode(input_ids)

outputs = llm.generate(text, use_tqdm=False, sampling_params=sampling_params)

print(tokenizer.decode(outputs[0].outputs[0].token_ids, skip_special_tokens=True))

```

Для запуска в llama.cpp и ollama можно воспользоваться нашей квантизованной моделью, которая выложена в репозитории [YandexGPT-5-Lite-8B-instruct-GGUF](https://huggingface.co/yandex/YandexGPT-5-Lite-8B-instruct-GGUF).

## Особенности токенизации
Для полного соответствия токенизации мы рекомендуем пользоваться оригинальным [sentencepiece](https://github.com/google/sentencepiece) — файл токенизатора лежит в папке `original_tokenizer`. В нашей инфраструктуре каждую реплику диалога мы токенизируем отдельно. 

Из-за этого, в частности, появляется пробел в начале каждой реплики. Также `\n` токены мы заменяем на `[NL]`, это можно сделать с помощью `text.replace("\n", "[NL]")` перед токенизацией.

## Особенности шаблона
Мы используем нестандартный шаблон диалога — модель обучена генерировать только одну реплику после последовательности `Ассистент:[SEP]`, завершая её токеном `</s>`. При этом диалог в промпте может быть любой длины.

Это приводит к тому, что в интерактивном режиме модель может выдавать результаты, отличающиеся от вызова модели в режиме генерации на фиксированном диалоге. Поэтому мы рекомендуем использовать интерактивный режим только для ознакомления с моделью.