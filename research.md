# runa — исследование: как написать программу для запуска AI-моделей

Дата: 8 сентября 2026. Отчёт на русском; план (`plan.md`), код и команды — на английском.

Метод: шесть параллельных Haiku-агентов собрали факты по первоисточникам (GitHub, crates.io, Hugging Face, официальная документация), три адверсариальных Haiku-агента перепроверили 130+ утверждений, синтез и решения — Fable. Всё, что не подтвердилось, помечено в §13.

---

## 1. Резюме

**Вердикт.** Программу нужно строить как тонкий, но умный слой на Rust поверх ggml/llama.cpp (C), а не как новый движок. Все двадцать с лишним аналогов, которые реально используют люди (Ollama, LM Studio, Jan, koboldcpp, LocalAI, llamafile, Lemonade), — это оболочки над llama.cpp. Самые нагруженные части (матричные ядра под NEON/SVE/SME, AVX2/AVX-512/AMX, CUDA/Metal/Vulkan) там уже написаны на C/asm под каждую платформу и обновляются ~50 сборок в неделю. Свои C/asm-ядра имеют смысл только там, где профилировщик показывает, что ggml слаб, и только если они дают ≥ 5 % на том же железе.

**Где ниша.** Ни один аналог не замыкает цикл «прогноз → запуск → замер → поправка». Оценка «влезет ли модель» есть (llama.cpp `--fit`, LM Studio `lms load --estimate-only`, gguf-parser-go, llmfit), но она либо живёт отдельно от запуска, либо не предсказывает скорость, либо не учится на реальных замерах. Ollama и вовсе молча падает на CPU (issue #14258 открыт). Единого управления «глубоким мышлением» для локальных моделей и облаков OpenAI/Anthropic тоже нет ни у кого. Аудио + видео + мышление + облако в одном Rust-бинарнике — нет ни у кого.

**Что делает runa.**
1. `runa fit <model>` — до скачивания читает заголовок GGUF по HTTP range-запросу, опрашивает железо и печатает вердикт: влезет ли (GPU / гибрид / CPU / нет), сколько памяти на каждом устройстве, прогноз скорости с доверительным интервалом, что изменить, если не влезает. После каждого запуска прогноз калибруется по замерам.
2. Три режима `cpu | gpu | hybrid` + `auto` — один планировщик размещения тензоров (слои на GPU, эксперты MoE на CPU, KV-кэш и его квантование).
3. `ThinkConfig` — выключено / включено / бюджет токенов / уровень усилий — одинаково для локальных моделей (принудительное закрытие блока рассуждений в нашем цикле семплирования) и для OpenAI (`reasoning.effort`) и Anthropic (`thinking: adaptive` + `output_config.effort`).
4. Аудио и видео — декодирование на Rust, кодировщики на C (mtmd в llama.cpp, whisper.cpp), три маршрута: нативная аудио/визуальная модель, ASR → текст, облако (с учётом того, что OpenAI не принимает видео, а Anthropic — ни аудио, ни видео).
5. CLI + TOML-конфиги + профили + OpenAI-совместимый сервер; явная политика `on_unfit = error | cpu | cloud:<model>` вместо тихого фолбэка.

Пошаговый план (P0–P7, ~70 задач с машинной проверкой каждой) — в `plan.md`.

---

## 2. Требования и как они закрываются

| # | Требование | Решение | Где в плане |
|---|-----------|---------|-------------|
| R1 | Глубокое мышление | `ThinkConfig`, парсер блоков рассуждений по семействам моделей, принудительный бюджет, маппинг на облака | D7, P3 |
| R2 | Аудио и видео | `runa-media` (symphonia/rubato/ffmpeg-sidecar) + mtmd/whisper.cpp + облачные адаптеры | D8, P4 |
| R3 | Rust + C/asm в горячих местах | ggml (C) как ядро; `runa-kernels` (C/`.S`) под benchmark-гейтом ≥ 5 % | D1, D14, P5 |
| R4 | Готовые библиотеки | llama-cpp-2, whisper-rs, sherpa-onnx, async-openai, hf-hub, sysinfo, nvml-wrapper, objc2-metal, axum, clap, figment | D2, D9, D10 |
| R5 | Три режима CPU / GPU / CPU+GPU | планировщик размещения `Placement`, `--mode` | D4, P1.8, P2 |
| R6 | Командная строка + конфиги | clap, figment (defaults < user < project < env < flags), профили | D10 |
| R7 | OpenAI / Anthropic API | trait `Backend`; async-openai (Responses API); свой тонкий клиент Anthropic (reqwest + SSE) | D9, P3.5–P3.7 |
| R8 | Проверить, запустится ли, предупредить и предсказать, как будет работать | `runa-fit`: заголовок GGUF (локально/удалённо), проба железа, аналитическая оценка + точный режим, модель скорости + калибровка, вердикт с кодами выхода | D5, D6, D12, P1 |

---

## 3. Аналоги

### 3.1. Кто есть кто (сентябрь 2026)

| Проект | Язык / движок | Версия | Режимы | Проверка «влезет ли» | Прогноз скорости | Мышление | Аудио вход | Видео вход | OpenAI/Anthropic как бэкенд | Лицензия |
|--------|---------------|--------|--------|----------------------|------------------|----------|-----------|-----------|------------------------------|----------|
| **llama.cpp** | C/C++, ggml | v0.4.0 / b10853 (сен 2026) | cpu / gpu / hybrid (`-ngl`, `-ot`, `--n-cpu-moe`, `--device`) | авто-подгонка при загрузке: `--fit on` (по умолчанию), `--fit-margin`, `--fit-target`, `llama_params_fit`, утилита `llama-fit-params` | нет | `--reasoning-budget`, `--reasoning-budget-{message,soft-ratio,grace-tokens}`, `--reasoning-format`, per-request `reasoning_budget_tokens` (PR #25961, июль 2026) | да (mtmd: Ultravox, Voxtral, Qwen2-Audio, Qwen3-ASR) | да (PR #24269, июнь 2026, через ffmpeg) | нет | MIT |
| **Ollama** | Go + ggml (MLX-бэкенд в preview с 0.19) | 0.33.2 (27 авг 2026) | авто-оффлоад слоёв | внутренняя оценка (`llm/memory.go`), но **тихий откат на CPU** (#14258 открыт) | нет | `think: true/false/low/medium/high/max` | нет | нет | свои облачные модели, не OpenAI/Anthropic | MIT |
| **LM Studio** | TS/Electron + llama.cpp, MLX | 0.4.x | GPU-offload слайдер | `lms load --estimate-only`, индикаторы в GUI | нет | вкл/выкл, парсинг блоков | нет | нет | нет | проприетарная (бесплатно) |
| **mistral.rs** | Rust (candle) | 0.9.3 (7 сен 2026) | cpu / cuda / metal, авто device-map | авто device-map | нет | бюджета нет | заявлено в README для части моделей, в релизах 0.9.3 не подтверждено | Qwen-VL | сервер отдаёт OpenAI **и** Anthropic-совместимые API (не потребляет) | MIT |
| **Jan** | TS + llama.cpp | 0.8.x | GPU-offload | предупреждения о памяти в GUI | нет | вкл/выкл | нет | нет | да (провайдеры OpenAI, Anthropic и др.) | открытая |
| **koboldcpp** | C++/Python + llama.cpp | — | cpu / gpu / hybrid | нет | нет | частично | нет (TTS/whisper отдельно) | нет | нет | AGPL-3.0 |
| **LocalAI** | Go + llama.cpp и др. | 4.x | cpu / gpu | нет | нет | частично | транскрипция (whisper) | нет | нет | MIT |
| **llamafile** | C (cosmopolitan) + llama.cpp | 0.10.x | cpu / gpu | нет | нет | нет | нет | нет | нет | Apache-2.0 |
| **vLLM** | Python/CUDA | 0.21.x | gpu (CPU-бэкенд ограничен) | «не влезло» = ошибка при старте | нет | `reasoning_effort`, бюджет мыслей (PR #37112) | да (аудио-модели) | да (VLM) | нет | Apache-2.0 |
| **SGLang** | Python/CUDA | — | gpu | ошибка при старте | нет | `--reasoning-parser`, `separate_reasoning` | да | да | нет | Apache-2.0 |
| **MLX-LM** | Python (Apple) | 0.31.x | Apple GPU | нет | нет | частично | нет | нет | нет | MIT |
| **AMD Lemonade** | Python + llama.cpp / ONNX / NPU | 10.8.0 (июн 2026) | cpu / gpu / npu (Ryzen AI) | нет | нет | частично | нет | нет | нет | Apache-2.0 |
| **Nexa SDK** | C++/Rust, свой рантайм | — | cpu / gpu / npu (Qualcomm, Apple) | нет | нет | частично | да (Qwen3-Omni) | да | нет | не подтверждена |
| **gguf-parser-go** (GPUStack) | Go, утилита | — | — | RAM/VRAM по устройствам **без скачивания** | да, «MAX TPS» через `--device-metric` | — | — | — | — | MIT |
| **llmfit** | Rust, CLI + TUI | — | — | оценка до загрузки по таблице моделей и памяти (Ollama, llama.cpp, MLX, LM Studio) | да, bandwidth-модель | — | — | — | — | открытая |
| **Kalosm / Crane** | Rust (candle) | — | cpu / gpu | нет | нет | нет | Kalosm: whisper | нет | нет | MIT/Apache |

Остальные из просмотренных (exo, GPT4All, text-generation-webui, RamaLama, llama-swap, Docker Model Runner, Foundry Local) не добавляют ничего нового по нашим восьми требованиям: это либо оболочки над llama.cpp/vLLM, либо оркестраторы переключения моделей.

### 3.2. Матрица по требованиям

| Требование | llama.cpp | Ollama | LM Studio | mistral.rs | Jan | vLLM | llmfit | **runa (цель)** |
|-----------|-----------|--------|-----------|------------|-----|------|--------|-----------------|
| Мышление: вкл/выкл | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | — | ✓ |
| Мышление: бюджет токенов | ✓ | ✗ (уровни) | ✗ | ✗ | ✗ | ✓ | — | ✓ |
| Мышление: одинаково для локальных и облачных | ✗ | частично | ✗ | ✗ | ✗ | ✗ | — | ✓ |
| Аудио вход | ✓ | ✗ | ✗ | ? | ✗ | ✓ | — | ✓ (native / ASR / cloud) |
| Видео вход | ✓ (июн 2026) | ✗ | ✗ | частично | ✗ | ✓ | — | ✓ (кадры + аудиодорожка) |
| Rust | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ | ✓ |
| C/asm-ядра под платформу | ✓ (ggml) | через ggml | через ggml | candle (Rust/CUDA) | через ggml | CUDA | — | ggml + свои под гейтом |
| cpu / gpu / hybrid | ✓ | ✓ | ✓ | ✓ | ✓ | gpu | — | ✓ + `auto` |
| CLI + конфиги | ✓ | Modelfile | GUI + `lms` | ✓ | GUI | ✓ | ✓ | ✓ TOML + профили |
| OpenAI / Anthropic как бэкенд | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | — | ✓ + `on_unfit=cloud:` |
| Проверка «влезет ли» до загрузки | при загрузке | внутри, молча | ✓ | ✗ | GUI | ✗ | ✓ | ✓ до **скачивания** |
| Прогноз скорости | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ с интервалом |
| Калибровка прогноза по замерам | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Учёт mmproj / кадров / бюджета мыслей в прогнозе | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Явная политика при «не влезает» | ошибка/подгонка | тихий CPU | ошибка | ошибка | ошибка | ошибка | — | `error \| cpu \| cloud` |

### 3.3. Боли пользователей аналогов (подтверждённые)

- **Ollama, тихий откат на CPU.** Модель «запустилась», но работает в 10 раз медленнее; предупреждения нет (issue #14258, открыт; предложение — поднять лог с debug до warn). Это главная причина, почему runa печатает строку-вердикт перед первым токеном всегда.
- **llama.cpp, память контекстных чекпоинтов.** `--ctx-checkpoints` (по умолчанию 32) накапливал RAM; чинилось в 2026 (issue #24055, PR #22929). Вывод для fit-чекера: считать не только веса и KV, а всё, что аллоцирует движок, и калибровать по логам конкретной версии.
- **llama.cpp, `--fit` не считает mmproj.** В документации `fit-params` учёт мультимодального проектора не подтверждён; при загрузке VLM/аудио-модели «подогнанная» конфигурация может не влезть. runa считает mmproj и буферы кодировщиков отдельно (P4.8).
- **Ollama MLX.** Бэкенд MLX с 0.19 (март 2026) — preview, только ≥ 32 ГБ unified memory и одна модель; ставка на ggml остаётся верной для Apple.

### 3.4. Что не делает никто (ниша runa)

1. **Замкнутый цикл fit.** Прогноз до скачивания → запуск тем же планировщиком → замер `runa bench` → поправка коэффициента эффективности для этой пары (устройство, квант). gguf-parser-go и llmfit считают, но не запускают и не учатся; llama.cpp подгоняет при загрузке, но не предсказывает скорость и не работает до скачивания.
2. **Одна ручка мышления на всё.** Бюджет/усилие одинаково применяется к Qwen3, gpt-oss, DeepSeek локально и к OpenAI/Anthropic в облаке; в прогнозе времени ответа учитывается бюджет рассуждений («2048 токенов мыслей при 50 tok/s ≈ 40 с»).
3. **Аудио + видео + облако с честной таблицей возможностей.** Если бэкенд не умеет аудио (Anthropic) — маршрут ASR → текст; если не умеет видео (оба облака) — кадры + транскрипт; `runa fit` печатает, каким маршрутом пойдёт медиа.
4. **Отсутствие тихих решений.** `on_unfit` задаётся явно; вердикт печатается всегда; коды выхода 0/1/2.

---

## 4. Движок и языки

### 4.1. Почему ggml/llama.cpp, а не свой движок и не candle/burn

| Критерий | ggml / llama.cpp | mistral.rs (candle) | candle / burn напрямую |
|----------|------------------|---------------------|------------------------|
| Бэкенды | CUDA, Metal, Vulkan, ROCm, SYCL, OpenCL/Adreno, Hexagon NPU, CANN, MUSA, OpenVINO, WebGPU, RPC | CPU (MKL/Accelerate), CUDA, Metal | candle: CPU/CUDA/Metal; burn: + Vulkan/ROCm/WebGPU через CubeCL |
| Кванты | все GGUF-форматы (Q2_K…Q8_0, IQ*, MXFP4), квантованный KV | ISQ, GGUF, GPTQ, AWQ, FP8, MXFP4 | GGUF-загрузчик есть, ядер меньше |
| CPU-ядра | NEON/i8mm/SVE, AVX2/AVX-512-VNNI/AMX, KleidiAI, ZenDNN — на C/asm | Rust + `gemm` | Rust |
| Мультимодальность | mtmd: изображения, аудио, видео (июн 2026) | vision (Qwen-VL), аудио заявлено | ограничено |
| Подгонка памяти | `--fit`, `llama_params_fit` | авто device-map | вручную |
| Спекулятивное декодирование | draft, EAGLE-3, n-gram без draft-модели | частично | нет |
| Мышление | `--reasoning-budget*`, `--reasoning-format` | нет | нет |
| Rust-обвязка | `llama-cpp-2` 0.1.133 (авг 2026): features `cuda`, `metal`, `vulkan`, `openmp`, `native`, `dynamic-link`, `mtmd`; `rig-llama-cpp` поверх | нативно | нативно |
| Лицензия | MIT | MIT | Apache-2.0 |

Решение (D2): ggml через `llama-cpp-2` — основной бэкенд; mistral.rs — опциональный (feature) для моделей, которых нет в ggml (safetensors, omni-модели без mtmd). Риск `llama-cpp-2` (отстаёт от upstream, нестабильный API) закрывается пином версии и собственным `bindgen` над `llama.h`/`mtmd.h` как запасным путём.

### 4.2. Rust и «самые нагруженные части на C/asm»

Состояние стабильного Rust 1.98.1 (20 авг 2026):

| Возможность | Статус | Следствие |
|-------------|--------|-----------|
| `asm!`, `global_asm!` | стабильно с 1.59 | ассемблерные ядра можно встраивать прямо из Rust |
| `#[unsafe(naked)]` | стабильно с 1.88 | голые функции для ручного пролога |
| `std::simd` (portable SIMD) | только nightly | переносимый SIMD на Rust в stable недоступен |
| SVE/SVE2 intrinsics | nightly (PR в stdarch апр 2026) | ARM-серверы (Graviton, Ampere) — только C/asm |
| SME/SME2 intrinsics (Apple M4+) | стадия дизайна (цель 2026) | матричные ядра для Apple M4/M5/M6 — только C/asm |
| AMX intrinsics (x86) | nightly, неполные (#126622) | Sapphire Rapids+ — только C/asm |
| `cc` crate | стабильно | компиляция `.c`/`.S` с флагами на файл |

Отсюда правило D1: ggml уже содержит эти пути (в том числе SME через KleidiAI и AMX), и «переписать горячие части на C/asm» означает не переписывать ggml, а (а) собирать его с правильными флагами под платформу и (б) добавлять свои ядра только там, где профиль показывает пробел: семплирование по словарю 150k токенов, препроцессинг изображений (resize/normalize/patchify), аудио-фронтенд (ресемплинг, мел-спектрограмма), и — как исследовательская задача — квантованный mat-vec на SME2/AMX там, где у ggml ещё нет пути под конкретную платформу. Каждое ядро: скалярная эталонная реализация, fuzz-тест эквивалентности, `criterion`-бенч, runtime-диспетчер (`is_x86_feature_detected!`, `is_aarch64_feature_detected!`, `sysctl hw.optional.arm.FEAT_SME2`). Гейт слияния — ≥ 5 % на сквозном tok/s или ≥ 2× на изолированной операции.

---

## 5. Три режима: CPU, GPU, CPU + GPU

В llama.cpp это не три режима, а один набор ручек размещения тензоров:

| Ручка | Что делает | Использование в runa |
|-------|-----------|----------------------|
| `-ngl N` / `--gpu-layers auto` | сколько слоёв на GPU | `gpu` = все, `cpu` = 0, `hybrid` = N от планировщика |
| `-ot <regex>=CPU` / `--n-cpu-moe N` | тензоры экспертов MoE на CPU | `hybrid` для MoE: внимание и общие слои на GPU, эксперты на CPU |
| `--device`, `--tensor-split` | список устройств, доля на каждом | multi-GPU (P2.9) |
| `--cache-type-k/v q8_0` | квантованный KV-кэш (требует flash attention) | `--kv q8_0` в fit и в запуске |
| `--fit`, `--fit-margin` | авто-подгонка при загрузке | «точный режим» fit-чекера для локального файла |
| `ggml_backend_sched` | распределение графа между бэкендами | внутри движка |

Почему hybrid важен именно для MoE: у Qwen3-30B-A3B или Qwen3.6-35B-A3B активны ~3 млрд параметров на токен, но веса экспертов занимают 90 % файла. Если эксперты лежат в RAM, а внимание, нормализации и роутер — на GPU, модель работает на 8 ГБ VRAM с приемлемой скоростью, потому что на токен читаются только выбранные эксперты. Планировщик runa (P1.8) кладёт эксперты на CPU в первую очередь и уже потом уменьшает число слоёв на GPU.

Скорость гибрида считается гармонически: `t_token = bytes_gpu / (eff_gpu × BW_gpu) + bytes_cpu / (eff_cpu × BW_cpu)`; узкое место — почти всегда CPU-часть.

---

## 6. Fit-чекер: «запустится ли и как быстро»

### 6.1. Как это делают аналоги

| Инструмент | Метод | До скачивания | Скорость | Обучается |
|-----------|-------|---------------|----------|-----------|
| llama.cpp `--fit` / `llama_params_fit` | пробная сборка графа аллокатором, подгонка `-ngl`, `-c`, `-ot` под свободную память минус `--fit-margin` (по умолчанию 1024 MiB) | нет | нет | нет |
| Ollama `llm/memory.go` | оценка по слоям: веса + KV + графовые буферы по семейству архитектуры | нет | нет | нет |
| LM Studio `lms load --estimate-only` | оценка памяти перед загрузкой | нет (файл локальный) | нет | нет |
| gguf-parser-go | парсит заголовок GGUF по URL, считает RAM/VRAM по устройствам, `--device-metric` → MAX TPS | **да** | да | нет |
| llmfit | таблица моделей + память машины + bandwidth-модель | да (по таблице) | да | нет |
| **runa fit** | заголовок GGUF (локально или HTTP range), проба железа, аналитическая оценка с интервалом, точный режим через аллокатор движка, калибровка | да | да, с интервалом | да |

### 6.2. Память

`total = weights + kv + compute + mmproj + margin`

**Веса** — точная сумма размеров тензоров из заголовка GGUF: `Σ n_elements(tensor) × bytes_per_block(type) / elements_per_block(type)`. Биты на вес по `ggml-common.h`:

| Тип | байт / блок | элементов / блок | бит на вес |
|-----|-------------|------------------|-----------|
| Q4_0 | 18 | 32 | 4.50 |
| Q4_1 | 20 | 32 | 5.00 |
| Q5_0 | 22 | 32 | 5.50 |
| Q8_0 | 34 | 32 | 8.50 |
| Q2_K | 84 | 256 | 2.625 |
| Q3_K | 110 | 256 | 3.4375 |
| Q4_K | 144 | 256 | 4.50 |
| Q5_K | 176 | 256 | 5.50 |
| Q6_K | 210 | 256 | 6.5625 |
| IQ4_XS | 136 | 256 | 4.25 |
| IQ4_NL | 18 | 32 | 4.50 |
| MXFP4 | 17 | 32 | 4.25 |
| F16 / BF16 | 2 | 1 | 16 |

«Q4_K_M» — это не тип, а смесь типов по тензорам (часть в Q6_K), поэтому эффективно ≈ 4.8 бит/вес; именно поэтому нужно суммировать по тензорам, а не умножать параметры на «средний bpw» (в одном из отчётов агентов bpw перепутали с гигабайтами файла — см. §13).

**KV-кэш** — `2 × n_layer × n_ctx × n_head_kv × head_dim × bytes(type)` (f16 = 2, q8_0 = 1.0625, q4_0 = 0.5625). Исключения, которые ломают формулу и должны читаться из метаданных: слои со скользящим окном (Gemma 3/4, gpt-oss: `min(n_ctx, window)`), MLA (DeepSeek V3/V4: `kv_lora_rank + rope_dim` на токен), рекуррентные/Mamba-слои (Granite 4.x, Nemotron 3: константный размер).

**Вычислительные буферы** — зависят от `n_ubatch`, `n_embd`, `n_vocab`, `n_head`, бэкенда; на практике от сотен МиБ до 1–2 ГиБ для больших словарей и батчей. Оценка «20–50 МБ» из одного из отчётов — ошибка; runa калибрует формулу по ≥ 20 логам llama.cpp на пиновой версии и добавляет 15 % (P1.5), а для локального файла берёт точное число у аллокатора.

**mmproj** — отдельный файл кодировщика (vision/audio), плюс буфер кодировщика; в `--fit` llama.cpp не учитывается (не подтверждено документацией), runa учитывает.

**Доступная память** — не «всего», а: на NVIDIA `nvmlDeviceGetMemoryInfo().free`; на Apple `recommendedMaxWorkingSetSize` (≈ 75 % unified memory по умолчанию, поднимается `sysctl iogpu.wired_limit_mb`); на CPU — `available` из `sysinfo` минус запас на систему.

### 6.3. Скорость

Декодирование одного токена ограничено пропускной способностью памяти (Капулкин, «LLM inference speed of light», 2024): каждый токен читает все активные веса и KV-кэш.

```
bytes_per_token = active_weights_bytes + kv_bytes_at(ctx/2)
decode_tok_s    = eff(device, quant) × BW_bytes_s / bytes_per_token
prefill_tok_s   = min(eff_c × FLOPS / (2 × active_params), BW-bound)
ttft_s          = prompt_tokens / prefill_tok_s
hybrid          = 1 / (bytes_gpu / (eff_gpu × BW_gpu) + bytes_cpu / (eff_cpu × BW_cpu))
```

Для MoE `active_weights = shared + n_expert_used / n_expert × expert_bytes`. Реальная эффективность `eff`: по замерам Капулкина llama.cpp достигает 58–82 % теоретической полосы RTX 4090 в зависимости от формата весов, специализированный движок — ~90 %. runa стартует с 0.60 (CUDA, Metal), 0.50 (Vulkan, CPU) и после каждого запуска обновляет медиану `measured / predicted` для пары (устройство, квант) — так прогноз сходится к ±15 % после трёх запусков (M3 в плане).

### 6.4. Железо: полоса памяти (проверено по спецификациям)

| Устройство | Память | Полоса, ГБ/с | Прогноз tok/s для 8B Q4_K_M (≈ 4.9 ГиБ), eff 0.6 |
|-----------|--------|--------------|-----------------------------------------------|
| Apple M4 | до 32 ГБ | 120 | ~14 |
| Apple M4 Pro | до 64 ГБ | 273 | ~31 |
| Apple M4 Max | до 128 ГБ | 546 | ~62 |
| Apple M5 | до 32 ГБ | 153 | ~17 |
| Apple M6 (авг 2026) | до 32 ГБ | 153 / 170 | ~19 |
| Apple M5 Ultra (авг 2026) | до 512 ГБ | 1 200 | ~137 |
| AMD Ryzen AI Max+ 395 | до 128 ГБ unified | 256 | ~29 |
| NVIDIA DGX Spark | 128 ГБ unified | 273 | ~31 |
| Snapdragon X2 Elite / Extreme | — | 152 / 228 | ~17 / ~26 |
| NVIDIA RTX 5070 | 12 ГБ | 672 | ~77 |
| NVIDIA RTX 5070 Ti | 16 ГБ | 896 | ~102 |
| NVIDIA RTX 5080 | 16 ГБ | 960 | ~110 |
| NVIDIA RTX 4090 | 24 ГБ | 1 008 | ~115 |
| NVIDIA RTX 5090 | 32 ГБ | 1 792 | ~205 |
| AMD RX 9070 XT | 16 ГБ | 640 | ~73 |
| Intel Arc B580 | 12 ГБ | 456 | ~52 |
| Десктоп DDR5-6000, 2 канала | — | ~96 | ~11 |
| Сервер EPYC, 12 каналов DDR5 | — | ~460 | ~50 |

Столбец tok/s — иллюстрация формулы, не обещание; runa всегда печатает интервал и уточняет его замерами. Полоса памяти важнее числа ядер и TFLOPS для декодирования; для prefill (обработки промпта) — наоборот.

### 6.5. Вердикт

```
Verdict  FITS · hybrid · 36/48 layers on GPU, experts on CPU · confidence: medium (estimate)
Memory   GPU 11.2 / 12.0 GiB · CPU 15.9 / 32.0 GiB
Speed    decode 9–14 tok/s · prefill 220–350 tok/s · first token (2k prompt) ≈ 7 s
Warn     decode below 15 tok/s: thinking with budget 2048 will take ~3 min per answer
Try      Q4_K_M → IQ4_XS saves 1.1 GiB (all layers on GPU, ≈ 2× faster) · --kv q8_0 saves 0.6 GiB
```

Коды выхода: 0 — влезает, 1 — влезает с предупреждениями, 2 — не влезает. `--json` для скриптов.

---

## 7. Глубокое мышление

### 7.1. Модели с рассуждениями, которые реально запускаются локально (проверено)

| Модель | Размер | Дата | Лицензия | Управление мышлением |
|--------|--------|------|----------|----------------------|
| LFM2.5-1.2B-Thinking / LFM2.5-2.6B | 1.2B / 2.6B | 2026 | LFM Open | всегда думает |
| SmolLM3-3B | 3B | июл 2025 | Apache-2.0 | `/think` / `/no_think` |
| Gemma 4 E2B / E4B / 12B / 26B-MoE / 31B | 2–31B | апр–июн 2026 | Apache-2.0 | токен `<\|think\|>` в системном промпте; явного выключателя нет |
| gpt-oss-20b / 120b | 21B-A3.6B / 117B-A5.1B | авг 2025 | Apache-2.0 | `Reasoning: low/medium/high` в системном промпте (Harmony) |
| Qwen3 4B–32B, Qwen3-30B-A3B | 4–32B | 2025 | Apache-2.0 | `enable_thinking` в шаблоне, `/no_think` |
| Qwen3.6-27B (dense), Qwen3.6-35B-A3B (MoE) | 27B / 35B-A3B | 2026 | Apache-2.0 | как Qwen3 |
| Phi-4-reasoning-vision-15B | 15B | мар 2026 | MIT | всегда думает; vision |
| Olmo 3.1 Think 7B / 32B | 7B / 32B | 2026 | Apache-2.0 | отдельные Think-варианты |
| Granite 4.2 3B / 8B / 30B | гибрид Mamba-2 | 2026 | Apache-2.0 | `thinking` в шаблоне |
| Nemotron 3 Nano-Omni | гибрид Mamba-2 + MoE | июн 2026 | NVIDIA Open | omni-вход |
| DeepSeek-V4-Flash | 284B-A13B | июл 2026 | MIT | `thinking: {type: enabled}` (API); локально ≥ 160 ГБ в Q4 |

Серверные (не для ноутбука): Qwen3.8-Max 2.4T-A95B (авг 2026, своя лицензия), DeepSeek-V4-Pro 1.6T-A49B (авг 2026, MIT), Kimi K2-Thinking / K2.5–K2.7 1T-A32B, GLM-5 / 5.1 / 5.2, Nemotron 3 Ultra 550B-A55B. Qwen3.7 и GLM-5.3 не существуют (см. §13).

### 7.2. Как управляют мышлением движки и API

| Слой | Включить / выключить | Бюджет | Уровень | Где рассуждения в ответе |
|------|----------------------|--------|---------|--------------------------|
| llama.cpp | `--chat-template-kwargs '{"enable_thinking":false}'`, `--reasoning-budget 0` | `--reasoning-budget N`, `--reasoning-budget-grace-tokens`, `--reasoning-budget-soft-ratio`, `--reasoning-budget-message`; per-request `reasoning_budget_tokens` | — | `reasoning_content` (`--reasoning-format deepseek`), `auto` определяет формат по шаблону |
| Ollama | `think: false` | — | `think: low/medium/high/max` (gpt-oss: без max) | `message.thinking` |
| vLLM | reasoning parser | бюджет мыслей (PR #37112) | `reasoning_effort` | `reasoning` / `reasoning_content` (зависит от версии) |
| SGLang | `--reasoning-parser` | — | — | `separate_reasoning` |
| OpenAI (GPT-6 Astra, GPT-5.6) | `reasoning.effort: none` (на Astra → 400) | — | `minimal/low/medium/high/xhigh/max` | Responses API: reasoning summary |
| Anthropic (Claude 5) | `thinking` не передавать | `budget_tokens` только для моделей < 4.6 (на 5-й семье → 400) | `thinking: {type: adaptive}` + `output_config.effort: low/medium/high/xhigh/max`, `display: summarized/omitted/updates` | блоки `thinking` в стриме (`thinking_delta`) |
| DeepSeek API | `thinking: {type: enabled}` | — | — | `reasoning_content` |
| Mistral API | — | — | `reasoning_effort` | — |
| OpenRouter | `reasoning.exclude` | `reasoning.max_tokens` | `reasoning.effort` | `reasoning`, `reasoning_details` |
| Gemini 3.x | — | — | `thinking_level: minimal/low/medium/high` | — |

### 7.3. Унификация в runa

```rust
pub enum ThinkMode { Off, On, Budget { tokens: u32, grace: u32 }, Effort(Effort) }
```

| ThinkMode | Локально (ggml) | OpenAI | Anthropic |
|-----------|-----------------|--------|-----------|
| `Off` | `enable_thinking=false` в шаблоне или закрытие блока на первом токене | `reasoning.effort = minimal` (или `none`, где допустимо) | без `thinking` |
| `On` | как обучена модель | `medium` | `adaptive` + `effort: medium` |
| `Budget{n}` | принудительное закрытие: считаем токены после открывающего тега, на `n − grace` смещаем логит закрывающего, на `n` вставляем его и фразу «Answer now.» | ближайший уровень по таблице | `adaptive` + ближайший `effort` (бюджеты на 5-й семье отклоняются) |
| `Effort(x)` | доли контекста: low 512, medium 2048, high 8192, max ∞ + подсказка модели (gpt-oss `Reasoning: high`) | `x` | `x` |

Вывод всегда разделён на `Event::Reasoning` и `Event::Text`; сервер отдаёт `reasoning_content`, как llama-server и DeepSeek, — это самое распространённое имя поля у клиентов. Fit-чекер учитывает бюджет мыслей во времени ответа.

Важно про Gemma 4: рассуждение задаётся токеном `<|think|>` в системном промпте, чёткого выключателя нет — для неё `Off` реализуется бюджетом 0 (немедленное закрытие блока).

---

## 8. Аудио и видео

### 8.1. Что умеют движки

| Движок | Аудио вход | Видео вход | Как |
|--------|-----------|-----------|-----|
| llama.cpp mtmd | Ultravox 0.5, Voxtral Mini 3B, Qwen2-Audio, Qwen3-ASR 0.6B/1.7B (по `docs/multimodal.md`) | да, с июня 2026 (PR #24269): `mtmd-cli --video file.mp4`, кадры через ffmpeg-подпроцесс | mmproj-файл кодировщика + основная GGUF |
| llama.cpp mtmd, vision | — | кадры как изображения | Qwen3-VL, Gemma 3/4, InternVL, MiniCPM-V, Pixtral, LLaVA |
| whisper.cpp 1.9.3 / whisper-rs 0.16 | ASR (99 языков, large-v3-turbo) | — | C, ggml; Metal/CUDA/CoreML |
| sherpa-onnx (официальная Rust-обвязка; `sherpa-rs` заархивирован 6 июня 2026) | Parakeet-TDT 0.6B v3 (25 европейских языков, включая русский, CC-BY-4.0), Canary, SenseVoice, Moonshine | — | ONNX Runtime |
| mistral.rs | заявлено (Phi-4-multimodal и др.), в релизе 0.9.3 не подтверждено | Qwen-VL | candle |
| Nexa SDK | Qwen3-Omni (omni: аудио + видео + текст) | да | свой рантайм, NPU |
| voxtral-mini-realtime-rs | Voxtral Mini 4B Realtime (ASR + TTS) | — | Burn, Vulkan/Metal/WASM |

Не поддерживаются в llama.cpp (на сентябрь 2026): аудио Qwen3-Omni, Qwen3.5-Omni (весов нет — только API), аудио Gemma 4 E2B/E4B, LFM2-Audio. Для них — маршрут ASR → текст или mistral.rs/Nexa как опциональный бэкенд.

### 8.2. Модели (проверено)

| Модель | Вход | Размер | Лицензия | Дата |
|--------|------|--------|----------|------|
| Voxtral Mini 3B 2507 / Small 24B 2507 | аудио → текст, диалог | 3B / 24B | Apache-2.0 | 2025 |
| Voxtral Mini 4B Realtime 2602 | потоковый ASR | 4B | Apache-2.0 | фев 2026 |
| Qwen3-ASR 0.6B / 1.7B | ASR | — | Apache-2.0 | 2026 |
| Parakeet-TDT 0.6B v3 | ASR, 25 языков (RU есть) | 0.6B | CC-BY-4.0 | авг 2025 |
| whisper large-v3-turbo | ASR, 99 языков | 0.8B | MIT | 2024 |
| Qwen3-Omni-30B-A3B (Instruct / Thinking) | аудио + видео + текст → текст/речь | 30B-A3B | Apache-2.0 | сен 2025 |
| Gemma 4 E2B / E4B | аудио + изображения + видео | 2B / 4B | Apache-2.0 | апр 2026 |
| Gemma 4 12B / 26B-MoE / 31B | изображения + видео (без аудио) | — | Apache-2.0 | апр–июн 2026 |
| Qwen3-VL | изображения + видео | 2B–235B | Apache-2.0 | 2025 |
| MiniCPM-o 4.5 | omni | — | открытая | фев 2026 |
| LLaVA-OneVision-2 | изображения + видео | — | не указана | апр 2026 |
| Nemotron 3 Nano-Omni | omni | гибрид | NVIDIA Open | июн 2026 |
| Phi-4-multimodal | аудио + изображения | 5.6B | MIT | 2025 |

### 8.3. Маршруты в runa

```
audio ──▶ decode (symphonia/hound) ──▶ 16 kHz mono f32 (rubato)
   ├─ native : model has audio mmproj ─▶ mtmd audio chunk ─▶ LLM
   ├─ asr    : whisper.cpp | parakeet ─▶ transcript ─▶ LLM (any model, any cloud)
   └─ cloud  : OpenAI input_audio (gpt-audio) | Anthropic: transcript only

video ──▶ ffmpeg-sidecar ──▶ frames @1 fps, ≤ 32, scene-change keyframes ──▶ VLM (mtmd) or cloud images
      └─▶ audio track ──▶ audio route above
```

Ограничения облаков (проверено по документации на сентябрь 2026): OpenAI — изображения, PDF, `input_audio` для аудио-моделей, **видео нет**; Anthropic — изображения и PDF, **ни аудио, ни видео**; OpenAI-совместимый слой Anthropic не поддерживает PDF и аудио и урезает thinking — использовать только нативный Messages API. Gemini через OpenAI-совместимый endpoint — изображения и аудио, видео нет.

Rust-крейты (актуальные версии): `cpal` 0.18.2 (захват), `symphonia` 0.5.5, `hound`, `rubato` 3.0.0, `ffmpeg-sidecar` 2.5.2 (бинарник ffmpeg, не линкуется — нет проблем с GPL), `rsmpeg` 0.18 (если понадобится линковка), `image`, `fast_image_resize`, `whisper-rs` 0.16.0.

---

## 9. Облачные API из Rust

| Провайдер | Актуальные модели и цены ($/1M in / out) | Мышление | Мультимодальность | Крейт |
|----------|-------------------------------------------|----------|-------------------|-------|
| Anthropic | `claude-fable-5-1` 10 / 50 (1M контекст), `claude-opus-5` 5 / 25, `claude-sonnet-5` 2 / 10, `claude-haiku-4-5` 1 / 5 | `thinking: {type: adaptive}` + `output_config.effort`; `budget_tokens` только < 4.6 | изображения, PDF, Files API; стриминг SSE; `stop_reason: refusal`; batches −50 %; prompt caching | официального Rust SDK нет → тонкий `reqwest` + `eventsource-stream` (D9); альтернативы: `adk-anthropic` 2.2.0 (adaptive thinking, effort, Files), `misanthropic`, `anthropic-sdk-rust` |
| OpenAI | GPT-6 Astra 10 / 50; GPT-5.6 Sol / Terra / Luna; Responses API (рекомендуется), Chat Completions поддерживается, Assistants закрыт 26 авг 2026 | `reasoning.effort: none…max` | изображения, PDF, `input_audio` (gpt-audio); Realtime API; видео нет | `async-openai` 0.41.3 (июл 2026): стриминг, аудио, Realtime |
| OpenAI-совместимые (OpenRouter, DeepSeek, Groq, Gemini, llama-server, runa serve) | — | `reasoning`, `reasoning_effort`, `thinking` — нормализуется адаптером | зависит | тот же `async-openai` с `base_url` |
| Мульти-провайдерные обёртки | — | частично | частично | `genai` 0.6 (май 2026), `rig-core` 0.42 (авг 2026) — не дают полного контроля над thinking, поэтому не используются как основа |

Секреты — только из `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` или системного keychain; конфиг с ключом внутри отклоняется.

---

## 10. Архитектура runa (кратко)

```
runa (CLI, config, server)
 ├─ runa-core     Backend trait · Request/Event · ThinkConfig · Mode
 ├─ runa-fit      gguf header (local/HTTP range) · hw probe · estimator · planner · calibration db
 ├─ runa-engine   llama-cpp-2: load(Placement) · sampling loop · budget forcing · mtmd · state cache
 ├─ runa-media    symphonia/rubato/ffmpeg-sidecar · frame sampling · whisper-rs / sherpa-onnx
 ├─ runa-cloud    openai (async-openai, Responses) · anthropic (reqwest + SSE) · price table
 └─ runa-kernels  C/.S kernels · cc build · runtime dispatch · reference impls · criterion gate
```

Поток `runa run`: конфиг → ссылка на модель → `runa-fit` (вердикт, `Placement`) → политика `on_unfit` → бэкенд (локальный или облачный) → поток событий `Reasoning`/`Text` → вывод; после завершения `runa-fit` записывает замер в БД калибровки.

Подробно: `plan.md` (D1–D16, семь крейтов, метрики M1–M10, фазы P0–P7).

---

## 11. Порядок работ и почему такой

1. **P0 (1 нед.)** — скелет, CI на трёх ОС, спайки, которые проверяют рискованные допущения (`llama-cpp-2` + Metal/CUDA, `mtmd`), базовые замеры `llama-bench` для трёх эталонных моделей в трёх режимах.
2. **P1 (2–3 нед.) — fit-чекер раньше движка**, потому что это отличие от аналогов, и его можно разрабатывать без GPU: парсер GGUF, HTTP range, дескриптор модели, оценка KV с исключениями, буферы, проба железа, планировщик, модель скорости, калибровка, вердикт.
3. **P2 (2–3 нед.)** — движок, три режима, `auto`, MoE-гибрид, квантованный KV, кэш промптов, `runa bench`.
4. **P3 (2 нед.)** — мышление (парсер, бюджет, уровни) и облака (OpenAI, Anthropic, маршрутизация, сервер v1).
5. **P4 (2–3 нед.)** — аудио (ASR, нативный), видео (кадры), облачные ограничения, учёт медиа в fit.
6. **P5 (2–3 нед., с жёстким лимитом времени)** — профилирование, `runa-kernels` с гейтом ≥ 5 %, спекулятивное декодирование, perf-CI.
7. **P6 (2 нед.)** — сервер (мульти-модель, Anthropic-совместимый endpoint), упаковка `cargo-dist`, Homebrew, документация, релиз 1.0.
8. **P7** — mistral.rs как feature, RPC-распределение, tool calling, NPU, рекомендации моделей под машину.

Итого ≈ 14–18 недель до 1.0 одним разработчиком с агентами; каждая задача плана имеет проверку, которую можно выполнить машиной.

---

## 12. Целевое сравнение: runa против аналогов

| Что | Ollama | LM Studio | llama.cpp | llmfit | **runa** |
|-----|--------|-----------|-----------|--------|----------|
| Проверка до скачивания | нет | нет | нет | по таблице | по заголовку GGUF |
| Прогноз скорости | нет | нет | нет | да | да, интервал + калибровка |
| Что делать, если не влезает | тихий CPU | ошибка | подгонка | — | список: другой квант, ctx, KV, облако |
| Мышление локально + облако | уровни / свои облака | вкл/выкл | бюджет | — | одна ручка на всё |
| Аудио / видео | нет | нет | да | — | да + маршруты + учёт в fit |
| OpenAI / Anthropic как бэкенд | нет | нет | нет | — | да + `on_unfit=cloud` |
| Язык | Go | TS | C++ | Rust | Rust + C/asm |

---

## 13. Fact-check: что не подтвердилось и что исправлено

| # | Утверждение из отчётов агентов | Вердикт | Факт |
|---|-------------------------------|---------|------|
| 1 | Биты на вес: Q4_0 4.34, Q8_0 8.0, Q4_K_M 4.58 | опровергнуто | агент спутал ГБ файла с bpw; точные значения по `ggml-common.h` в §6.2 (Q4_0 4.5, Q8_0 8.5, Q4_K 4.5) |
| 2 | Вычислительные буферы «20–50 МБ» | опровергнуто | сотни МиБ – ГиБ; калибруется по логам |
| 3 | Ryzen AI Max+ 395 ≈ 96 ГБ/с | опровергнуто | 256 ГБ/с (LPDDR5X-8000, 256 бит) |
| 4 | RTX 5070 Ti и RTX 5070 ≈ 576 ГБ/с | опровергнуто | 896 и 672 ГБ/с |
| 5 | Поле llama-server `thinking_budget_tokens` | опровергнуто | `reasoning_budget_tokens` (PR #25961) |
| 6 | llama.cpp «v0.27+» | опровергнуто | семантические теги есть, но последний — v0.4.0 (4 сен 2026); сборки b10853 |
| 7 | mtmd не умеет видео | опровергнуто | видео добавлено PR #24269 (8 июн 2026), через ffmpeg |
| 8 | mistral.rs 0.8.2, Apache-2.0 | опровергнуто | 0.9.3 (7 сен 2026), MIT |
| 9 | mistral.rs поддерживает аудио/видео нативно | не подтверждено | в релиз-нотах 0.9.3 нет; в README заявлено — считать «частично» |
| 10 | Ollama MLX — основной бэкенд на Apple | опровергнуто | preview с 0.19 (30 мар 2026), только ≥ 32 ГБ, одна модель |
| 11 | Qwen3.6-35B — dense, text-only | опровергнуто | Qwen3.6-35B-A3B — MoE; есть dense Qwen3.6-27B |
| 12 | Kimi K2 ≈ 13B | опровергнуто | 1T-A32B |
| 13 | Qwen3.7, GLM-5.3-Flash | не найдены | линейки: Qwen3.5 / 3.6 / 3.8; GLM-5 / 5.1 / 5.2 |
| 14 | Qwen3.8-Max — CC-BY-NC | опровергнуто | своя лицензия «qwen3.8-max» |
| 15 | DeepSeek V4 «Think High/Max» | не найдено | режимы не документированы |
| 16 | Gemma 4 12B в апрельском релизе | частично | 12B добавлен 3 июн 2026; апрель: E2B, E4B, 26B-MoE, 31B |
| 17 | Gemma 4: аудио во всех размерах; выключатель мышления | опровергнуто | аудио только E2B/E4B; мышление токеном `<\|think\|>`, выключателя нет |
| 18 | Qwen3-Omni — 2026, проприетарная; ggml-org выложил GGUF с аудио | опровергнуто | Qwen3-Omni — сен 2025, Apache-2.0; Qwen3.5-Omni (мар 2026) — только API; официальных GGUF с аудио-кодировщиком нет, llama.cpp не поддерживает |
| 19 | sherpa-rs 0.6.8 — актуальная обвязка | опровергнуто | заархивирован 6 июн 2026; использовать официальную Rust-обвязку sherpa-onnx |
| 20 | rubato 0.19/0.20, symphonia 0.17 | опровергнуто | rubato 3.0.0 (май 2026), symphonia 0.5.5 (окт 2025) |
| 21 | Nexa AI куплена Qualcomm | не подтверждено | основной репозиторий NexaAI/nexa-sdk, у Qualcomm — форк |
| 22 | Lemonade: 55 tok/s Qwen3.5-35B-A3B на Ryzen AI Max+ | не подтверждено | AMD публикует ~24.5 tok/s для Qwen3.8-27B; 61 tok/s на iGPU для другой модели |
| 23 | OpenAI: актуальны o1/o3/o4-mini | опровергнуто | GPT-6 Astra, GPT-5.6 Sol/Terra/Luna |
| 24 | OpenAI принимает видео | опровергнуто | только изображения/PDF/аудио |
| 25 | Anthropic OpenAI-совместимый слой полноценен | опровергнуто | нет PDF, нет аудио, thinking урезан — только нативный API |
| 26 | Mistral: параметр `prompt_mode` | опровергнуто | `reasoning_effort` |
| 27 | reqwest-eventsource заброшен | опровергнуто | 0.6.0 (30 авг 2026); годится, как и eventsource-stream 0.2.3 |
| 28 | `adk-anthropic` не существует | опровергнуто (моё сомнение) | 2.2.0 (1 сен 2026): adaptive thinking, effort, Files API |
| 29 | Canary-1B-v2 = те же языки и лицензия, что Parakeet | опровергнуто | отдельная модель, покрытие и лицензия не совпадают |
| 30 | mmproj учитывается в `--fit` | не подтверждено | документация fit-params не упоминает; runa считает сам |
| 31 | Точные размеры блоков K-квантов | не подтверждено веб-агентом | взяты из `ggml-common.h` (block_q4_K = 144 байт и т. д.); проверить в P1.3 тестами на реальных файлах |
| 32 | vLLM переименовал `reasoning_content` → `reasoning` | частично | зависит от версии; адаптер нормализует оба |

Подтверждено (выборочно): llama.cpp `--fit`/`--fit-margin`/`--fit-target`/`llama_params_fit`/`llama-fit-params`, `--fit on` по умолчанию; все флаги `--reasoning-budget*`; Ollama 0.33.2 и issue #14258; `llama-cpp-2` feature `mtmd`; `rig-llama-cpp`; llmfit; gguf-parser-go MAX TPS; `lms load --estimate-only`; M5 Ultra 1.2 ТБ/с и 512 ГБ; M6 153/170 ГБ/с; DGX Spark 273 ГБ/с; Voxtral 3B/4B/24B Apache-2.0; Parakeet v3 с русским; whisper-rs 0.16; async-openai 0.41.3; цены и параметры Anthropic (из актуальной документации API).

---

## 14. Источники

Движки и обвязки
- llama.cpp: https://github.com/ggml-org/llama.cpp — `tools/fit-params`, `tools/server/README.md`, `docs/multimodal.md`, `docs/speculative.md`, `docs/build.md`, `common/reasoning-budget.cpp`; PR #25961 (reasoning budget), PR #24269 (video), PR #11607 (reasoning-format), issue #24055 / PR #22929 (ctx-checkpoints)
- llama-cpp-2: https://crates.io/crates/llama-cpp-2 · https://github.com/utilityai/llama-cpp-rs
- rig-llama-cpp: https://crates.io/crates/rig-llama-cpp
- mistral.rs: https://github.com/EricLBuehler/mistral.rs/releases/tag/v0.9.3
- candle: https://github.com/huggingface/candle · burn: https://github.com/tracel-ai/burn
- whisper.cpp / whisper-rs: https://github.com/ggml-org/whisper.cpp · https://crates.io/crates/whisper-rs
- sherpa-onnx: https://github.com/k2-fsa/sherpa-onnx · sherpa-rs (архив): https://github.com/thewh1teagle/sherpa-rs
- Nexa SDK: https://github.com/NexaAI/nexa-sdk
- voxtral-mini-realtime-rs: https://github.com/TrevorS/voxtral-mini-realtime-rs

Раннеры и оценщики
- Ollama: https://github.com/ollama/ollama/releases/tag/v0.33.2 · https://github.com/ollama/ollama/issues/14258 · https://ollama.com/blog/mlx · https://docs.ollama.com/capabilities/thinking
- LM Studio: https://lmstudio.ai/docs/cli/local-models/load
- gguf-parser-go: https://github.com/gpustack/gguf-parser-go
- llmfit: https://github.com/AlexsJones/llmfit
- AMD Lemonade: https://www.phoronix.com/news/Lemonade-SDK-10.2-Released · https://www.amd.com/en/blogs/2026/run-qwen-3-8-27b-on-amd-ryzen-ai-max-and-radeon-graphics-cards-day-0.html
- vLLM: https://docs.vllm.ai/en/latest/features/reasoning_outputs/ · https://github.com/vllm-project/vllm/pull/37112

Модели
- Qwen: https://huggingface.co/Qwen/Qwen3.6-35B-A3B · https://huggingface.co/Qwen/Qwen3.6-27B · https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B · https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct
- Gemma 4: https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/ · https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12b/
- gpt-oss: https://huggingface.co/openai/gpt-oss-20b
- DeepSeek V4: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro · https://api-docs.deepseek.com/guides/thinking_mode/
- Kimi: https://huggingface.co/moonshotai/Kimi-K2-Thinking · GLM: https://huggingface.co/collections/zai-org/glm-52
- Nemotron 3: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 · https://huggingface.co/blog/nvidia/nemotron-3-nano-omni-multimodal-intelligence
- Liquid: https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking · Granite 4.2: https://www.ibm.com/granite/docs/models/granite4-2 · Olmo 3.1: https://huggingface.co/allenai/Olmo-3.1-32B-Think · Phi-4-reasoning-vision: https://huggingface.co/microsoft/Phi-4-reasoning-vision-15B · SmolLM3: https://huggingface.co/HuggingFaceTB/SmolLM3-3B
- Voxtral: https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602 · Parakeet: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3 · MiniCPM-o: https://github.com/OpenBMB/MiniCPM-V · LLaVA-OneVision-2: https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-2

Облачные API
- Anthropic: https://platform.claude.com/docs/en/build-with-claude/thinking · https://platform.claude.com/docs/en/build-with-claude/effort · https://platform.claude.com/docs/en/api/openai-sdk · https://platform.claude.com/docs/en/build-with-claude/working-with-messages
- OpenAI: https://developers.openai.com/api/docs/models · https://developers.openai.com/api/docs/guides/reasoning · https://developers.openai.com/api/docs/guides/audio · https://developers.openai.com/api/docs/guides/migrate-to-responses
- Mistral: https://docs.mistral.ai/capabilities/reasoning · OpenRouter: https://openrouter.ai/docs/docs/best-practices/reasoning-tokens · Gemini: https://ai.google.dev/gemini-api/docs/generate-content/thinking · https://ai.google.dev/gemini-api/docs/openai
- Крейты: https://docs.rs/crate/async-openai/latest · https://docs.rs/adk-anthropic · https://docs.rs/reqwest-eventsource · https://docs.rs/eventsource-stream · https://github.com/jeremychone/rust-genai · https://docs.rs/crate/rig-core/latest

Железо и физика скорости
- Kapoulkine, «LLM inference speed of light»: https://zeux.io/2024/03/15/llm-inference-sol/
- Apple M6 / M5 Ultra: https://www.apple.com/newsroom/2026/08/apple-introduces-m6-and-m5-ultra-for-a-big-leap-in-performance-and-ai-compute/
- AMD Ryzen AI Max+ 395: https://www.amd.com/en/products/processors/laptop/ryzen-pro/ai-max-pro-300-series/amd-ryzen-ai-max-plus-pro-395.html
- Intel Arc B580: https://www.intel.com/content/www/us/en/products/sku/241598/intel-arc-b580-graphics/specifications.html
- RTX 50: https://www.techspot.com/news/106565-nvidia-reveals-complete-geforce-rtx-5070-rtx-5070.html · RX 9070 XT: https://www.techspot.com/review/2961-amd-radeon-9070-xt/ · Snapdragon X2: https://www.cnx-software.com/2025/10/02/snapdragon-x2-elite-extreme-and-x2-elite-processors-target-high-end-windows-pcs/

Rust
- Rust 1.98: https://blog.rust-lang.org/2026/08/20/Rust-1.98.0/ · SVE/SME цель 2026: https://rust-lang.github.io/rust-project-goals/2026/scalable-vectors.html · AMX: https://github.com/rust-lang/rust/issues/126622 · portable SIMD: https://github.com/rust-lang/portable-simd
- GPU: https://crates.io/crates/cudarc · https://docs.rs/objc2-metal · https://crates.io/crates/ash · https://wgpu.rs/
- Медиа: https://crates.io/crates/symphonia · https://crates.io/crates/rubato · https://crates.io/crates/ffmpeg-sidecar · https://crates.io/crates/cpal · https://crates.io/crates/rsmpeg
