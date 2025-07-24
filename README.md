# llm_electric_vehicle

# Project Structure for Fine-tune & Serve Small LLM Pipeline

```
fine-tune-llm-pipeline/
├── config/
│   ├── __init__.py
│   ├── config.yaml
│   └── settings.py
├── src/
│   ├── __init__.py
│   ├── data_collection/
│   │   ├── __init__.py
│   │   ├── web_scraper.py
│   │   ├── pdf_extractor.py
│   │   └── metadata_extractor.py
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── cleaner.py
│   │   ├── deduplicator.py
│   │   ├── quality_filter.py
│   │   └── tokenizer.py
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── augmentation.py
│   │   ├── qa_generator.py
│   │   └── formatter.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── fine_tuner.py
│   │   ├── lora_trainer.py
│   │   └── experiment_tracker.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── benchmark_generator.py
│   │   ├── evaluator.py
│   │   └── metrics.py
│   ├── deployment/
│   │   ├── __init__.py
│   │   ├── model_registry.py
│   │   ├── inference_server.py
│   │   └── api_endpoint.py
│   ├── orchestration/
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   └── workflow.py
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       ├── monitoring.py
│       └── storage.py
├── tests/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── scripts/
│   ├── run_pipeline.py
│   └── deploy.sh
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Key Technology Choices:
- **Framework**: Transformers + PEFT (LoRA/QLoRA)
- **Data Processing**: BeautifulSoup, PyMuPDF, Pandas
- **Training**: Accelerate + DeepSpeed for memory efficiency
- **Monitoring**: MLflow for experiment tracking
- **Deployment**: FastAPI + Uvicorn
- **Orchestration**: Apache Airflow or Prefect
- **Storage**: SQLite/PostgreSQL + S3-compatible storage