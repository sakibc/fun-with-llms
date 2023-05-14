#!/usr/bin/env bash

uvicorn llm.hosted_model_server:app --host 0.0.0.0
