FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# libgomp is required by the FAISS CPU 
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# The default Linux PyTorch wheel now brings CUDA dependencies. The app uses
# CPU embeddings, so install the lean CPU wheel explicitly first.
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 \
    && pip install -r requirements.txt

COPY . ./

EXPOSE 8501

CMD ["streamlit", "run", "App.py", "--server.address=0.0.0.0", "--server.port=8501"]
