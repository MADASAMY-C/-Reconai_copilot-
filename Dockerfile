FROM python:3.11-slim
RUN useradd -m -u 1000 user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    DATABASE_URL=sqlite:////home/user/reconai.db \
    LLM_PROVIDER=local
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
USER user
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM as M; n='Qwen/Qwen2.5-0.5B-Instruct'; AutoTokenizer.from_pretrained(n); M.from_pretrained(n)"
COPY --chown=user . .
EXPOSE 7860
CMD ["streamlit","run","app.py","--server.port=7860","--server.address=0.0.0.0","--server.enableXsrfProtection=false","--server.enableCORS=false"]