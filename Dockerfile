FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expect data/processed, models/, and reports/ to already be generated and
# committed (run `python src/build_dataset.py && python main.py` locally
# first) so the container doesn't need to retrain on every boot. If they're
# missing, generate them now:
RUN python src/build_dataset.py && python main.py || true

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
