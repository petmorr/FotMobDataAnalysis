FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY fotmob_analytics ./fotmob_analytics
COPY app.py .
COPY .streamlit ./.streamlit
RUN pip install --no-cache-dir --no-deps .

# Run as an unprivileged user; the FotMob response cache lives in its home.
RUN useradd --create-home --shell /usr/sbin/nologin appuser
USER appuser

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
