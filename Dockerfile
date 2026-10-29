# Container for hosting the app as a Hugging Face Space (Docker SDK), so the site stays online
# without Colab. Free Spaces have no GPU: this installs the CPU build of PyTorch (much smaller and
# faster to build). config.DEVICE then resolves to "cpu" and the page shows its CPU-mode notice.
# For a paid GPU Space, replace the CPU index URL below with a CUDA one.
FROM python:3.12-slim

# Spaces run the container as user 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# PyTorch first from the CPU-only index, then the rest (requirements.txt's torch is then satisfied).
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

# Bake the weights into the image (pinned revision, SHA-256 checked) so startup does not download.
RUN python scripts/download_weights.py

# Spaces route traffic to port 7860 by default.
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
