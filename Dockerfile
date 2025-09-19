# =====================================================================
# Stage 1: Builder - Install dependencies
# =====================================================================
FROM python:3.10-slim-bookworm

WORKDIR /routine

# Definir a versão do Quarto para facilitar atualizações
ARG QUARTO_VERSION=1.8.24

# Instalar dependências do sistema para bibliotecas geoespaciais, Quarto e LaTeX (para PDF)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    gcc \
    git \
    wget \
    libgdal-grass \
    texlive-latex-base \
    texlive-latex-recommended \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-binaries \
    texlive-luatex \
    && rm -rf /var/lib/apt/lists/*


# Baixar e instalar o Quarto
RUN wget "https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-linux-amd64.deb" && \
    dpkg -i "quarto-${QUARTO_VERSION}-linux-amd64.deb" && \
    rm "quarto-${QUARTO_VERSION}-linux-amd64.deb"

# RUN wget "https://github.com/quarto-dev/quarto-cli/releases/download/v1.8.24/quarto-1.8.24-linux-amd64.deb" && \
#     dpkg -i "quarto-1.8.24-linux-amd64.deb" && \
#     rm "quarto-1.8.24-linux-amd64.deb"

# Install a recent version of uv and sync Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir "uv>=0.1.17"
RUN pip install papermill
RUN uv sync

# Add the virtual environment's bin directory to the PATH
# This ensures that Quarto uses the Python from the .venv
ENV PATH="/routine/.venv/bin:${PATH}"

# Copy your application's code and data into the container
COPY bacen_report.py .
COPY routine.py .
COPY bocom_bbm_report.qmd .
COPY 513438999.xml .

# Comando para executar o script Python que renderiza o documento Quarto.
CMD ["python", "routine.py"]