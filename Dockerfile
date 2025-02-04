FROM python:3.10.16-slim-bullseye as builder

# Set environment variables
ENV POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache
ENV PATH="$POETRY_HOME/bin:$PATH"

# Install poetry
RUN apt-get update && apt-get install -y curl
RUN curl -sSL https://install.python-poetry.org/ | python3 -
    
# A directory to have app data 
WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry install --no-root && rm -rf $POETRY_CACHE_DIR


# The runtime image, used to just run the code provided its virtual environment
FROM python:3.10.16-slim-bullseye as runtime

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

COPY ./src ./src
COPY app.py app.py

CMD ["streamlit", "run", "app.py", "--server.port", "8051"]