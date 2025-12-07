FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
RUN apt-get update && apt-get install -y \
    default-libmysqlclient-dev build-essential \
    && rm -rf /var/lib/apt/lists/*
    
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# COPY REQUIREMENTS TRƯỚC
COPY requirements.txt /app/

# INSTALL DEPENDENCIES (được cache)
RUN pip install -r requirements.txt

# COPY SOURCE CODE SAU (thay đổi source không làm mất cache của pip)
COPY . /app/

EXPOSE 8000
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
