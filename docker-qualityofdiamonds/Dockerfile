FROM jupyter/pyspark-notebook

USER root

RUN apt-get update && \
    apt-get install -y python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install seaborn tensorflow

USER jovyan

COPY Diamonds.py .

EXPOSE 5000

CMD ["python3", "Diamonds.py"]
