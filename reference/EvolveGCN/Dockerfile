FROM tensorflow/tensorflow:2.4.1-gpu

WORKDIR /workspace

RUN pip install --no-cache-dir \
    numpy==1.19.5 \
    pandas \
    networkx \
    scipy

COPY . .

CMD ["python", "train_gcn_skip.py"]
