FROM tensorflow/tensorflow:2.13.0-gpu

WORKDIR /workspace

RUN pip install --no-cache-dir \
    numpy \
    pandas \
    networkx \
    scipy

COPY . .

CMD ["python", "train_gcn_skip.py"]
