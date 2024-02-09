# syntax=docker/dockerfile:1
FROM rayproject/ray-ml:2.9.0-py310-gpu
COPY requirements.txt .
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# RUN wget https://github.com/protocolbuffers/protobuf/releases/download/v21.12/protobuf-all-21.12.tar.gz
# RUN tar -xvf protobuf-all-21.12.tar.gz
# RUN cd protobuf-21.12
# RUN ./configure
# RUN make
# RUN make install
# RUN sudo ldconfig
RUN pip install -r requirements.txt
