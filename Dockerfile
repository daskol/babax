#FROM python:3.10

FROM daskol/ttpy

ARG UID=1000

RUN useradd -m -U -u $UID dev

WORKDIR /workspace

RUN pip install --no-cache-dir \
        git+https://github.com/AndreiChertkov/teneva.git \
        ipython \
        jax \
        jaxlib \
        pytest \
        pytest-benchmark

USER 1000
