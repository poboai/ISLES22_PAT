FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

#Initial GrandChallenge Docker Setting 
RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm
RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output
USER algorithm
WORKDIR /opt/algorithm
ENV PATH="/home/algorithm/.local/bin:${PATH}"

#Install packages
RUN python -m pip install --user -U pip
RUN pip install --upgrade pip
COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m pip install --user -rrequirements.txt

#Update Edited Files
COPY nnUNet_model /opt/algorithm/nnUNet_model

COPY --chown=algorithm:algorithm process.py /opt/algorithm/

ENTRYPOINT python -m process $0 $@
