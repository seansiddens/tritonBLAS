FROM rocm/pytorch:latest

RUN apt-get update && \
    apt-get install -y git openssh-client && \
    rm -rf /var/lib/apt/lists/*

# Clone Triton and install
RUN git clone https://github.com/triton-lang/triton.git /opt/triton && \
    cd /opt/triton && git checkout 71e794323fab8f0b1bc0280ae95 && \
    pip install -e /opt/triton
