FROM rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35X_prealpha

RUN apt-get update && \
    apt-get install -y git openssh-client && \
    rm -rf /var/lib/apt/lists/*

# Clone Triton and install
RUN git clone https://github.com/triton-lang/triton.git /opt/triton && \
    pip install -e /opt/triton
