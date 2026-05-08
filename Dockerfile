ARG BASE_IMAGE=ecr.vip.ebayc3.com/baliao/slime:a100_wandb

# ---------- Stage 0: fetch Jupyter Docker Stacks helper scripts ----------
FROM jupyter/scipy-notebook:latest AS jupyter-scripts

# ---------- Stage 1: main build ----------
FROM ${BASE_IMAGE}

# SHELL ["/bin/bash", "-lc"]

# ARG DEBIAN_FRONTEND=noninteractive
# ARG BASE_DIR=/opt
# ARG SGLANG_COMMIT=bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2
# ARG MEGATRON_COMMIT=3714d81d418c9f1bca4594fc35f9e8289f652862
# ARG DEEPEP_COMMIT=1d3963d
# ARG SLIME_REPO=https://github.com/THUDM/slime.git
# ARG SLIME_REF=
# ARG ENABLE_DEEPEP=0
# ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
# ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
# ARG FLASH_ATTN_WHEEL_URL=https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.6.8/flash_attn-2.8.3%2Bcu128torch2.9-cp312-cp312-linux_x86_64.whl
# ARG OPENSSL_VERSION=3.6.2
# ARG GO_VERSION=1.26.2
# ARG WANDB_VERSION=v0.26.0
# ARG RUST_VERSION=stable

# ENV BASE_DIR=${BASE_DIR} \
#     CUDA_HOME=/usr/local/cuda \
#     MAX_JOBS=64 \
#     NVCC_APPEND_FLAGS="--threads 32" \
#     PIP_DISABLE_PIP_VERSION_CHECK=1 \
#     PIP_NO_CACHE_DIR=1 \
#     TORCH_ALLOW_CUDA_MISMATCH=1 \
#     TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# WORKDIR ${BASE_DIR}

# RUN sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirrors.edge.kernel.org/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://mirrors.edge.kernel.org/ubuntu|g' /etc/apt/sources.list /etc/apt/sources.list.d/*.list 2>/dev/null || true && \
#     apt-get update && \
#     apt-get dist-upgrade -y && \
#     apt-get install -y --allow-change-held-packages --no-install-recommends \
#       build-essential \
#       ca-certificates \
#       cargo \
#       cmake \
#       curl \
#       git \
#       libcudnn9-cuda-12 \
#       libcudnn9-dev-cuda-12 \
#       libnccl-dev \
#       libnccl2 \
#       libnuma1 \
#       ninja-build \
#       patch \
#       pkg-config \
#       python3-dev \
#       rustc && \
#     rm -rf /var/lib/apt/lists/*

# RUN cd /tmp && \
#     curl -fsSL -o openssl.tar.gz "https://github.com/openssl/openssl/releases/download/openssl-${OPENSSL_VERSION}/openssl-${OPENSSL_VERSION}.tar.gz" && \
#     tar -xzf openssl.tar.gz && \
#     cd "openssl-${OPENSSL_VERSION}" && \
#     ./config --prefix=/opt/conda --openssldir=/opt/conda/ssl shared && \
#     make -j"$(nproc)" && \
#     make install_sw install_ssldirs && \
#     rm -rf /tmp/openssl*

# RUN python -m pip install --upgrade pip setuptools wheel && \
#     python -m pip uninstall -y \
#       xgboost \
#       transformer_engine \
#       flash_attn \
#       pynvml \
#       opencv-python-headless || true

# # Match the requested package set without creating a conda environment.
# RUN python -m pip install cuda-python==13.1.0 && \
#     python -m pip install \
#       torch==2.9.1 \
#       torchvision==0.24.1 \
#       torchaudio==2.9.1 \
#       --index-url ${TORCH_INDEX_URL} && \
#     python -m pip install cmake ninja

# RUN git clone https://github.com/sgl-project/sglang.git ${BASE_DIR}/sglang && \
#     cd ${BASE_DIR}/sglang && \
#     git checkout ${SGLANG_COMMIT} && \
#     python -m pip install -e "python[all]" && \
#     rm -rf ${BASE_DIR}/sglang/.git

# # The original script applies local patch files that are not present in this
# # build context, so this installs the upstream DeepEP sources only when enabled.
# RUN if [[ "${ENABLE_DEEPEP}" == "1" ]]; then \
#       git clone https://github.com/deepseek-ai/DeepEP.git ${BASE_DIR}/DeepEP && \
#       cd ${BASE_DIR}/DeepEP && \
#       git checkout ${DEEPEP_COMMIT} && \
#       DISABLE_SM90_FEATURES=1 python setup.py install; \
#     fi

# RUN python -m pip install "${FLASH_ATTN_WHEEL_URL}" && \
#     python -m pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps && \
#     MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=1 python -m pip install --no-build-isolation "transformer_engine[pytorch]==2.10.0" && \
#     python -m pip install flash-linear-attention==0.4.1

# RUN python -m pip install -v \
#       --disable-pip-version-check \
#       --no-build-isolation \
#       --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
#       git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4 && \
#     python -m pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@dc6876905830430b5054325fa4211ff302169c6b --force-reinstall && \
#     python -m pip install git+https://github.com/fzyzcjy/Megatron-Bridge.git@dev_rl --no-build-isolation && \
#     python -m pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation && \
#     python -m pip install https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-5f8d397/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl --force-reinstall

# RUN git clone --recursive https://github.com/NVIDIA/Megatron-LM.git ${BASE_DIR}/Megatron-LM && \
#     cd ${BASE_DIR}/Megatron-LM && \
#     git checkout ${MEGATRON_COMMIT} && \
#     git submodule update --init --recursive && \
#     python -m pip install -e . && \
#     rm -rf ${BASE_DIR}/Megatron-LM/.git

# # Install the local slime checkout so image rebuilds pick up in-repo patches.
# COPY third_party/slime ${BASE_DIR}/slime
# RUN cd ${BASE_DIR}/slime && \
#     python -m pip install -e . && \
#     rm -rf ${BASE_DIR}/slime/.git

# # slime imports wandb from its logging utilities at module import time, including
# # the HF->torch-dist conversion tool. Build wandb from source with a fixed Go
# # toolchain so the bundled wandb-core binary is rebuilt locally.
# RUN cd /tmp && \
#     curl -fsSL -o go.tgz "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" && \
#     rm -rf /usr/local/go && \
#     tar -C /usr/local -xzf go.tgz && \
#     rm -f go.tgz
# RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain ${RUST_VERSION}
# ENV PATH="/root/.cargo/bin:/usr/local/go/bin:${PATH}" \
#     WANDB_BUILD_SKIP_GPU_STATS=1 \
#     WANDB_BUILD_SKIP_ORJSON=1
# RUN python -m pip install build hatchling typing_extensions && \
#     git clone --depth 1 --branch ${WANDB_VERSION} https://github.com/wandb/wandb.git /tmp/wandb-src && \
#     cd /tmp/wandb-src && \
#     python -m build --wheel --no-isolation && \
#     python -m pip install dist/wandb-*.whl && \
#     rm -rf /tmp/wandb-src

# RUN python -m pip install nvidia-cudnn-cu12==9.16.0.29 && \
#     python -m pip install "numpy<2"

# RUN python - <<'PY'
# import torch
# print(f"torch CUDA: {torch.version.cuda}")
# import flash_attn
# print(f"flash_attn: {flash_attn.__version__}")
# try:
#     import sgl_kernel
#     print("sgl_kernel OK")
# except ImportError as exc:
#     print(f"sgl_kernel check skipped during image build: {exc}")
# import fused_weight_gradient_mlp_cuda
# print("gradient_accumulation_fusion OK")
# import amp_C
# print("amp_C OK")
# import transformer_engine
# print(f"transformer_engine: {transformer_engine.__version__}")
# PY

# CMD ["/bin/bash"]

# ---------- Jupyter support ----------

USER root

# Additional apt packages Jupyter stacks need
# Also upgrade libcap2/libcap2-bin to address CVE-2026-4878
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends --only-upgrade \
        libcap2 libcap2-bin && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        tini sudo locales fonts-dejavu tzdata bash-completion nano vim && \
    rm -rf /var/lib/apt/lists/*

# Remove stray .git directories embedded in conda package test fixtures
# (e.g. /opt/conda/etc/conda/test-files/referencing/1/suite/.git) which trip
# repository scanners.
RUN find /opt/conda -type d -name ".git" -prune -exec rm -rf {} + 2>/dev/null || true

# Locale
RUN sed -i 's/^# *\(en_US.UTF-8\)/\1/' /etc/locale.gen && locale-gen
ENV LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 LANGUAGE=en_US.UTF-8

# Jupyter Docker Stacks conventions
ENV NB_USER=jovyan \
    NB_UID=1000 \
    NB_GID=100 \
    NB_UMASK=0002 \
    JUPYTER_ENABLE_LAB=yes \
    JUPYTER_PORT=8888 \
    HOME=/home/jovyan \
    XDG_CACHE_HOME=/home/jovyan/.cache

# Create jovyan user (collision-safe)
RUN set -eux; \
    if ! getent group "${NB_GID}" >/dev/null; then \
        groupadd --gid "${NB_GID}" "${NB_USER}"; \
    fi; \
    if getent passwd "${NB_UID}" >/dev/null; then \
        new_uid="$(awk -F: '$3>=1000&&$3<65534{u[$3]=1}END{for(i=1000;i<65534;i++) if(!u[i]){print i;break}}' /etc/passwd)"; \
        echo "UID ${NB_UID} in use; creating ${NB_USER} with UID ${new_uid}"; \
        useradd --comment "Default Jupyter user" --create-home --gid "${NB_GID}" \
                --no-log-init --shell /bin/bash --uid "${new_uid}" "${NB_USER}"; \
    else \
        useradd --comment "Default Jupyter user" --create-home --gid "${NB_GID}" \
                --no-log-init --shell /bin/bash --uid "${NB_UID}" "${NB_USER}"; \
    fi; \
    mkdir -p "${HOME}/work"; \
    chown -R "$(id -u ${NB_USER}):$(id -g ${NB_USER})" "${HOME}"

# Install JupyterLab into the existing Python environment
ARG CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}
RUN set -eux; \
    if [ -x "${CONDA_DIR}/bin/conda" ]; then \
        ${CONDA_DIR}/bin/conda install -y -c conda-forge jupyterlab ipykernel ipywidgets jupyterlab_widgets && \
        ${CONDA_DIR}/bin/python -m ipykernel install --sys-prefix --name python3 --display-name "Python 3" && \
        ${CONDA_DIR}/bin/conda clean -afy; \
    else \
        python3 -m pip install --no-cache-dir jupyterlab ipykernel ipywidgets jupyterlab_widgets && \
        python3 -m ipykernel install --sys-prefix --name python3 --display-name "Python 3"; \
    fi

# Strip any .git directories pulled in by package test fixtures so security
# scanners do not flag them (e.g. CVE name_match on /**/.git).
RUN find /opt/conda /usr/local /usr/lib /usr/share -type d -name ".git" -prune -exec rm -rf {} + 2>/dev/null || true

# Bring in Jupyter Docker Stacks helper scripts
COPY --from=jupyter-scripts /usr/local/bin/start.sh              /usr/local/bin/start.sh
COPY --from=jupyter-scripts /usr/local/bin/run-hooks.sh          /usr/local/bin/run-hooks.sh
COPY --from=jupyter-scripts /usr/local/bin/start-notebook.sh     /usr/local/bin/start-notebook.sh
COPY --from=jupyter-scripts /usr/local/bin/start-singleuser.sh   /usr/local/bin/start-singleuser.sh
COPY --from=jupyter-scripts /usr/local/bin/start-notebook.py     /usr/local/bin/start-notebook.py
COPY --from=jupyter-scripts /usr/local/bin/start-singleuser.py   /usr/local/bin/start-singleuser.py
COPY --from=jupyter-scripts /usr/local/bin/fix-permissions       /usr/local/bin/fix-permissions

RUN chmod +x /usr/local/bin/start*.sh /usr/local/bin/start*.py \
              /usr/local/bin/run-hooks.sh /usr/local/bin/fix-permissions && \
    sed -i 's/\r$//' /usr/local/bin/start.sh \
                     /usr/local/bin/start-notebook.sh \
                     /usr/local/bin/start-singleuser.sh && \
    mkdir -p /usr/local/bin/start-notebook.d /usr/local/bin/before-notebook.d && \
    fix-permissions "${HOME}"

USER ${NB_USER}
WORKDIR /home/jovyan
EXPOSE 8888

ENTRYPOINT ["tini", "-g", "--"]
CMD ["/bin/bash"]

# CMD ["start-notebook.sh"]
