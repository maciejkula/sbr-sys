FROM quay.io/pypa/manylinux1_x86_64

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

ADD . .

CMD /bin/bash