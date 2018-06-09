import os
import shutil
import subprocess
import pathlib


def build(build_dir="build", features="", cpu_features="avx2"):

    with open("env.list", "w") as envfile:
        envfile.write(
            "RUSTFLAGS=-C target-feature=+{cpu_features}\n".format(cpu_features=cpu_features)
        )

        # Make OpenBLAS compilation use the right target architecture
        envfile.write('CFLAGS="-march=+{}"\n'.format(cpu_features))
        envfile.write('CXXFLAGS="-march=+{}"\n'.format(cpu_features))

        # Make OpenBLAS support lots of threads.
        envfile.write("OPENBLAS_NUM_THREADS=128\n")

    docker_build_cmd = ["docker", "build", "-t", "musl-builder", ".travis/"]
    subprocess.check_call(docker_build_cmd)

    musl_cmd = (
        "docker run "
        "--env-file env.list "
        "--rm -it -v "
        "{pwd}:/home/rust/src "
        "musl-builder".format(pwd=os.getcwd(), cpu_features=cpu_features)
    ).split(
        " "
    )

    cargo_cmd = ["cargo", "build", "--verbose", "--release", "--features={}".format(features)]

    print(" ".join(musl_cmd + cargo_cmd))
    subprocess.check_call(musl_cmd + cargo_cmd)

    build_dir = os.path.join(build_dir, cpu_features)

    try:
        os.makedirs(build_dir)
    except FileExistsError:
        pass

    for fname in pathlib.Path(".").glob("target/x86_64-unknown-linux-musl/release/libsbr*"):
        shutil.copy(fname, os.path.join(build_dir, fname.name))


if __name__ == "__main__":

    for cpu_features in ("sse", "avx", "avx2")[1:]:
        print("Building for {}...".format(cpu_features))
        build(features="openblas", cpu_features=cpu_features)
