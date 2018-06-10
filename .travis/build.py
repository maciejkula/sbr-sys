import os
import shutil
import subprocess
import uuid
import platform


def build_osx(build_dir="build", features="", cpu_features="avx2"):

    # Set rustflags
    os.environ["RUSTFLAGS"] = "-C target-feature=+{}".format(cpu_features)

    # Make OpenBLAS compilation use the right target architecture
    os.environ["CFLAGS"] = "-m{}".format(cpu_features)
    os.environ["CXXFLAGS"] = "-m{}".format(cpu_features)

    # Make OpenBLAS support lots of threads.
    os.environ["OPENBLAS_NUM_THREADS"] = "128"
    os.environ["NUM_THREADS"] = "128"

    cargo_cmd = ["cargo", "build", "--verbose", "--release", "--features={}".format(features)]
    subprocess.check_call(cargo_cmd)

    build_dir = os.path.join(build_dir, "darwin", cpu_features)

    try:
        os.makedirs(build_dir)
    except FileExistsError:
        pass

    for fname in ("libsbr_sys.a", "libsbr_sys.dylib"):
        shutil.copy(os.path.join("target/release/", fname), build_dir)


def build_linux(build_dir="build", features="", cpu_features="avx2"):

    with open("env.list", "w") as envfile:
        envfile.write(
            "RUSTFLAGS=-C target-feature=+{cpu_features}\n".format(cpu_features=cpu_features)
        )

        # Make OpenBLAS compilation use the right target architecture
        envfile.write("CFLAGS=-m{}\n".format(cpu_features))
        envfile.write("CXXFLAGS=-m{}\n".format(cpu_features))

        # Make OpenBLAS support lots of threads.
        envfile.write("OPENBLAS_NUM_THREADS=128\n")
        envfile.write("NUM_THREADS=128\n")

    container_name = str(uuid.uuid4())

    docker_build_cmd = ["docker", "build", "-t", "musl-builder", "-f", ".travis/Dockerfile", "."]
    subprocess.check_call(docker_build_cmd)

    musl_cmd = (
        "docker run "
        "--name {container_name} "
        "--env-file env.list "
        "-it "
        "musl-builder".format(
            pwd=os.getcwd(), cpu_features=cpu_features, container_name=container_name
        )
    ).split(
        " "
    )

    cargo_cmd = ["cargo", "build", "--verbose", "--release", "--features={}".format(features)]

    subprocess.check_call(musl_cmd + cargo_cmd)

    build_dir = os.path.join(build_dir, "linux", cpu_features)

    try:
        os.makedirs(build_dir)
    except FileExistsError:
        pass

    container_dir = "/home/rust/src/target/x86_64-unknown-linux-musl/release/"

    fname = os.path.join(container_dir, "libsbr_sys.a")
    subprocess.check_call(
        [
            "docker",
            "cp",
            "{}:{}".format(container_name, fname),
            os.path.join(build_dir, fname.split("/")[-1]),
        ]
    )


def compress_binaries(archive_name, build_dir="build"):

    shutil.make_archive(archive_name, "zip", build_dir)


if __name__ == "__main__":

    if platform.system() == "Linux":
        for cpu_features in ("sse", "avx", "avx2")[:1]:
            print("Building for {}...".format(cpu_features))
            build_linux(features="openblas", cpu_features=cpu_features)
            compress_binaries("libsbr_linux")
    else:
        for cpu_features in ("sse", "avx")[:1]:
            build_osx(features="openblas", cpu_features=cpu_features)
            compress_binaries("libsbr_darwin")
