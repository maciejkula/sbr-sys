import argparse
import os
import shutil
import subprocess
import uuid
import platform
import zipfile


def build_osx(build_dir="build", features="", cpu_features="avx2"):

    # Set rustflags
    os.environ["RUSTFLAGS"] = "-C target-feature=+{}".format(cpu_features)

    # Make OpenBLAS compilation use the right target architecture
    os.environ["CFLAGS"] = "-m{}".format(cpu_features)
    os.environ["CXXFLAGS"] = "-m{}".format(cpu_features)

    # Make OpenBLAS support lots of threads.
    os.environ["OPENBLAS_NUM_THREADS"] = "128"
    os.environ["NUM_THREADS"] = "128"

    # Make OpenBLAS detect architectures
    os.environ["DYNAMIC_ARCH"] = "1"
    os.environ["BINARY"] = "64"
    os.environ["USE_OPENMP"] = "0"

    # Allow multithreaded calls
    os.environ["NO_AFFINITY"] = "1"

    cargo_cmd = ["cargo", "build", "--verbose", "--release", "--features={}".format(features)]
    subprocess.check_call(cargo_cmd)

    try:
        os.makedirs(build_dir)
    except Exception:
        pass

    for fname in ("libsbr_sys.a", "libsbr_sys.dylib"):
        dest_name = "darwin_{}_{}".format(cpu_features, fname)
        dest_location = os.path.join(build_dir, dest_name)
        shutil.copy(os.path.join("target/release/", fname), dest_location)

        with zipfile.ZipFile(dest_location + ".zip", "w") as archive:
            archive.write(dest_location, fname)

        os.unlink(dest_location)


def build_windows(build_dir="build", features="", cpu_features="avx2"):

    # Set rustflags
    os.environ["RUSTFLAGS"] = "-C target-feature=+{}".format(cpu_features)

    # Make OpenBLAS compilation use the right target architecture
    os.environ["CFLAGS"] = "-m{}".format(cpu_features)
    os.environ["CXXFLAGS"] = "-m{}".format(cpu_features)

    # Make OpenBLAS support lots of threads.
    os.environ["OPENBLAS_NUM_THREADS"] = "128"
    os.environ["NUM_THREADS"] = "128"

    # Make OpenBLAS detect architectures
    os.environ["DYNAMIC_ARCH"] = "1"
    os.environ["BINARY"] = "64"
    os.environ["USE_OPENMP"] = "0"

    # Allow multithreaded calls
    os.environ["NO_AFFINITY"] = "1"

    cargo_cmd = ["cargo", "build", --"target", os.environ["TARGET"],
                 "--verbose", "--release", "--features={}".format(features)]
    subprocess.check_call(cargo_cmd)


def build_linux(build_dir="build", features="", cpu_features="avx2"):

    # Rewrite Cargo.toml to enable lto
    with open("Cargo.toml", "r") as cargo_toml:
        contents = cargo_toml.read()

    with open("Cargo.toml", "w") as cargo_toml:
        cargo_toml.write(contents.replace("lto = false", "lto = true"))

    with open("Cargo.toml", "r") as cargo_toml:
        print("Amended Cargo.toml:")
        print(cargo_toml.read())

    with open("env.list", "w") as envfile:
        envfile.write(
            "RUSTFLAGS=-C target-feature=+{cpu_features}\n".format(cpu_features=cpu_features)
        )

        # Make OpenBLAS compilation use the right target architecture
        envfile.write("CFLAGS=-m{}\n".format(cpu_features))
        envfile.write("CXXFLAGS=-m{}\n".format(cpu_features))

        # Make OpenBLAS support lots of threads and other flags.
        envfile.write("OPENBLAS_NUM_THREADS=128\n")
        envfile.write("NUM_THREADS=128\n")

        # Make OpenBLAS detect architectures
        envfile.write("DYNAMIC_ARCH=1\n")
        envfile.write("BINARY=64\n")
        envfile.write("USE_OPENMP=0\n")

        # Allow multithreaded calls
        envfile.write("NO_AFFINITY=1\n")

    container_name = str(uuid.uuid4())

    docker_build_cmd = [
        "docker", "build", "-t", "manylinux-builder", "-f", ".travis/Dockerfile", "."
    ]
    subprocess.check_call(docker_build_cmd)

    manylinux_cmd = (
        "docker run "
        "--name {container_name} "
        "--env-file env.list "
        "manylinux-builder".format(
            pwd=os.getcwd(), cpu_features=cpu_features, container_name=container_name
        )
    ).split(
        " "
    )

    cargo_cmd = ["cargo", "build", "--verbose", "--release", "--features={}".format(features)]

    subprocess.check_call(manylinux_cmd + cargo_cmd)

    try:
        os.makedirs(build_dir)
    # Can't use FileExistsError on Py27
    except Exception:
        pass

    container_dir = "/target/release/"

    for fname in ("libsbr_sys.a", "libsbr_sys.so"):
        local_filename = "linux_{}_{}".format(cpu_features, fname)
        local_dest = os.path.join(build_dir, local_filename)

        container_fname = os.path.join(container_dir, fname)
        subprocess.check_call(
            ["docker", "cp", "{}:{}".format(container_name, container_fname), local_dest]
        )

        # Strip the binary
        subprocess.check_call(["strip", "-g", local_dest])

        with zipfile.ZipFile(local_dest + ".zip", "w") as archive:
            archive.write(local_dest, fname)

        os.unlink(local_dest)


def compress_binaries(archive_name, build_dir="build"):

    shutil.make_archive(archive_name, "zip", build_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("cpu_features", type=str)

    args = parser.parse_args()

    if platform.system() == "Linux":
        build_linux(features="openblas", cpu_features=args.cpu_features)
    elif platform.system() == "Windows":
        build_windows(features="openblas", cpu_features=args.cpu_features)
    else:
        # There are a bunch of performance problems in OpenBLAS on OSX:
        # https://github.com/xianyi/OpenBLAS/issues/533.
        # Upgrade to v0.3.0 is blocked by https://github.com/xianyi/OpenBLAS/issues/1586.
        # So we use accelerate on OSX.
        build_osx(features="accelerate", cpu_features=args.cpu_features)
