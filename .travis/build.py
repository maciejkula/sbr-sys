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

    cargo_cmd = ["cargo", "build", "--verbose", "--release", "--features={}".format(features)]
    subprocess.check_call(cargo_cmd)

    try:
        os.makedirs(build_dir)
    except FileExistsError:
        pass

    for fname in ("libsbr_sys.a", "libsbr_sys.dylib"):
        dest_name = "darwin_{}_{}".format(cpu_features, fname)
        dest_location = os.path.join(build_dir, dest_name)
        shutil.copy(os.path.join("target/release/", fname), dest_location)

        with zipfile.ZipFile(dest_location + ".zip", "w") as archive:
            archive.write(dest_location, fname)

        os.unlink(dest_location)


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

    docker_build_cmd = [
        "docker", "build", "-t", "manylinux-builder", "-f", ".travis/Dockerfile", "."
    ]
    subprocess.check_call(docker_build_cmd)

    manylinux_cmd = (
        "docker run "
        "--name {container_name} "
        "--env-file env.list "
        "-it "
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
    except FileExistsError:
        pass

    container_dir = "/target/release/"

    for fname in ("libsbr_sys.a", "libsbr_sys.so"):
        local_filename = "linux_{}_{}".format(cpu_features, fname)
        local_dest = os.path.join(build_dir, local_filename)

        container_fname = os.path.join(container_dir, fname)
        subprocess.check_call(
            ["docker", "cp", "{}:{}".format(container_name, container_fname), local_dest]
        )

        with zipfile.ZipFile(local_dest + ".zip", "w") as archive:
            archive.write(local_dest, fname)

        os.unlink(local_dest)


def compress_binaries(archive_name, build_dir="build"):

    shutil.make_archive(archive_name, "zip", build_dir)


if __name__ == "__main__":

    if platform.system() == "Linux":
        for cpu_features in ("sse", "avx", "avx2"):
            print("Building for {}...".format(cpu_features))
            build_linux(features="openblas", cpu_features=cpu_features)
    else:
        for cpu_features in ("sse", "avx")[:1]:
            build_osx(features="openblas", cpu_features=cpu_features)
