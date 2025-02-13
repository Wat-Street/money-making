import docker
import os

client = docker.from_env()

def build_docker_image(name, dockerfile_path):
    try:
        image, logs = client.images.build(path=dockerfile_path, tag=name)
        return image
    except Exception as e:
        raise RuntimeError(f"Error building Docker image: {e}")


def run_docker_container(image_name, command=None):
    try:
        container = client.containers.run(
            image_name,
            command=command
        )
        return container
    except Exception as e:
        raise RuntimeError(f"Error running Docker container: {e}")
    

def stop_docker_container(image_name):
    try:
        container = client.containers.get(image_name)
        container.stop()
    except docker.errors.NotFound:
        print(f"Container '{image_name}' not found")
    except Exception as e:
        raise RuntimeError(f"Error stopping Docker container: {e}")
