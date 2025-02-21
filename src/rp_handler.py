import runpod

from runpod.serverless.utils import rp_cleanup, rp_upload
from runpod.serverless.utils.rp_validator import validate
import os
from runner import Predictor
import base64

INPUT_SCHEMA = {
    'image_path': {
        'type': str,
        'required': True
    },
    'mask_image_path': {
        'type': str,
        'required': True
    },
    'num_images': {
        'type': int,
        'required': False,
        'default': 1,
        'constraints': lambda num_outputs: num_outputs in range(1, 4)
    },
    'num_steps': {
        'type': int,
        'required': False,
        'default': 50,
        'constraints': lambda num_inference_steps: num_inference_steps in range(1, 500)
    },
    'guidance_scale': {
        'type': float,
        'required': False,
        'default': 5.0,
        'constraints': lambda guidance_scale: 0 <= guidance_scale <= 20
    },
    'controlnet_scale': {
        'type': float,
        'required': False,
        'default': 0.9,
        'constraints': lambda guidance_scale: 0 <= guidance_scale <= 1
    },
    'strength': {
        'type': float,
        'required': False,
        'default': 0.7,
        'constraints': lambda guidance_scale: 0 <= guidance_scale <= 1
    },
    'grow_mask_by': {
        'type': int,
        'required': False,
        'default': 33,
        'constraints': lambda grow_mask_by: grow_mask_by in range(1, 100)
    },
    'seed': {
        'type': int,
        'required': False,
        'default': int.from_bytes(os.urandom(2), "big")
    }
}

model = Predictor()
model.setup()


def save_and_upload_images(images, job_id):
    os.makedirs(f"/{job_id}", exist_ok=True)
    image_urls = []
    for index, image in enumerate(images):
        image_path = os.path.join(f"/{job_id}", f"{index}.png")
        image.save(image_path)

        if os.environ.get('BUCKET_ENDPOINT_URL', False):
            image_url = rp_upload.upload_image(job_id, image_path)
            image_urls.append(image_url)
        else:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(
                    image_file.read()).decode("utf-8")
                image_urls.append(f"data:image/png;base64,{image_data}")

    rp_cleanup.clean([f"/{job_id}"])
    return image_urls


def handler(job):
    job_input = job['input']

    # -------------------------------- Validation -------------------------------- #
    validated_input = validate(job_input, INPUT_SCHEMA)
    if 'errors' in validated_input:
        return {"errors": validated_input['errors']}

    valid_input = validated_input['validated_input']

    images = model.predict(
        image_path=valid_input['image_path'],
        mask_image_path=valid_input['mask_image_path'],
        num_images=valid_input['num_images'],
        num_steps=valid_input['num_steps'],
        guidance_scale=valid_input['guidance_scale'],
        controlnet_scale=valid_input['controlnet_scale'],
        strength=valid_input['strength'],
        grow_mask_by=valid_input['grow_mask_by'],
        seed=valid_input['seed']
    )
    image_urls = save_and_upload_images(images, job['id'])
    # Remove downloaded input objects
    # rp_cleanup.clean(['input_objects'])
    results = {
        "images": image_urls
    }
    return results


if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
