FROM runpod/base:0.6.2-cuda12.4.1

WORKDIR /

RUN pip install -r requirements.txt
RUN mkdir -p checkpoints/controlnet-union-sdxl
RUN wget https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/resolve/main/config_promax.json -O checkpoints/controlnet-union-sdxl/config.json
RUN wget https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/resolve/main/diffusion_pytorch_model_promax.safetensors -O checkpoints/controlnet-union-sdxl/diffusion_pytorch_model.json
ADD src .
# Start the container
CMD ["python3", "-u", "rp_handler.py"]
