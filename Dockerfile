FROM runpod/base:0.6.2-cuda12.4.1

WORKDIR /
RUN pip install -r requirements.txt

# Start the container
CMD ["python3", "-u", "rp_handler.py"]