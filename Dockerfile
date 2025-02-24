FROM runpod/base:0.6.2-cuda12.4.1

COPY requirements.txt /requirements.txt
RUN pip3 install -r requirements.txt
RUN python3 -m pip3 show accelerate
ADD src .
# Start the container
RUN python3 script/download_weights
CMD ["python3", "-u", "rp_handler.py"]