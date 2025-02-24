FROM runpod/base:0.6.2-cuda12.4.1

COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
ADD src .
# Start the container
RUN python3 script/download_weights.py
CMD ["python3", "-u", "rp_handler.py"]