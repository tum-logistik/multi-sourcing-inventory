FROM python:3.8
ADD *.py /
ADD *.py /mount/
ADD requirements.txt /
ADD gurobi.lic /opt/gurobi/
ADD gurobi.lic /opt/gurobi950/

RUN apt-get update -y
RUN yes Y | apt install iputils-ping
RUN python3 -m pip install gurobipy
# RUN python3 -m pip install -i https://pypi.gurobi.com gurobipy
RUN pip install -r /requirements.txt
# CMD ["python3"]
