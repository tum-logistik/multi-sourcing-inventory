FROM python:3
ADD *.py /
ADD *.py /mount/
ADD requirements.txt /
ADD gurobi.lic /opt/gurobi/
ADD gurobi.lic /opt/gurobi950/

RUN apt-get update -y
RUN apt install iputils-ping -Y
RUN python3 -m pip install gurobipy
RUN pip install -r /requirements.txt
# CMD ["python3"]
