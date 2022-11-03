FROM python:3
ADD *.py /
ADD *.py /mount/
ADD requirements.txt /
ADD gurobi.lic /opt/gurobi/
ADD gurobi.lic /opt/gurobi950/

RUN apt-get update -y
RUN yes Y | apt install iputils-ping
RUN python3 -m pip install gurobipy==9.5.2
# RUN python3 -m pip install -i https://pypi.gurobi.com gurobipy
RUN pip install -r /requirements.txt
# CMD ["python3"]
