# FROM jupyter/scipy-notebook
FROM jupyter/datascience-notebook
USER jovyan

# Add permanent pip/conda installs, data files, other user libs here
# e.g., RUN pip install jupyter_dashboards
RUN conda install bokeh 
# RUN $CONDA_DIR/envs/python2/bin/pip install pandas-summary
COPY jupyter_notebook_config.py ~/.jupyter/jupyter_notebook_config.py
RUN mkdir -p ~/.local/share/jupyter
RUN pip install https://github.com/ipython-contrib/IPython-notebook-extensions/archive/master.zip --user

RUN pip install xgboost
ENV PIP2 $CONDA_DIR/envs/python2/bin/pip

RUN $PIP2 install pybrain
RUN pip install git+https://github.com/pybrain/pybrain.git
RUN pip install blaze 
RUN pip install boto 
RUN conda install netCDF4
#- RUN $PIP2 install pyside

USER root
EXPOSE 8000

# Add permanent apt-get installs and other root commands here
# e.g., RUN apt-get install npm nodejs
#- RUN apt-get install python3-pyside
