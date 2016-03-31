# FROM jupyter/scipy-notebook
FROM jupyter/datascience-notebook
USER jovyan

# Add permanent pip/conda installs, data files, other user libs here
# e.g., RUN pip install jupyter_dashboards
RUN conda install bokeh 
# RUN $CONDA_DIR/envs/python2/bin/pip install pandas-summary

USER root

# Add permanent apt-get installs and other root commands here
# e.g., RUN apt-get install npm nodejs
