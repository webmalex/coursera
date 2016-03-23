FROM jupyter/scipy-notebook
USER jovyan

# Add permanent pip/conda installs, data files, other user libs here
# e.g., RUN pip install jupyter_dashboards

USER root

# Add permanent apt-get installs and other root commands here
# e.g., RUN apt-get install npm nodejs
