FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

MAINTAINER Brian Schmidt "6666331+schmidtbri@users.noreply.github.com"

WORKDIR ./service

COPY ./regression_model ./app
COPY ./Makefile ./Makefile
COPY ./requirements.txt ./requirements.txt

RUN make dependencies

ENV APP_MODULE=rest_model_service.main:app
