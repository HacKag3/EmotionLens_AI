# Controlla se siamo in GitHub Actions
IS_GITHUB_ACTIONS := $(findstring true,$(CI))
# Comandi in base al SO
ifeq ($(OS),Windows_NT)
    R_PYTHON = venv\Scripts\python.exe
    R_PIP = venv\Scripts\pip.exe
    SET_PYTHONPATH = set PYTHONPATH=. &
	RM = del /s /q
else
    R_PYTHON = ./venv/bin/python
    R_PIP = ./venv/bin/pip
    SET_PYTHONPATH = PYTHONPATH=.
	RM = rm -rf
endif
# Sovrascrittura se in GitHub Actions
ifeq ($(IS_GITHUB_ACTIONS),true)
    R_PYTHON = python
    R_PIP = pip
endif
# Crea venv se non esiste e non su github
ifeq ($(IS_GITHUB_ACTIONS),)
venv:
	python -m venv venv
endif

# Scarica tutte le dipendeze per preparare l'ambiente
install:
	$(R_PIP) install --upgrade pip
	$(R_PIP) install -r requirements.txt
	@echo Installazione delle dipendenze terminata.
# Analisi Statistica del codice sorgente
lint:
	$(SET_PYTHONPATH) $(R_PYTHON) -m pylint --disable=R,C src/*.py tests/*.py
	@echo Linting complete.
# Unit test
test:
	$(SET_PYTHONPATH) $(R_PYTHON) -m pytest -vv --cov=src tests/
	@echo Testing complete.

# PACKAGING
# 	poetry add $(cat requirements.txt) reso compatibile con ogni SO
init-poetry:
	$(R_PIP) install poetry
	-poetry init --no-interaction
	$(R_PYTHON) -c "import os; [os.system(f'poetry add {line.strip()}') for line in open('requirements.txt') if line.strip()]"
	@if not exist README.md echo "# EmotionLens AI" > README.md

build:
	$(R_PYTHON) -m build
	@echo "Build complete. Check dist/ directory."

clean_build:
ifeq ($(OS),Windows_NT)
	@if exist dist rmdir /s /q dist
	@if exist build rmdir /s /q build
	@if exist *.egg-info rmdir /s /q *.egg-info
	@if exist .pytest_cache rmdir /s /q .pytest_cache
	@if exist __pycache__ rmdir /s /q __pycache__
	@if exist .coverage del /q .coverage
else
	$(RM) dist/ build/ *.egg-info .pytest_cache .coverage __pycache__
endif

ifeq ($(OS),Windows_NT)
    DOCKER_PWD := $(subst \,/,${CURDIR})
else
    DOCKER_PWD := $(CURDIR)
endif
docker:
	docker build -t emotionslens .
docker_run: docker
	docker run --rm --name $(name) \
		-v "$(DOCKER_PWD)/persistent_data:/app/results" \
		emotionslens \
		python $(script)
	docker image prune -f
docker_fer2013:
	make docker_run script=src/fer2013.py name=fer2013
docker_fed2013:
	make docker_run script=src/fed2013.py name=fed2013
docker_emotionlens: docker
	docker run --rm --name emotionslens \
		--device /dev/video0:/dev/video0 \
		-v "$(DOCKER_PWD)/persistent_data:/app/results" \
		-p 7860:7860 \
		emotionslens \
		python src/emotionlens_ai.py
	docker image prune -f