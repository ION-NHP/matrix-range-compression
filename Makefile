SHELL := /usr/bin/fish

.PHONY: test publish

test:
	pyenv init - | source && pyenv local system (pyenv versions --bare) && poetry run tox

publish:
	poetry publish --build
