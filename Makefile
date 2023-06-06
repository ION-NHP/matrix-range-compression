SHELL := /usr/bin/fish

.PHONY: test publish

test:
	pyenv init - | source && poetry run tox

publish:
	poetry publish --build
