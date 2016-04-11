main:

run:
	docker run --rm -it -p 8000:8000 -v `pwd`:/home/jovyan/work --name slides nb jupyter nbconvert slide.ipynb --to slides --post serve

exec:
	docker exec -it nb jupyter nbconvert slide.ipynb --to slides --post serve

py:
	python -m http.server
	# python -m http.server 8001
