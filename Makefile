main:
	
restart:
	pinata restart
	docker-compose rm
	docker-compose up -d

slide-run:
	docker run --rm -it -p 8000:8000 -v `pwd`:/home/jovyan/work --name slides nb jupyter nbconvert slide.ipynb --to slides --post serve

slide-exec:
	docker exec -it nb jupyter nbconvert slide.ipynb --to slides --post serve

py-server:
	python -m http.server
	# python -m http.server 8001
