all:
	python main.py

experiments: frame detect

frame:
	python3 frames.py

clip:
	python3 clip.py

detect:
	python3 google_detect.py

clean:
	rm -rf ./frames/*.jpg ./bbox/*.jpg