from openbarnes import BarnesMaze
import sys

if len(sys.argv) != 2:
    print('Usage: python run.py <path to video>')
    sys.exit(1)

maze = BarnesMaze(sys.argv[1])
img = maze.detect()
maze.detect_correct()
maze.scan()
maze.plot()
