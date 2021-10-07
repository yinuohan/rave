os.chdir('/Users/yinuo/Documents/Github/rave')
from rave import *

image = Image(np.zeros([10, 10]), np.ones([3, 3]))
image.add_flip()