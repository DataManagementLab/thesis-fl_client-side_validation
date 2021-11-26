from cliva_fl.utils import freivalds_rounds
from matplotlib import pyplot as plt
from pathlib import Path

plot_path = Path('thesis_plots/concept/freivald_scalability.png')

fig, ax = plt.subplots()

x = list(range(1, 401))
y = list(map(lambda x: freivalds_rounds(x, 0.99), x))
ax.plot(x, y)
ax.set_title("99% correctness guarantee with Freivalds' Algorithm")
ax.set_xlabel('number of layers')
ax.set_ylabel('number freivald rounds')

fig.savefig(plot_path)
plt.close(fig)
